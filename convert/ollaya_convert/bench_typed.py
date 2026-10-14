"""typed-decisions (LocalLLaMA/typed-decisions, Apache-2.0), test split, run against any TypeSafe-compatible
`/v1/systemone` server: 400 realistic JSON states from four workflows, five typed questions each (2,000
questions: 600 choice, 600 noul, 800 score), with gold labels and gold distributions from several annotators.

    uv run python -m ollaya_convert.bench_typed export --out typed-decisions-test.jsonl   # once, needs pyarrow
    python -m ollaya_convert.bench_typed run --url http://127.0.0.1:11435 --model winnow:e4b \
        --data typed-decisions-test.jsonl --out runs/winnow-e4b
    python -m ollaya_convert.bench_typed report runs/*

Each request is one row as the dataset has it: its `state` and all five `questions`, one request at a time,
timed by the client. A rejected request (any HTTP error) counts as wrong on every question of the row, as in
Bespoke Labs' public benchmark. Scoring:

  * the predicted option is the argmax of the returned probabilities (choice and score: the `probabilities`
    object; noul: `noul` against 0.5), compared with the gold `label`;
  * calibration: ten-bin ECE of the top probability, pooled over all questions; multiclass Brier and
    cross-entropy against the gold distribution.

`run` writes `rows.jsonl` (the request id, the answers as returned, the latency) and resumes where it stopped.
Standard library only, so it runs without the convert environment.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TYPES = ("choice", "noul", "score")


def export(a):
    import pyarrow.parquet as pq

    from .cases import TYPED_DECISIONS

    rows = pq.read_table(TYPED_DECISIONS).to_pylist()
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "workflow": r["workflow"], "state": json.loads(r["state"]),
                                "questions": json.loads(r["questions"]), "gold": json.loads(r["gold"])},
                               ensure_ascii=False) + "\n")
    print("wrote %d rows to %s" % (len(rows), a.out))


def post(url, body, timeout=1800):
    req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer local", "Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def run(a):
    data = [json.loads(line) for line in open(a.data, encoding="utf-8") if line.strip()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "rows.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["id"] for line in open(path, encoding="utf-8") if line.strip()}
    # One untimed request first, so the model load is not in the first row's latency.
    try:
        post(a.url, {"model": a.model, "state": data[0]["state"], "questions": data[0]["questions"]})
    except (urllib.error.URLError, OSError):
        pass
    with open(path, "a", encoding="utf-8") as f:
        for i, row in enumerate(data):
            if row["id"] in done:
                continue
            rec = {"id": row["id"]}
            t = time.perf_counter()
            try:
                resp = post(a.url, {"model": a.model, "state": row["state"], "questions": row["questions"]})
                rec["elapsed_seconds"] = time.perf_counter() - t
                rec["model"] = resp.get("model")
                rec["answers"] = resp.get("answers")
                rec["input_tokens"] = (resp.get("usage") or {}).get("input_tokens")
            except urllib.error.HTTPError as e:
                rec["elapsed_seconds"] = time.perf_counter() - t
                rec["error"] = "HTTP %d: %s" % (e.code, e.read()[:300].decode(errors="replace"))
            except (urllib.error.URLError, OSError, ValueError) as e:
                rec["elapsed_seconds"] = time.perf_counter() - t
                rec["error"] = str(e)[:300]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 100 == 0:
                print("  %d/%d rows" % (i + 1, len(data)), flush=True)


def option_keys(qdef):
    if qdef["type"] == "choice":
        return list(qdef["criteria"])
    if qdef["type"] == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(qdef["criteria"]))]


def answer_probs(qdef, answer):
    """The returned distribution over `option_keys(qdef)`, or None when the answer is missing or malformed."""
    if not isinstance(answer, dict):
        return None
    keys = option_keys(qdef)
    if qdef["type"] == "noul":
        p = answer.get("noul")
        return None if not isinstance(p, (int, float)) else [1.0 - float(p), float(p)]
    probs = answer.get("probabilities")
    if not isinstance(probs, dict):
        return None
    v = [float(probs.get(k, 0.0)) for k in keys]
    s = sum(v)
    return None if s <= 0 else [x / s for x in v]


def score_rows(data, rows):
    """One item per question: type, workflow, row index, correct, top probability, Brier, soft CE."""
    by_id = {r["id"]: r for r in rows}
    items = []
    for ri, row in enumerate(data):
        rec = by_id.get(row["id"])
        for qid, qdef in row["questions"].items():
            gold = row["gold"].get(qid)
            if gold is None:
                continue
            keys = option_keys(qdef)
            label = str(gold["label"])
            gp = gold.get("probabilities") or {}
            gv = [float(gp.get(k, 0.0)) for k in keys]
            gs = sum(gv) or 1.0
            gv = [x / gs for x in gv]
            p = None if rec is None or "error" in rec else answer_probs(qdef, (rec.get("answers") or {}).get(qid))
            item = {"type": qdef["type"], "workflow": row["workflow"], "row": ri}
            if p is None:
                item.update({"correct": 0, "answered": False})
            else:
                top = max(range(len(p)), key=lambda i: p[i])
                onehot = [1.0 if k == label else 0.0 for k in keys]
                item.update({
                    "correct": int(keys[top] == label), "answered": True, "top": p[top],
                    "brier": sum((pi - oi) ** 2 for pi, oi in zip(p, onehot)),
                    "soft_ce": -sum(g * math.log(max(pi, 1e-12)) for g, pi in zip(gv, p)),
                })
            items.append(item)
    return items


def ece(pairs, bins=10):
    pairs = list(pairs)
    total, err = len(pairs), 0.0
    for b in range(bins):
        sel = [(p, c) for p, c in pairs if (b / bins < p <= (b + 1) / bins) or (b == 0 and p == 0)]
        if sel:
            err += len(sel) / total * abs(sum(c for _, c in sel) / len(sel) - sum(p for p, _ in sel) / len(sel))
    return err


def summarize(run_dir, data):
    rows = [json.loads(line) for line in open(Path(run_dir) / "rows.jsonl", encoding="utf-8") if line.strip()]
    items = score_rows(data, rows)
    answered = [x for x in items if x["answered"]]
    times = sorted(r["elapsed_seconds"] for r in rows if "error" not in r)
    acc = lambda sel: statistics.mean(x["correct"] for x in sel) if sel else None  # noqa: E731
    models = sorted({r.get("model") for r in rows if r.get("model")})
    return {
        "run": str(run_dir), "rows": len(rows), "questions": len(items),
        "errors": sum(1 for r in rows if "error" in r), "answered_by": models,
        "accuracy": acc(items),
        "by_type": {t: acc([x for x in items if x["type"] == t]) for t in TYPES},
        "by_workflow": {w: acc([x for x in items if x["workflow"] == w]) for w in sorted({x["workflow"] for x in items})},
        "ece": ece((x["top"], x["correct"]) for x in answered) if answered else None,
        "brier": statistics.mean(x["brier"] for x in answered) if answered else None,
        "soft_ce": statistics.mean(x["soft_ce"] for x in answered) if answered else None,
        "median_ms": 1000 * statistics.median(times) if times else None,
        "p95_ms": 1000 * times[math.ceil(0.95 * len(times)) - 1] if times else None,
    }


def report(a):
    data = [json.loads(line) for line in open(a.data, encoding="utf-8") if line.strip()]
    out = [summarize(d, data) for d in a.runs]
    if a.json:
        json.dump(out, open(a.json, "w", encoding="utf-8"), indent=1)
    f = lambda x, d=3: "n/a" if x is None else ("%.*f" % (d, x))  # noqa: E731
    print("| Run | Rows | Errors | Accuracy | Choice | Noul | Score | ECE | Brier | Median |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(out, key=lambda r: -(r["accuracy"] or 0)):
        print("| %s | %d | %d | %s | %s | %s | %s | %s | %s | %s ms |" % (
            r["run"], r["rows"], r["errors"], f(r["accuracy"]), f(r["by_type"]["choice"]), f(r["by_type"]["noul"]),
            f(r["by_type"]["score"]), f(r["ece"]), f(r["brier"]), f(r["median_ms"], 0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--data", required=True, help="the JSONL that `export` writes")
    r.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("runs", nargs="+")
    p.add_argument("--data", required=True)
    p.add_argument("--json")
    a = ap.parse_args()
    {"export": export, "run": run, "report": report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
