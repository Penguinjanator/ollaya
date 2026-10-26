"""Goldens for `jev-style-v1` from the author's own runtime: the render of `jev_style_decision_gguf.py` (its
`Renderer` on the pinned `tokenizer/tokenizer.json`) and the scores of `jev-score-v2` (`jev_score_v2.cpp`,
unchanged, built against the llama.cpp build Ollaya ships, b11146). ADR 0003: a `seq_cp` split needs its own
reference, and this is it.

    python -m ollaya_convert.families.jev_style.goldens out/jev-style-2b-q8_0 \
        --upstream <the GGUF repo at the pinned commit> --scorer <build>/jev-score-v2 \
        --gguf <Jev-Style-2B-Decision-v3-Q8_0.gguf> --requests ollaya-shared-requests.jsonl [--device cpu|CUDA0]

Build the scorer with the author's `build_jev_score.sh` against a b11146 checkout, or compile it against the
b11146 headers and link the stock release libraries (`scripts/llama-cpp.sh linux-amd64`): the same libllama
the runner loads. Needs the author's pinned `tokenizers` (requirements.txt) and numpy.

The requests are Ollaya's shared set (the wire form, as an HTTP client sends it) plus this layout's own cases:
a state of several attention blocks, a catalogue overflow of 150 options (several catalogue blocks), an input of
exactly 25,600 tokens and one of 25,601 (rejected).

A wire question becomes the author's `{"t", "ins", "crit"}` with Ollaya's request semantics (laya's), which the
author's `make_question` shares: a choice given as a list of labels is `{label: None}` (duplicates collapse onto
their first position), and noul keys are lower-cased (the last spelling wins). Everything else, including the
author's limits (2..10 score levels, a non-empty string instruction), is the author's code.

Writes <export dir>/goldens-<device>.jsonl, one line per request:
    {"id", "state", "questions", "error": "invalid", "message"}          the author's runtime rejects it
    {"id", "state", "questions", "state_tokens",
     "rows": [{"qid", "names", "ids", "slots", "blocks", "catalogue_overflow",
               "logits": [[yes, no], ...], "scores": [...], "probabilities": [...]}]}
A rejected request is followed by "<id>#valid" with the questions the author's runtime accepts on their own.
Every accepted render is also checked against this family's `ref.py` before it is written.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np

from . import ref

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "out")


def load_upstream(path):
    spec = importlib.util.spec_from_file_location("jev_style_decision_gguf",
                                                  os.path.join(path, "jev_style_decision_gguf.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def author_question(q):
    """A wire question as the author's {"t", "ins", "crit"}, with Ollaya's (laya's) request semantics."""
    if not isinstance(q, dict):
        return q
    t, crit = q.get("type"), q.get("criteria")
    if t == "choice" and isinstance(crit, list) and crit and all(isinstance(c, str) for c in crit):
        crit = dict.fromkeys(crit)
    elif t == "noul" and isinstance(crit, dict):
        crit = {str(k).lower(): v for k, v in crit.items()}
    return {"t": t, "ins": q.get("instructions"), "crit": crit}


def device_class(device):
    return "cpu" if device == "cpu" else "cuda" if device.lower().startswith("cuda") else device.lower()


class Scorer:
    """The author's jev-score-v2, a JSON-lines server on stdin/stdout."""

    def __init__(self, binary, gguf, device, threads):
        args = [binary, "--model", gguf, "--ngl", "0" if device == "cpu" else "999", "--threads", str(threads)]
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.info = json.loads(self.p.stdout.readline())
        if not self.info.get("ready"):
            raise SystemExit("jev-score-v2 did not start: %s" % self.info)

    def ask(self, req):
        self.p.stdin.write(json.dumps(req) + "\n")
        self.p.stdin.flush()
        resp = json.loads(self.p.stdout.readline())
        if "error" in resp:
            raise RuntimeError("jev-score-v2: %s" % resp["error"])
        return resp

    def close(self):
        try:
            self.p.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            self.p.stdin.flush()
        except BrokenPipeError:
            pass
        self.p.wait(timeout=60)


def softmax(scores, t):
    z = np.asarray(scores, dtype=np.float64) / t
    e = np.exp(z - z.max())
    return (e / e.sum()).tolist()


def layout_cases(renderer):
    """This layout's own requests: several state blocks, catalogue overflow, the 25,600-token edge."""
    noul = {"type": "noul", "instructions": "Did the deploy cause the checkout failures?"}
    sentence = ("The customer reports intermittent failures on the checkout page after the latest deploy; "
                "payments time out, carts empty themselves and the support queue keeps growing. ")
    out = [("jev/multi_block_state", sentence * 220, {
        "caused": noul,
        "team": {"type": "choice", "instructions": "Which team should take it?",
                 "criteria": {"payments": "card processing and refunds", "web": "the checkout front end",
                              "infra": None}},
        "severity": {"type": "score", "instructions": "How severe is it?",
                     "criteria": ["minor", "degraded", "major", "outage"]}})]
    options = {"intent_%03d" % i: "the customer asks about item %d of the catalogue: billing, shipping, returns, "
               "warranty or account changes for that item" % i for i in range(150)}
    out.append(("jev/overflow_150", "I want to return the blender I bought last week and get my money back.",
                {"intent": {"type": "choice", "instructions": "Which intent matches the message?",
                            "criteria": options}}))
    # Digits are one token each in this tokenizer (and never merge), so the state can hit an exact length.
    q = author_question(noul)
    for want in (ref.CONTEXT_LIMIT, ref.CONTEXT_LIMIT + 1):
        m = 0
        for _ in range(8):
            state = "Ticket log follows. " + "7" * m
            p = renderer.pieces(state, q)
            n = len(p["prefix"]) + len(p["head"]) + sum(len(renderer.dash) + len(o) + 1 + len(renderer.newline)
                                                        for o in p["opts"])  # the short form's length
            if n == want:
                break
            m += want - n
        else:
            raise SystemExit("could not build a %d-token input" % want)
        out.append(("jev/budget_%d" % want, state, {"q": noul}))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_dir", help="out/jev-style-<slug>, with the decision.json export_llama wrote")
    ap.add_argument("--upstream", required=True, help="the GGUF repo at the pinned commit (jev_style_decision_gguf.py, "
                                                      "readout_config.json, tokenizer/tokenizer.json)")
    ap.add_argument("--scorer", required=True, help="jev-score-v2 built against llama.cpp b11146")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--requests", required=True, help="Ollaya's shared request set (JSON lines)")
    ap.add_argument("--device", default="cpu", help="cpu, or CUDA0 (all layers offloaded)")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    up = load_upstream(a.upstream)
    readout = json.load(open(os.path.join(a.upstream, "readout_config.json"), encoding="utf-8"))
    enc = up.TextEncoder(os.path.join(a.upstream, "tokenizer", "tokenizer.json"))
    renderer = up.Renderer(enc, readout)
    temperature = float(readout["temperatures"]["global"])
    decision = json.load(open(os.path.join(a.export_dir, "decision.json"), encoding="utf-8"))
    yes, no, arrow = decision["yes"], decision["no"], decision["arrow"]
    assert (yes, no, arrow) == (renderer.yes, renderer.no, renderer.arrow[0]), "readout ids differ from decision.json"

    cases = [(r["id"], r["state"], r["questions"]) for r in map(json.loads, open(a.requests, encoding="utf-8"))
             if r.get("id")]
    cases += layout_cases(renderer)

    scorer = Scorer(a.scorer, a.gguf, a.device, a.threads)
    name = "goldens-%s" % device_class(a.device)
    with open(os.path.join(a.export_dir, name + ".meta.json"), "w", encoding="utf-8") as f:
        json.dump({"reference": "jev_style_decision_gguf.py Renderer + jev-score-v2 (jev_score_v2.cpp, unchanged)",
                   "scorer": {k: scorer.info.get(k) for k in ("protocol", "flash_attn", "gpu_offload", "n_ctx",
                                                              "n_ubatch", "n_seq_max", "n_outputs_max", "cache_type",
                                                              "n_threads", "desc", "system_info")},
                   "device": a.device, "gguf": os.path.basename(a.gguf), "temperature": temperature,
                   "requests": os.path.basename(a.requests)}, f, indent=1)

    def render_all(state, questions):
        """The author's render of every question (all before any scoring, as score_many), checked against ref.py."""
        if not isinstance(questions, dict) or not questions:
            raise up.QuestionError("questions must be a non-empty object")
        prefix = renderer.prefix_ids(state)
        out = []
        for qid, q in questions.items():
            r = renderer.render(state, up.make_question(author_question(q)), prefix=prefix)
            mine = ref.render(state, q, enc, arrow, prefix=prefix)
            assert (mine.ids, mine.slots, mine.blocks, mine.names) == (r.ids, r.slots, [tuple(b) for b in r.blocks],
                                                                        r.names), "ref.py differs on %s" % qid
            out.append((qid, r))
        return prefix, out

    def record(cid, state, questions):
        prefix, rendered = render_all(state, questions)
        p = len(prefix)
        resp = scorer.ask({"prefix": prefix, "prefix_blocks": rendered[0][1].state_blocks,
                           "questions": [{"ids": r.ids[p:], "blocks": r.question_blocks, "slots": r.slots,
                                          "rows": [yes, no]} for _, r in rendered],
                           "keep_prefix": False, "return_logits": True})
        rows = []
        for (qid, r), res in zip(rendered, resp["results"]):
            if not res["finite"]:
                raise RuntimeError("%s/%s: non-finite score from jev-score-v2" % (cid, qid))
            rows.append({"qid": qid, "names": r.names, "ids": r.ids, "slots": r.slots,
                         "blocks": [list(b) for b in r.blocks], "catalogue_overflow": r.catalogue_overflow,
                         "logits": res["logits"], "scores": res["scores"],
                         "probabilities": softmax(res["scores"], temperature)})
        return {"id": cid, "state": state, "questions": questions, "state_tokens": p, "rows": rows}

    n, t0 = 0, time.time()
    try:
        with open(os.path.join(a.export_dir, name + ".jsonl"), "w", encoding="utf-8") as f:
            for cid, state, questions in cases:
                try:
                    rec = record(cid, state, questions)
                except (up.QuestionError, up.InputBudgetError) as e:
                    for qid, q in (questions.items() if isinstance(questions, dict) else []):
                        try:  # the port must reject what the author's runtime rejects
                            ref.render(state, q, enc, arrow)
                        except (ref.QuestionError, ref.InputBudgetError):
                            break
                    else:
                        raise AssertionError("%s: the author's runtime rejects it (%s), ref.py accepts" % (cid, e))
                    f.write(json.dumps({"id": cid, "state": state, "questions": questions, "error": "invalid",
                                        "message": str(e)}, ensure_ascii=False) + "\n")
                    valid = {}
                    for qid, q in (questions.items() if isinstance(questions, dict) else []):
                        try:
                            render_all(state, {qid: q})
                            valid[qid] = q
                        except (up.QuestionError, up.InputBudgetError):
                            pass
                    if not valid:
                        continue
                    cid, rec = cid + "#valid", record(cid + "#valid", state, valid)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                print("%4d %-48s %6d tokens %6.0fs" % (n, cid, max(len(r["ids"]) for r in rec["rows"]),
                                                       time.time() - t0), flush=True)
    finally:
        scorer.close()
    print("wrote %d scored requests to %s" % (n, os.path.join(a.export_dir, name + ".jsonl")))


if __name__ == "__main__":
    sys.exit(main())
