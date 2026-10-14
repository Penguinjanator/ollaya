"""Time every model on one Ollaya server the way a client sees it: the triage preset (five questions)
on one short message through `/api/decide`, one request at a time, HTTP included.

    python -m ollaya_convert.bench_latency --url http://127.0.0.1:11435 --machine rtx-5090 --device cuda \
        --out ../results/runs/2026-10-01-latency-rtx5090-cuda.json [--models winnow:e4b kev:4b] [--warm 5 --n 20]

Without `--models` it times every model the server has (routers left out: they answer with the
model they route to). Each model: one load, `--warm` untimed requests, then `--n` timed ones; the
result keeps their median, 90th percentile and minimum, the device and precision `/api/ps`
reports, and the server's own load time. The model is unloaded before the next one, so only one
model is in memory at a time.

Every request carries a different message (STATES, in turn), as a client's traffic would: a
model that caches by text (clm keeps every projection) answers a repeated request from its cache,
which would time the cache rather than the model.

With `--memory` (Linux, the server on the same machine) each result also has the memory the model takes:
`vram_mb`, the rise of the GPU's used memory over the idle level just before the load, at its highest
while the model is loaded (`nvidia-smi`, sampled every 100 ms; nothing else should use the GPU meanwhile),
and `ram_peak_mb` / `ram_mb`, the runner process's peak and final resident memory (`VmHWM` and `VmRSS`).
`size_mb` is the server's own estimate from `/api/ps`.

Standard library only, so it runs on Windows, macOS and a bare Linux box without the convert
environment. The output is one JSON file in the format of `results/README.md`.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request

# Short customer messages of similar length, one per request in turn. The first is the one the
# website and the docs use for the triage preset.
STATES = [
    "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor.",
    "I was charged twice for the same invoice this month. Please reverse the duplicate charge as soon as possible.",
    "The app crashes every time I open the reports tab on my phone. It started after yesterday's update.",
    "Can you tell me whether the annual plan includes priority support, and how much it costs per seat?",
    "Please cancel my subscription at the end of this billing period. We are moving to another tool.",
    "Our API calls have been returning 502 errors for the last hour and our checkout page is down.",
    "Hi, I just wanted to say the new dashboard is great. Quick question: can I export it to PDF?",
    "This is the third time my package was left at the wrong address. I want my money back now.",
    "How do I add a new admin to our workspace? The settings page does not show the invite button.",
    "Your last invoice lists 40 seats but we only have 25 users. Can someone fix the amount, please?",
    "I forgot my password and the reset email never comes. I have a client meeting in twenty minutes.",
    "We would like to downgrade from the business plan to the starter plan next month. Is that possible?",
    "The integration with our CRM stopped syncing contacts on Monday. Nothing in our setup has changed.",
    "I received a damaged blender. The glass jar is cracked. Can you send a replacement or refund it?",
    "Is there a discount for nonprofits? We are a small charity and the current price is too high for us.",
    "Your support team closed my ticket without answering. I am really frustrated and considering leaving.",
    "Could you confirm that my data is stored in the EU? Our legal team needs this before we renew.",
    "The mobile app keeps logging me out every few minutes. It's making the product unusable for me.",
    "I was promised a callback two days ago and nobody called. Please escalate this to a manager today.",
    "What payment methods do you accept? We can't use credit cards and would prefer a bank transfer.",
    "After the price increase our bill doubled without any notice. Explain this or we will cancel.",
    "Search results are very slow since this morning, about ten seconds per query. Is there an outage?",
    "I'd like a copy of all invoices from last year for our accountant. Where can I download them?",
    "The product works fine, but the onboarding emails are too frequent. How do I turn them off?",
    "My account was suspended for no reason and I can't access any of my projects. Fix this immediately.",
    "We are evaluating your tool against two competitors. Do you offer a free trial for teams of fifty?",
    "The shipment tracking has not updated in six days. Is my order lost? I need it before Friday.",
    "Please refund the add-on I bought by mistake an hour ago. I never used it and don't need it.",
    "Two-factor authentication codes from your app are always rejected. I'm locked out of my account.",
    "Thanks for fixing the sync issue so fast. Everything works again and the team is happy with it.",
]
STATE = STATES[0]

# The triage preset with its score question on six levels, for a model whose head takes no other
# (arbiter): the same five questions of the same types.
TRIAGE6 = {
    "intent": {"type": "choice", "instructions": "What does the customer want in `message`?", "criteria": {
        "refund": "money returned or a duplicate charge reversed",
        "technical_help": "a bug, outage or integration problem",
        "billing_question": "a question about an invoice, plan or payment method",
        "information": "general information, pricing or how-to",
        "cancellation": "wants to cancel or downgrade",
        "other": "none of the other options fits"}},
    "is_urgent": {"type": "noul", "instructions": "Does `message` communicate time pressure or a deadline?"},
    "frustration": {"type": "score", "instructions": "How frustrated does the customer sound in `message`?", "criteria": [
        "calm and neutral", "polite but concerned", "mildly annoyed", "clearly annoyed", "angry",
        "very angry or using strong language"]},
    "refund_requested": {"type": "noul", "instructions": "Does the customer ask for money back?"},
    "churn_risk": {"type": "noul", "instructions": "Does `message` suggest the customer may leave for a competitor or cancel?"},
}


def call(url, path, body=None, timeout=1800):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, method="GET" if body is None else "POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


_turn = [0]


def decide(url, model, preset="triage", questions=None):
    state = STATES[_turn[0] % len(STATES)]
    _turn[0] += 1
    body = {"model": model, "state": state, "keep_alive": "30m"}
    if preset:
        body["preset"] = preset
    if questions:
        body["questions"] = questions
    t = time.perf_counter()
    out = call(url, "/api/decide", body)
    return (time.perf_counter() - t) * 1000, out


def unload(url, model):
    try:
        call(url, "/api/decide", {"model": model, "keep_alive": 0})
    except (urllib.error.URLError, OSError):
        pass


def run_text(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def machine_info(name):
    """Best effort: the CPU, the GPUs and the OS of this machine (the server's, when local)."""
    cpu = platform.processor()
    if os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    elif platform.system() == "Darwin":
        cpu = run_text(["sysctl", "-n", "machdep.cpu.brand_string"]) or cpu
    elif platform.system() == "Windows":
        cpu = run_text(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]) or cpu
    gpus = []
    smi = run_text(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    for line in smi.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            gpus.append({"name": parts[0], "driver": parts[1], "memory": parts[2]})
    return {"id": name, "cpu": cpu, "threads": os.cpu_count(), "gpus": gpus,
            "os": "%s %s" % (platform.system(), platform.release())}


class GpuMemory:
    """The GPU's used memory in MiB, sampled every 100 ms by one long-running nvidia-smi."""

    def __init__(self):
        self.samples = []
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "100"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                self.samples.append((time.monotonic(), float(line.split(",")[0])))
            except ValueError:
                pass

    def window(self, t0, t1):
        return [mb for t, mb in self.samples if t0 <= t <= t1]

    def stop(self):
        self.proc.terminate()


def runner_pids():
    """The Ollaya runner processes on this machine (`<exe> runner ...`)."""
    pids = set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            argv = open("/proc/%s/cmdline" % pid, "rb").read().split(b"\0")
        except OSError:
            continue
        if len(argv) > 1 and argv[1] == b"runner" and b"ollaya" in os.path.basename(argv[0]):
            pids.add(int(pid))
    return pids


def proc_memory(pid):
    """(VmHWM, VmRSS) of a process in MiB."""
    out = {}
    try:
        for line in open("/proc/%d/status" % pid, encoding="utf-8"):
            if line.startswith(("VmHWM:", "VmRSS:")):
                key, value = line.split(":", 1)
                out[key] = int(value.split()[0]) / 1024
    except OSError:
        pass
    return out.get("VmHWM"), out.get("VmRSS")


def time_model(url, model, warm, n, gpu=None, memory=False):
    entry = {"model": model, "questions": "triage"}
    unload(url, model)  # time a cold load even if something loaded the model before
    if memory:
        time.sleep(2)  # let the previous runner exit and its memory go
    before = runner_pids() if memory else set()
    if gpu is not None:
        t_idle = time.monotonic()
        time.sleep(1)
        idle = gpu.window(t_idle, time.monotonic())
    t_start = time.monotonic()
    try:
        # The triage preset; for a model that rejects it, the same questions with a six-level score
        # (arbiter); for a model with its own fixed questions (qwen3guard), those.
        modes = [("triage", None, "triage"), (None, TRIAGE6, "triage, frustration on six levels"), (None, None, None)]
        for i, (preset, questions, label) in enumerate(modes):
            try:
                load_ms, first = decide(url, model, preset, questions)
                break
            except urllib.error.HTTPError as e:
                if e.code != 422 or i == len(modes) - 1:
                    raise
        if i:  # the rejected tries loaded the model: load it again to time a cold load
            unload(url, model)
            time.sleep(1)
            load_ms, first = decide(url, model, preset, questions)
        entry["questions"] = label or "builtin (%d)" % len(first.get("answers", {}))
        entry["load_ms"] = round(first.get("load_duration", 0) / 1e6, 1)
        entry["first_request_ms"] = round(load_ms, 1)
        for _ in range(warm):
            decide(url, model, preset, questions)
        times, evals = [], []
        for _ in range(n):
            ms, out = decide(url, model, preset, questions)
            times.append(ms)
            evals.append(out.get("eval_duration", 0) / 1e6)
        times.sort()
        entry.update({
            "answered_by": first.get("model"),
            "p50_ms": round(statistics.median(times), 1),
            "p90_ms": round(times[min(len(times) - 1, int(round(0.9 * (len(times) - 1))))], 1),
            "min_ms": round(times[0], 1),
            "eval_p50_ms": round(statistics.median(evals), 1),
            "input_tokens": first.get("usage", {}).get("input_tokens"),
        })
        ps = {m["name"]: m for m in call(url, "/api/ps").get("models", [])}
        running = ps.get(first.get("model")) or next(iter(ps.values()), {})
        entry["device"] = running.get("device")
        entry["precision"] = running.get("details", {}).get("quantization_level")
        entry["digest"] = running.get("digest")
        if memory:
            entry["size_mb"] = round(running.get("size", 0) / 2**20) if running.get("size") else None
            new = sorted(runner_pids() - before)
            if len(new) == 1:
                hwm, rss = proc_memory(new[0])
                entry["ram_peak_mb"] = round(hwm) if hwm is not None else None
                entry["ram_mb"] = round(rss) if rss is not None else None
            if gpu is not None:
                busy = gpu.window(t_start, time.monotonic())
                if idle and busy:
                    entry["vram_mb"] = round(max(busy) - min(idle))
                    entry["gpu_idle_mb"] = round(min(idle))
    except urllib.error.HTTPError as e:
        entry["error"] = "HTTP %d: %s" % (e.code, e.read()[:300].decode(errors="replace"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        entry["error"] = str(e)[:300]
    unload(url, model)
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:11435")
    ap.add_argument("--machine", required=True, help="the machine's name in results/machines.json")
    ap.add_argument("--device", required=True, help="what the server runs on: cuda, vulkan, metal or cpu")
    ap.add_argument("--models", nargs="*", help="model tags (default: every model the server has)")
    ap.add_argument("--skip", nargs="*", default=[], help="model tags to leave out")
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--resume", action="store_true", help="keep the models --out already has without an error")
    ap.add_argument("--memory", action="store_true", help="also measure each model's GPU and CPU memory (Linux, local server)")
    a = ap.parse_args()
    url = a.url.rstrip("/")

    version = call(url, "/api/version").get("version")
    models = a.models
    if not models:
        tags = call(url, "/api/tags")["models"]
        # One entry per model: aliases (`kev:latest` = `kev:4b`) share a digest.
        seen, models = set(), []
        for m in sorted(tags, key=lambda m: (m["name"].endswith(":latest"), m["name"])):
            if m["details"].get("format") == "router" or m["digest"] in seen:
                continue
            seen.add(m["digest"])
            models.append(m["name"])
    models = [m for m in models if m not in a.skip]

    result = {
        "schema": 1,
        "suite": "latency-triage",
        "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ollaya": version,
        "machine": a.machine,
        "machine_detail": machine_info(a.machine),
        "device": a.device,
        "protocol": {"preset": "triage", "questions": 5, "states": len(STATES), "warm": a.warm, "n": a.n,
                     "client": "one request at a time over HTTP to /api/decide, timed by the client; "
                               "each request a different short message, in turn"},
        "note": a.note,
        "results": [],
    }
    if a.memory:
        result["protocol"]["memory"] = ("vram_mb: peak rise of the GPU's used memory over its idle level before the load "
                                        "(nvidia-smi, 100 ms); ram_peak_mb and ram_mb: the runner's VmHWM and VmRSS")
    gpu = GpuMemory() if a.memory and a.device.startswith("cuda") else None
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.resume and os.path.exists(a.out):
        kept = [e for e in json.load(open(a.out, encoding="utf-8"))["results"] if "error" not in e]
        result["results"] = kept
        models = [m for m in models if m not in {e["model"] for e in kept}]
    for model in models:
        entry = time_model(url, model, a.warm, a.n, gpu, a.memory)
        result["results"].append(entry)
        print("%-24s %s" % (model, entry.get("error") or "p50 %.1f ms on %s" % (entry["p50_ms"], entry["device"])),
              flush=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
    if gpu is not None:
        gpu.stop()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
