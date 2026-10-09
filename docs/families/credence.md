# Credence

Library names: `credence:e4b` (also `credence:latest`), `credence:e4b-calibrated`, and the vision tags
`credence:e4b-vision` and `credence:e4b-calibrated-vision`. Credence v1 by Txoka is a MiCA refinement of
EldanRing's Winnow-E4B, based on Google DeepMind's Gemma 4 E4B. It uses the existing
`winnow-v1` layout and llama.cpp runner; no new inference engine is necessary.
Adapter modifications are Apache-2.0. Preserve upstream LICENSE/NOTICE and the
training data's individual attribution requirements.

## Selected artifact

The selected checkpoint is `synth05-hardreal2-clip-25-s002500-T1calibrated`:
MiCA rank32/alpha32, clip25, 5% procedural synthetic questions, 19 real sources,
2,500 optimizer steps. Training used bitsandbytes NF4; the release candidate is
Q8_0 GGUF. Checkpoint-bound readout scaling is folded into weights and softcap;
external temperature is **1**, not Winnow's original 1.2574172. No new temperature
was fitted on benchmark data. This is supervised candidate-distribution training,
not a policy-gradient RL result.

The exact byte size and SHA256 are recorded in
`convert/releases/credence-e4b.json`. Original adapter/source identities
remain recorded alongside the local export. This is the highest public-accuracy
candidate among the four Q8 exports tested; it is not a Pareto improvement over
original Winnow on all probability metrics.

## Benchmarks through Ollaya

Both columns use Q8_0, the same local Ollaya runtime and reserved benchmark rows.
Original Winnow uses its published temperature; Credence uses folded calibration
with external T1. Independent wire rounding limits reconstructed log metrics.

| Metric | Original Winnow-E4B | Credence v1 |
|---|---:|---:|
| Public macro accuracy, 13 subsets / 3,880 rows | 73.4567% | 74.1244% |
| Public pooled ECE, 10 bins | 7.0893% | 5.4789% |
| Public mean Brier | 0.380216 | 0.372403 |
| Typed accuracy, 400 states / 2,000 questions | 72.30% | 72.35% |
| Typed soft CE | 1.0623 | 1.0655 |
| Typed NLL | 0.6860 | 0.6872 |
| Typed ECE | 2.51% | 4.83% |

Public accuracy and calibration improve; typed accuracy differs by one question,
and typed probability metrics worsen. These are one-checkpoint measurements,
not significance claims. Retain per-subset numbers in
`docs/measurements/credence/`; do not substitute public pooled ECE for macro ECE.
Local latency is recorded in the artifacts, but should not be mixed with Ollaya's
RTX 4090/5090 leaderboard latency. No independent latency advantage is claimed.

## Packaging and verification

Each tag pulls the author's Q8_0 GGUF, unmodified, from
[Txoka/Credence-v1-Gemma4-E4B](https://huggingface.co/Txoka/Credence-v1-Gemma4-E4B) at
`7d5ffc84145f34a3a76f7074e8c70bf1c1efce56`: `accuracy/model-Q8_0.gguf` and `calibrated/model-Q8_0.gguf`,
8,718,469,216 bytes each, with their hashes in `convert/releases/credence-e4b.json`. The package adds the
`decision.json` and `calibration.json` that `export_llama` derives from that file, the author's NOTICE and the
license. The vision tags add Winnow-E4B's projector from EldanRing's repository (see Image input).

Parity: stock `llama-server` of the pinned build (b11146), loaded with the same GGUF on the device under test and
driven by Winnow's Python reference through the fixed evaluation plan, against Ollaya's runner. All 505 questions
(123 requests, 15 rejected): the same token ids, split points, candidates and rejections, every decision the same,
and option log-probabilities within the unchanged 1e-3.

| Tag | Device | Measured by | Decisions | Option logits max | Probabilities max |
|---|---|---|---|---|---|
| `e4b` | CPU, x86-64 Linux | contributor, 2026-10-08 | 505/505 | 7.58e-6 | 1.74e-6 |
| `e4b` | CUDA, RTX 4070 | contributor, 2026-10-08 | 505/505 | 7.55e-6 | 1.87e-6 |
| `e4b` | CUDA, RTX 4090 | Ollaya, 2026-10-09 | 505/505 | 7.62e-6 | 1.68e-6 |
| `e4b-calibrated` | CPU, x86-64 Linux | contributor, 2026-10-08 | 505/505 | 7.58e-6 | 1.78e-6 |
| `e4b-calibrated` | CUDA, RTX 4070 | contributor, 2026-10-08 | 505/505 | 7.59e-6 | 1.82e-6 |
| `e4b-calibrated` | CUDA, RTX 4090 | Ollaya, 2026-10-09 | 505/505 | 7.63e-6 | 1.81e-6 |

On the RTX 4090, the runner answers five questions in 97 ms (median of 20 requests on the parity fixtures), the
same as `winnow:e4b` (96 ms).

The contributor's reference metadata and fixture hashes are in
`docs/measurements/credence/package-verification.json`. Metal and Windows parity have not been run.

## Public-calibration variant

Alternative tag: `credence:e4b-calibrated`. This is a different checkpoint,
trained with **0% synthetic**, not simply a temperature alias of the default.
Its residual temperature (approximately 1.040857) was fitted on the fixed 1,024
training-validation questions, not on either benchmark. Packaging must preserve
the full-precision fitted value and its validation provenance.

| Metric | Accuracy-focused tag | Public-calibration tag |
|---|---:|---:|
| Public macro accuracy | 74.1244% | 73.7228% |
| Public pooled ECE | 5.4789% | 3.7426% |
| Public Brier | 0.372403 | 0.369314 |
| Typed accuracy | 72.35% | 72.20% |
| Typed soft CE | 1.0655 | 1.0480 |
| Typed NLL | 0.6872 | 0.7011 |
| Typed ECE | 4.83% | 4.63% |

“Calibrated” describes the validation-fitted release and stronger public
calibration; it does not assert uniformly better calibration, perfect
probabilities or dominance over original Winnow. Its typed ECE remains above
original Winnow's 2.51%. Both tags retain the same source publication and package
verification procedure. Exact summary/provenance artifacts use the `calibrated-`
prefix in the results directory.

## Reproduce

The goldens and the derived `decision.json` and `calibration.json` (`convert/out/winnow-credence-e4b` and
`convert/out/winnow-credence-e4b-calibrated`):

```shell
PYTHONPATH=convert python -m ollaya_convert.families.llm_common.export_llama winnow \
    --server <b11146 build>/llama-server --gguf Credence-v1-E4B-accuracy-Q8_0.gguf --slug credence-e4b \
    --repo Txoka/Credence-v1-Gemma4-E4B --revision 7d5ffc84145f34a3a76f7074e8c70bf1c1efce56 \
    --file accuracy/model-Q8_0.gguf --upstream-commit 77d14580c6732ca2f3745750c1dc1fd446d8bcee \
    --temperature 1 --td 40 --device CUDA0          # CPU: --device cpu
# e4b-calibrated: --gguf <calibrated file> --slug credence-e4b-calibrated --file calibrated/model-Q8_0.gguf
#                 --temperature 1.0408574437121012
OLLAYA_LIBRARY_PATH=<install>/lib/ollaya cargo run --release -p ollaya-runner --example parity_llama -- \
    <dir with decision.json, calibration.json and model.gguf> convert/out/winnow-credence-e4b/goldens-cuda.jsonl cuda
PYTHONPATH=convert python -m ollaya_convert.package credence
```

## Image input

The separate vision packages reuse the exact unchanged F16 projector shipped
with Winnow E4B. Winnow's model card states that vision and audio modules stayed
frozen while its language tensors were trained; Credence's MiCA runs were
text-only. No multimodal training is claimed.

| Tag | Language checkpoint | Projector |
|---|---|---|
| `credence:e4b-vision` | Same as `e4b` | Frozen Gemma4 E4B F16 |
| `credence:e4b-calibrated-vision` | Same as `e4b-calibrated` | Same projector |

Projector: Winnow-E4B's `gguf/mmproj-Winnow-E4B.gguf`, pulled from EldanRing's repository at
`734302fe5fbfeb3f21a7ece62653c9539be4aaf3`, where it was first published: 990,372,672 bytes, SHA256
`ddf46c21d7078e95338cfc22306b19b276a29a5ad089023449dd54d4b6170a51`. Credence's repository carries a byte-identical
copy (`vision/mmproj-Gemma4-E4B-F16.gguf`); the package takes the original, so a store that also holds
`winnow:e4b-vision` keeps one copy. Revisions are recorded in `convert/releases/credence-e4b.json`.

The vision tags add one projector layer to the text packages; language weights
and decision/calibration configurations are shared. Text tags remain available
without downloading a projector. Image requests use the existing Winnow runtime
path described in [Winnow image documentation](winnow.md).

Against stock llama-server b11146 with the same projector, on each device, under the unchanged 1e-3 gate:

| Tag | Device | Measured by | Image requests / decisions | Max log-probability difference |
|---|---|---|---|---|
| `e4b-vision` | CPU, x86-64 Linux | contributor, 2026-10-08 | 21 / 65 | 7.43e-6 |
| `e4b-vision` | CUDA, RTX 4090 | Ollaya, 2026-10-09 | 21 / 65 | 7.52e-6 |
| `e4b-calibrated-vision` | CPU, x86-64 Linux | contributor, 2026-10-08 | 21 / 65 | 7.62e-6 |
| `e4b-calibrated-vision` | CUDA, RTX 4090 | Ollaya, 2026-10-09 | 21 / 65 | 7.60e-6 |

These comparisons use external temperature 1 to isolate runtime parity; the calibrated package keeps its text
temperature. The generated-color, multi-image ordering, repeat-request and escaped-state fixtures test runtime
parity, not visual reasoning accuracy, and the text temperatures are not validated as image calibration.

Image parity proof is recorded in
`convert/releases/credence-vision-verification.json`. To reproduce,
start stock llama-server b11146 with the selected language GGUF and the pinned
projector (`--mmproj`), then export and check fixtures using the existing tool:

```shell
python -m ollaya_convert.families.winnow.vision_parity export http://127.0.0.1:REFERENCE_PORT /path/to/decision.json /tmp/vision.json
python -m ollaya_convert.families.winnow.vision_parity check http://127.0.0.1:RUNNER_PORT /tmp/vision.json
```

Run Ollaya's runner with the same language weights, projector and `decision.json` on the same device as the
reference; `docs/families/winnow.md` has the full commands.
