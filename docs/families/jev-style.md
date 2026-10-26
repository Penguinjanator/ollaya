# jev-style (`jev-style-v1`)

**chaoliangUNSW/Jev-Style-2B-Decision-v3** is a Qwen3.5-2B fine-tune (Apache-2.0), a non-generative
typed-decision model. Unlike `jevk5` / `llm-logits`, it is **not** a last-token label-logit readout:
it uses a plain-text "render v2" input (no chat template, no system prompt, no answer letters) and a
**verdict readout** — `logit(" yes") - logit(" no")` at a per-option ` ->` slot — read under
**non-causal block attention**. The author ships its own GGUF scorer (`jev-score-v2`); this family
reproduces that scorer on Ollaya's llama.cpp runner. The author publishes GGUF builds in
**chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF**. Ollaya ships the Q8_0 file as `jev-style:2b` (also
`jev-style:latest`) and the Q4_K_M file as `jev-style:2b-q4_k_m`.

| | |
|---|---|
| Repo / commit | `chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF@283b9eb7903aeb184b16d49ab8f38b256f4afcb3` |
| Files | `Jev-Style-2B-Decision-v3-Q8_0.gguf` 2.01 GB, sha256 `f4826516125022362f4617e16d28b0e5f95b10d053bac18b59f49f27c89c83bb` (`2b`); `Jev-Style-2B-Decision-v3-Q4_K_M.gguf` 1.27 GB, sha256 `a0140af1710ccff8c00e7e911b1ec0697467a023fcdb3ffde6408dd474d5e124` (`2b-q4_k_m`); both as the repo's `manifest.json` lists them |
| Tokenizer | the author's `tokenizer/tokenizer.json` from the same commit, sha256 `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` (`readout_config.json` `tokenizer_sha256`), shipped as the package's tokenizer layer |
| Source model | `chaoliangUNSW/Jev-Style-2B-Decision-v3@ee807b23807f32813a1df33d83be4d276f2f488a` (Qwen3.5-2B fine-tune, Apache-2.0) |
| Reference | the model's own runtime `jev_style_decision_gguf.py` (shared core) and `jev_score_v2.cpp`, in the GGUF repo |
| Readout tokens | ` yes` = 9542, ` no` = 874, ` ->` = 1411 (`readout_config.json` `slot_tokens`; checked against the GGUF and the tokenizer at load) |
| Temperature | one global `0.8278650620942867` (`readout_config.json` `temperatures.global`), applied by Ollaya's calibration like every family |
| Architecture | `qwen35` (Gated-DeltaNet + attention); block-causal in the 6 full-attention layers |

**Why Q8_0 is `2b`.** The author's card puts Q8_0 at 99.7 % top-1 agreement with the PyTorch fp32
reference (1,000 rows) and 43/43 on its long fixture, and Q4_K_M at 95.7 % and 42/43, "noticeably noisier
on long inputs". Q4_K_M is the smaller download (1.27 GB against 2.01 GB).

## Engine: non-causal block attention on llama.cpp

Where `jevk5`/`llm-logits` read a single causal context, jev-style needs the non-causal, block-wise
decode its scorer uses. The context is jev-score-v2's (`decision.json` `llama`, checked at load):

- **One non-causal context.** `attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL`, `n_ctx = 32768`,
  `n_batch = n_ubatch = 2048`, `n_seq_max = 2`, `kv_unified = true` and 512 output slots, never toggled:
  a non-causal ubatch sees all earlier cache cells of its sequence plus itself, with no causal mask inside
  the block — the block-causal layout of training. The Gated-DeltaNet layers stay recurrent. A
  `decision.json` with another attention type, an `n_ubatch` below 2048 or fewer than 2 sequences is
  refused at load (each would score wrongly or trip a llama.cpp assert).
- **Flash attention** as jev-score-v2 sets it: off on the CPU (the author measured a precision loss there),
  llama.cpp's auto on a GPU.
- **One decode per renderer block** (never merged, never re-split), each `<= n_ubatch`.
- **State computed once.** The state decodes into sequence 0 once; each question does
  `seq_cp(0 -> 1)`, decodes its question-side block(s) into sequence 1 with logits requested **only**
  at the ` ->` slots, reads `logit(" yes") - logit(" no")` at each, then `seq_rm(1)`. With the unified
  cache, `seq_cp` only tags the existing cells, so a question costs its own tokens.
- **Non-finite scores** (NaN or infinite) are an error, never probabilities, as in the author's runtime.

See [llm-logits.md](llm-logits.md) for the shared llama.cpp runtime and
[decisions/0003-llama-cpp-runtime.md](../decisions/0003-llama-cpp-runtime.md).

## Tokenizer

The render is tokenised with the author's `tokenizer.json` (Hugging Face `tokenizers` 0.23.2, the
version the author pins), not with the GGUF's vocabulary: llama.cpp's Qwen3.5 pre-tokenizer splits
some scripts differently (the shared set's Hindi state is 26 tokens with llama.cpp's and 40 with the
author's tokenizer), and the model was trained and measured on the author's tokenisation. As in the
author's `TextEncoder`: no BOS/EOS, and special tokens in user text are split, so `<|im_end|>` in a state
is text. The GGUF's vocabulary only indexes the logits; the readout tokens are checked in both.

## Prompt (render v2, port of `jev_style_decision_gguf.py`)

```text
state prefix   State:\n<state>\n\n
short form     Question [<type>]: <question>\nOptions:\n
(q+opts+slots   - <option 1> ->\n ... - <option K> ->\n
 <= 2048 tok)
overflow form  Question [<type>]: <question>\nOptions:\n
(otherwise)    Option 1: <option 1>\n ... Option K: <option K>\n            (the catalogue)
               Judge each numbered option in the complete catalogue above:\n
               Option 1 ->\n ... Option K ->\n                               (the rubric)
```

- `<type>` is `choice`, `score` or `noul`. The state is `json.dumps`ed (`ensure_ascii=False`) unless
  it is already a string. Option bodies (`render_options`): choice `"{name}"` or `"{name}: {desc}"`;
  score `"level {i}: {desc}"`; noul `"false: …"` / `"true: …"` (default
  `"no, the statement does not hold"` / `"yes, the statement holds"`).
- Questions go through Ollaya's shared parser, so a choice given as a list of labels (duplicates
  collapse onto their first position) and noul keys in any case (`"True"`) are accepted as for every
  other family; the author's `make_question` takes a list of labels too.
- Attention blocks: the state is cut into 2,048-token blocks; the short form is one block; in the
  overflow form the catalogue and the rubric are each cut into 2,048-token blocks.

## Score and probabilities

Option `k`'s score is `logit(" yes") - logit(" no")` at its ` ->` slot, in float32. Probabilities are
`softmax(scores / T)` in canonical option order (choice: criteria keys; score: levels; noul:
`false, true`), with the one global temperature — applied by Ollaya's calibration, as for every
family, so `calibration.json` carries `T` once per type.

## Reference implementation

- `crates/ollaya-decision/src/jev_style.rs` ports the render v2 layout, verdict slots and blocks;
  `crates/ollaya-runner/src/llama/mod.rs` `run_jev_style` ports the block decode and readout of
  `jev_score_v2.cpp`.
- `convert/ollaya_convert/families/jev_style/ref.py` is the Python port of the render.
  `convert/tests/test_jev_style_render.py` renders the shared set and this layout's own cases with it and
  with the author's `Renderer` on the pinned `tokenizer.json`: identical ids, slots, blocks and option
  names, including the overflow form and both budget edges.
- `convert/ollaya_convert/families/jev_style/goldens.py` writes the goldens: the author's `Renderer`, scored
  by the author's `jev_score_v2.cpp`, unchanged, compiled against the b11146 headers and linked to
  ggml-org's stock b11146 release libraries, the ones the runner loads (`load_backends.c` loads their CPU
  backend variants before `main`, which those builds need). ADR 0003: a `seq_cp` split needs its own
  reference, and this is it.

## Parity

Goldens: `goldens.py` on Ollaya's shared request set (121 requests) plus this layout's own cases: a state
of several attention blocks (6,214 tokens), a catalogue overflow of 150 options, and inputs of exactly
25,600 and 25,601 tokens. A rejected request is followed by its `#valid` questions. The gate,
`parity_jev_style`, requires identical token ids, verdict slots, attention blocks and option names, every
decision the same and every verdict score within 1e-3 of jev-score-v2's.

Measured with `parity_run` on 2026-10-26: the CPU on `community-epyc-milan` (an AMD EPYC Milan cloud VM,
8 vCPUs, the AVX2 CPU backend of the b11146 libraries), Ollaya at `cf1fa64`
(`results/runs/2026-10-26-parity-communityepycmilan-cpu.json`); CUDA on `rtx-5090` (the CUDA 13.4 backend of
the b11146 libraries, goldens from jev-score-v2 on the same GPU), Ollaya at `ece7118`
(`results/runs/2026-10-26-parity-rtx5090-cuda.json`):

| Tag | File | Device | Questions | Decisions | Verdict scores max | Probabilities max |
|---|---|---|---|---|---|---|
| `2b` | Q8_0 | CPU (x86-64) | 582 | 582/582 | 4.4e-16 | 0 |
| `2b-q4_k_m` | Q4_K_M | CPU (x86-64) | 582 | 582/582 | 4.4e-16 | 0 |
| `2b` | Q8_0 | CUDA (RTX 5090) | 582 | 582/582 | 4.4e-16 | 0 |
| `2b-q4_k_m` | Q4_K_M | CUDA (RTX 5090) | 582 | 582/582 | 4.4e-16 | 0 |

- **Identical encodings.** 141 records: 124 scored requests, 582 questions, among them 2 in the overflow
  form (up to 150 options), 4 with a state of several blocks, and one input of exactly 25,600 tokens.
- **Rejected by both (17).** 13 edge cases whose structured `instructions` the author's runtime refuses,
  3 decoder cases whose score legend is an object, and the 25,601-token input.
- **Same scores.** On the same libraries and device the runner's scores are jev-score-v2's to the last
  bit; the 4.4e-16 is the float64 rounding of the stored scores.
- **Not run yet.** Metal and Windows; latency.

What the files are worth against the author's PyTorch fp32 reference is the author's measurement (the
GGUF card, 1,000 development rows): Q8_0 99.7 % top-1 agreement, maximum |Δp| 0.033, 43/43 on the long
fixture; Q4_K_M 95.7 %, 0.346, 42/43.

## Limits

- **Context.** The complete input (state + question + options + readout) is at most **25,600** tokens;
  a longer request is rejected (400), never truncated.
- **Score levels.** A score question takes 2 to 10 levels (the author's limit); other counts are a 400.
- **Instructions.** A question's `instructions` must be a non-empty string (the author's limit); the
  structured instructions other families `json.dumps` are a 400 here.
- **Options.** No option cap: more options than fit one 2,048-token block switch to the catalogue form.
- **Cost.** The state is decoded once per request; each question then costs only its own tokens.
- **Runtime.** Stock llama.cpp, Ollama or LM Studio can load the GGUF but would decode causally and cannot
  produce these scores; the decisions need this engine (or the author's runtime).

## Training data

From the author's [model card](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3) ("Training",
"Training data and licences"), at the commit above:

- **Base and run.** Qwen/Qwen3.5-2B (revision `15852e8c`) with the vision tower and multi-token-prediction
  head removed; a full fine-tune of one epoch, 458 steps, 60,032,377 tokens (181,449 rows), a reduced data
  pool, not the planned full recipe. EMA weights, chosen by a pre-declared rule; one global temperature
  fitted on 2,000 separate calibration rows.
- **Mixture.** 58 sources; 77.7 % of the tokens English and 17.0 % Chinese (MASSIVE adds nine more
  languages). The largest components: model-written typed-decision workflows (34,008 rows, 19.2 %),
  intents and yes/no QA (MASSIVE, CLINC150, BoolQ), knowledge and reasoning (ARC, CommonsenseQA, MedMCQA,
  GSM8K, MBPP and others), and jailbreak and prompt-injection sets.
- **Not zero-shot on 14 benchmarks.** The pool has the train splits of BANKING77, CLINC150 (+OOS), SGD,
  HellaSwag, GSM8K, ARC-Easy and ARC-Challenge, and format-imitating data for SATA-Bench, BRIGHT, NLI4CT,
  CRUXEval, CLadder, PhishNChips and BBH. No test split was used; JevBench, tweet_topic, fin_topic and
  daily_dialog are evaluation-only.
- **Restrictive or unclear terms**, kept in the pool: research-only jailbreak prompts (the jailbreak_llms
  collection, jackhhao/jailbreak-classification), HellaSwag (its original repository is blocked after a
  DMCA notice; only ActivityNet-caption items were used), and share-alike sources (CC BY-SA: SQuAD v2,
  BoolQ, DROP, SNLI, ARC, SGD, ShARC, TempReason).
- **Outputs of other models.** OpenAI GPT and Anthropic Claude models designed the typed-decision
  workflows and Claude models labelled them; a Claude model wrote 1,243 jailbreak and toxicity prompts;
  GPT models wrote two hard-case families and paraphrased Mac goal items; WANLI, GPTeacher and HelpSteer2
  carry model-written text. The card notes that the providers' terms may restrict how models trained on
  such outputs are used. No Jev or other TypeSafe outputs were used.

Ollaya does not redistribute the GGUF or the tokenizer; both are fetched from the author's repository at
the pinned commit.
