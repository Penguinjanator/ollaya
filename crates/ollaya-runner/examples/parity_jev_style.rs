//! Compare the llama.cpp engine on `jev-style-v1` (chaoliangUNSW/Jev-Style-2B-Decision-v3) against
//! golden fixtures from `ollaya_convert.families.jev_style.goldens`: the author's own render
//! (`jev_style_decision_gguf.py`) scored by the author's `jev-score-v2`, unchanged, on the same pinned
//! llama.cpp build (b11146) and device.
//!
//!     cargo run --release -p ollaya-runner --example parity_jev_style -- \
//!         <model-dir> <goldens.jsonl> [auto|cpu|cuda|cuda:<n>] [--smoke]
//!
//! `<model-dir>` holds `decision.json`, `calibration.json`, `model.gguf` (the author's GGUF, or a
//! link to it) and `tokenizer.json` (the author's `tokenizer/tokenizer.json`, the package's tokenizer
//! layer). llama.cpp is loaded from `$OLLAYA_LIBRARY_PATH/llama`, and its CUDA backend from
//! `$OLLAYA_LIBRARY_PATH/cuda_v13/libggml-cuda.so` or, failing that, `cuda_v12/libggml-cuda.so`.
//!
//! Every case is rendered first:
//! * a request the reference rejects must be rejected;
//! * per question: the token ids, the verdict slots, the attention blocks and the option names
//!   must be identical, and so must the state's token count.
//!
//! Then every question is scored:
//! * each verdict score `logit(" yes") - logit(" no")` within `SCORE_TOL` of jev-score-v2's;
//! * the calibrated probabilities (`calibration.json`, the author's one global temperature) must
//!   pick the reference's option; their difference is reported as max and p99.
//!
//! It prints one `record:` JSON line for `ollaya_convert.parity_run` before its checks.

use std::path::PathBuf;
use std::time::Instant;

use ollaya_decision::{Calibration, CalibrationFile, QType};
use ollaya_runner::llama::{Libraries, LlamaModel, Target};
use serde_json::Value;

mod smoke;

/// Largest verdict-score difference accepted: the family tolerance of every Ollaya parity check.
const SCORE_TOL: f64 = 1e-3;

fn argmax(p: &[f64]) -> usize {
    p.iter()
        .enumerate()
        .fold(
            (0, f64::NEG_INFINITY),
            |b, (i, &v)| if v > b.1 { (i, v) } else { b },
        )
        .0
}

fn percentile(sorted: &[f64], p: usize) -> f64 {
    sorted
        .get((sorted.len() * p / 100).min(sorted.len().saturating_sub(1)))
        .copied()
        .unwrap_or(0.0)
}

fn qtype(def: &Value) -> QType {
    match def["type"].as_str() {
        Some("choice") => QType::Choice,
        Some("score") => QType::Score,
        _ => QType::Noul,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (args, smoke) = smoke::args();
    let pos: Vec<&String> = args[1..].iter().filter(|a| !a.starts_with("--")).collect();
    if pos.len() < 2 {
        return Err(
            "usage: parity_jev_style <model-dir> <goldens.jsonl> [auto|cpu|cuda|cuda:<n>] [--smoke]"
                .into(),
        );
    }
    let dir = PathBuf::from(pos[0]);
    let goldens = PathBuf::from(pos[1]);
    let target = match pos.get(2).map(|s| s.as_str()).unwrap_or("auto") {
        "auto" => Target::Auto,
        "cpu" => Target::Cpu,
        "cuda" => Target::Device("CUDA0".into()),
        d => Target::Device(
            d.strip_prefix("cuda:")
                .map(|n| format!("CUDA{n}"))
                .unwrap_or_else(|| d.to_owned()),
        ),
    };
    let lib = std::env::var_os("OLLAYA_LIBRARY_PATH")
        .map(PathBuf::from)
        .ok_or("set OLLAYA_LIBRARY_PATH to an install's lib/ollaya")?;
    let cuda = ["cuda_v13", "cuda_v12"]
        .map(|pack| lib.join(pack).join(ollaya_runner::llama::CUDA_BACKEND))
        .into_iter()
        .find(|p| p.is_file());
    let libs = Libraries {
        dir: lib.join("llama"),
        cuda,
        rocm: None,
    };
    let calibration: CalibrationFile =
        serde_json::from_str(&std::fs::read_to_string(dir.join("calibration.json"))?)?;
    let calibration = Calibration::from_file(&calibration);

    let t0 = Instant::now();
    let mut model = LlamaModel::load(
        &dir.join("model.gguf"),
        &dir.join("decision.json"),
        &libs,
        &target,
        None,
    )?;
    model.load_tokenizer(&dir.join("tokenizer.json"))?;
    let dev = model.device.clone();
    println!(
        "loaded {} on {dev} in {:.1}s",
        model.precision,
        t0.elapsed().as_secs_f64()
    );

    let (mut cases, mut rejected, mut questions, mut agree) = (0usize, 0usize, 0usize, 0usize);
    let mut failures: Vec<String> = Vec::new();
    let mut score_diffs: Vec<f64> = Vec::new();
    let mut prob_diffs: Vec<f64> = Vec::new();
    let t_eval = Instant::now();
    for line in smoke::lines(&goldens, smoke)? {
        let case: Value = serde_json::from_str(&line)?;
        let id = case["id"].as_str().unwrap_or("?").to_owned();
        cases += 1;
        let render = model.jev_style_render(&case["state"], &case["questions"]);
        if case.get("error").is_some() {
            rejected += 1;
            // The engine may reject at render time or, for the context check, when it runs.
            let accepted =
                render.is_ok() && model.run_json(&case["state"], &case["questions"]).is_ok();
            if accepted {
                failures.push(format!(
                    "{id}: accepted; the reference rejects it ({})",
                    case["message"].as_str().unwrap_or("")
                ));
            }
            continue;
        }
        let (prefix, rendered) = match render {
            Ok(r) => r,
            Err(e) => {
                failures.push(format!("{id}: rejected ({e}); the reference accepts it"));
                continue;
            }
        };
        if prefix.len() as u64 != case["state_tokens"].as_u64().unwrap_or(u64::MAX) {
            failures.push(format!(
                "{id}: state {} tokens, the reference {}",
                prefix.len(),
                case["state_tokens"]
            ));
        }
        let rows = case["rows"].as_array().cloned().unwrap_or_default();
        if rows.len() != rendered.len() {
            failures.push(format!(
                "{id}: {} questions, the reference {}",
                rendered.len(),
                rows.len()
            ));
            continue;
        }
        let mut same = true;
        for ((qid, r), want) in rendered.iter().zip(&rows) {
            let mut bad = Vec::new();
            if qid != want["qid"].as_str().unwrap_or("") {
                bad.push(format!("order ({})", want["qid"]));
            }
            let w: Vec<u32> = serde_json::from_value(want["ids"].clone()).unwrap_or_default();
            if r.ids != w {
                let at = r.ids.iter().zip(&w).take_while(|(a, b)| a == b).count();
                bad.push(format!(
                    "token ids differ at {at} ({} vs {} tokens)",
                    r.ids.len(),
                    w.len()
                ));
            }
            let slots: Vec<usize> =
                serde_json::from_value(want["slots"].clone()).unwrap_or_default();
            if r.slots != slots {
                bad.push("slots".into());
            }
            let blocks: Vec<(usize, usize)> =
                serde_json::from_value(want["blocks"].clone()).unwrap_or_default();
            if r.blocks != blocks {
                bad.push(format!("blocks {:?} vs {blocks:?}", r.blocks));
            }
            let names: Vec<String> =
                serde_json::from_value(want["names"].clone()).unwrap_or_default();
            if r.names != names {
                bad.push("option names".into());
            }
            if r.catalogue_overflow != want["catalogue_overflow"].as_bool().unwrap_or(false) {
                bad.push("catalogue overflow".into());
            }
            if !bad.is_empty() {
                same = false;
                failures.push(format!("{id}/{qid}: {}", bad.join(", ")));
            }
        }
        if !same {
            continue;
        }
        let out = match model.run_json(&case["state"], &case["questions"]) {
            Ok(o) => o,
            Err(e) => {
                failures.push(format!("{id}: scoring failed ({e})"));
                continue;
            }
        };
        for (((qid, _), got), want) in rendered.iter().zip(&out.questions).zip(&rows) {
            questions += 1;
            let want_scores: Vec<f64> = serde_json::from_value(want["scores"].clone())?;
            let got = &got.logits;
            if got.len() != want_scores.len() {
                failures.push(format!(
                    "{id}/{qid}: {} scores, the reference {}",
                    got.len(),
                    want_scores.len()
                ));
                continue;
            }
            let diff = got
                .iter()
                .zip(&want_scores)
                .map(|(&a, &b)| (f64::from(a) - b).abs())
                .fold(0.0, f64::max);
            score_diffs.push(diff);
            if diff > SCORE_TOL || diff.is_nan() {
                failures.push(format!("{id}/{qid}: verdict scores differ by {diff:.3e}"));
            }
            let t = qtype(&case["questions"][qid]);
            let p_got = calibration.probabilities(t, got, prefix.len());
            let want32: Vec<f32> = want_scores.iter().map(|&x| x as f32).collect();
            let p_want = calibration.probabilities(t, &want32, prefix.len());
            let pd = p_got
                .iter()
                .zip(&p_want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            prob_diffs.push(pd);
            if argmax(&p_got) == argmax(&p_want) {
                agree += 1;
            } else {
                failures.push(format!(
                    "{id}/{qid}: the decision differs (p diff {pd:.3e})"
                ));
            }
        }
    }
    let eval_s = t_eval.elapsed().as_secs_f64();
    score_diffs.sort_by(f64::total_cmp);
    prob_diffs.sort_by(f64::total_cmp);
    println!(
        "{cases} cases ({rejected} rejected), {questions} questions on {dev} in {eval_s:.1}s: \
         decisions {agree}/{questions} ({:.2}%), verdict scores max {:.3e} p99 {:.3e}, \
         probabilities max {:.3e} p99 {:.3e}",
        100.0 * agree as f64 / questions.max(1) as f64,
        score_diffs.last().copied().unwrap_or(0.0),
        percentile(&score_diffs, 99),
        prob_diffs.last().copied().unwrap_or(0.0),
        percentile(&prob_diffs, 99),
    );
    // One JSON line for `ollaya_convert.parity_run`, which records it in results/runs/.
    println!(
        "record: {}",
        serde_json::json!({
            "questions": questions,
            "decisions_same": agree,
            "max_logit_diff": score_diffs.last().copied().unwrap_or(0.0),
            "max_prob_diff": prob_diffs.last().copied().unwrap_or(0.0),
            "p99_prob_diff": percentile(&prob_diffs, 99),
        })
    );
    // An empty or truncated goldens file (an export that stopped early) must not pass the gate.
    if questions == 0 {
        return Err(format!("{}: no questions to check", goldens.display()).into());
    }
    if !failures.is_empty() {
        for f in failures.iter().take(40) {
            eprintln!("FAIL {f}");
        }
        return Err(format!("{} failures", failures.len()).into());
    }
    println!("PASS");
    Ok(())
}
