"""jev-style render parity at the token level: `ref.py` against the author's own `Renderer`
(`jev_style_decision_gguf.py`) with the author's tokenizer (`tokenizer/tokenizer.json`), both at the pinned commit,
on Ollaya's shared request set and this layout's own cases (several state blocks, a 150-option catalogue overflow,
inputs of exactly 25,600 and 25,601 tokens). Token ids, verdict slots, attention blocks and option names must be
identical, and both must reject the same requests.

The pinned files (about 20 MB) are downloaded once into the package cache (OLLAYA_CACHE, default
~/.cache/ollaya-package); without network access the test is skipped. Needs `tokenizers` (the convert environment).
"""
import hashlib
import importlib.util
import json
import os
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
HAVE_DEPS = all(importlib.util.find_spec(m) for m in ("numpy", "tokenizers"))
if HAVE_DEPS:  # goldens.py imports numpy; CI's packaging job has no ML dependencies
    from ollaya_convert.families.jev_style import goldens, ref

REPO = "https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF/resolve/" \
       "283b9eb7903aeb184b16d49ab8f38b256f4afcb3/"
FILES = {
    "jev_style_decision_gguf.py": (REPO + "jev_style_decision_gguf.py",
                                   "edfed616477f0896e49e03c9b306b159903ee3d9bc5eeea2d4d30c3727c6135a"),
    "readout_config.json": (REPO + "readout_config.json",
                            "4af0d578c9126d4eb1a545b6e576e0a49a967243a47b8e99e3555f081fdaa1de"),
    # readout_config.json's tokenizer_sha256.
    "tokenizer.json": (REPO + "tokenizer/tokenizer.json",
                       "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"),
    "ollaya-shared-requests.jsonl": (
        "https://gist.githubusercontent.com/cobanov/dffe49d26ce328aa97716c3085249fef/raw/"
        "0844802134668f3b9daf478870419ace27968718/ollaya-shared-requests.jsonl",
        "c95b85a281ff542556f5c2177b07d3e26b5b14eb1658ca0cc28bf5e524c09e30"),
}
CACHE = Path(os.environ.get("OLLAYA_CACHE") or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
             / "ollaya-package") / "jev-style-283b9eb"


def fetch(name):
    url, sha = FILES[name]
    path = CACHE / name
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        if hashlib.sha256(data).hexdigest() != sha:
            raise AssertionError("%s: sha256 differs from the pin" % url)
        path.write_bytes(data)
    return path


@unittest.skipUnless(HAVE_DEPS, "needs the convert environment (numpy, tokenizers)")
class RenderParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            paths = {name: fetch(name) for name in FILES}
        except OSError as e:  # no network: nothing to compare against
            raise unittest.SkipTest("pinned jev-style files unavailable: %s" % e)
        spec = importlib.util.spec_from_file_location("jev_style_decision_gguf", paths["jev_style_decision_gguf.py"])
        cls.up = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.up)
        cls.enc = cls.up.TextEncoder(paths["tokenizer.json"])
        cls.renderer = cls.up.Renderer(cls.enc, json.loads(paths["readout_config.json"].read_text()))
        cls.arrow = cls.renderer.arrow[0]
        cases = [json.loads(line) for line in paths["ollaya-shared-requests.jsonl"].read_text().splitlines() if line]
        cls.cases = [(c["id"], c["state"], c["questions"]) for c in cases] + goldens.layout_cases(cls.renderer)

    def author(self, state, questions):
        prefix = self.renderer.prefix_ids(state)
        return [(qid, self.renderer.render(state, self.up.make_question(goldens.author_question(q)), prefix=prefix))
                for qid, q in questions.items()]

    def port(self, state, questions):
        prefix = ref.prefix_ids(state, self.enc)
        return [(qid, ref.render(state, q, self.enc, self.arrow, prefix=prefix)) for qid, q in questions.items()]

    def test_shared_set_and_layout_cases(self):
        seen = {"overflow": 0, "multi_state_block": 0, "rejected": 0, "budget_edge": 0}
        for cid, state, questions in self.cases:
            with self.subTest(cid):
                try:
                    want = self.author(state, questions)
                except (self.up.QuestionError, self.up.InputBudgetError):
                    with self.assertRaises((ref.QuestionError, ref.InputBudgetError)):
                        self.port(state, questions)
                    seen["rejected"] += 1
                    continue
                got = self.port(state, questions)
                self.assertEqual([q for q, _ in got], [q for q, _ in want])
                for (qid, g), (_, w) in zip(got, want):
                    self.assertEqual(g.ids, w.ids, qid)
                    self.assertEqual(g.slots, w.slots, qid)
                    self.assertEqual(g.blocks, [tuple(b) for b in w.blocks], qid)
                    self.assertEqual(g.names, w.names, qid)
                    self.assertEqual(g.prefix_len, w.prefix_len, qid)
                    self.assertEqual(g.catalogue_overflow, w.catalogue_overflow, qid)
                    seen["overflow"] += w.catalogue_overflow
                    seen["multi_state_block"] += len(w.state_blocks) > 1
                    seen["budget_edge"] += len(w.ids) == ref.CONTEXT_LIMIT
        # The set must exercise every form, or the test proves less than it says.
        self.assertTrue(all(seen.values()), seen)


if __name__ == "__main__":
    unittest.main()
