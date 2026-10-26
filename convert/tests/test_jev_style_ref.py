"""jev-style `ref.py` on the inputs the shared Rust parser accepts (a choice as a list of labels, noul keys in any
case) and on the author's stricter limits (2..10 score levels, a non-empty string instruction). The cases and the
expected texts are the ones `crates/ollaya-decision/src/jev_style.rs` tests, so the two renders stay in step. A
character tokenizer stands in for the GGUF's; no weights needed."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ollaya_convert.families.jev_style import ref

ARROW = 1411


def enc(text):
    """One id per character, except ` ->`, which is the single verdict-slot id (as in the Rust tests)."""
    ids, i = [], 0
    while i < len(text):
        if text.startswith(" ->", i):
            ids.append(ARROW)
            i += 3
        else:
            ids.append(ord(text[i]))
            i += 1
    return ids


def text_of(ids):
    return "".join(" ->" if i == ARROW else chr(i) for i in ids)


def render(question, state="s"):
    return ref.render(state, question, enc, ARROW)


class AcceptedInputs(unittest.TestCase):
    def test_choice_as_a_list_of_labels(self):
        r = render({"type": "choice", "instructions": "Which team?",
                    "criteria": ["billing", "tech", "billing", "sales"]})
        self.assertEqual(r.names, ["billing", "tech", "sales"])
        self.assertTrue(text_of(r.ids).endswith(
            "Question [choice]: Which team?\nOptions:\n- billing ->\n- tech ->\n- sales ->\n"))
        self.assertEqual(len(r.slots), 3)
        self.assertTrue(all(r.ids[s] == ARROW for s in r.slots))

    def test_noul_keys_in_any_case(self):
        r = render({"type": "noul", "instructions": "Finished?",
                    "criteria": {"TRUE": "it is done", "False": "still open"}})
        self.assertEqual(r.names, ["false", "true"])
        self.assertTrue(text_of(r.ids).endswith("- false: still open ->\n- true: it is done ->\n"))


class AuthorLimits(unittest.TestCase):
    def test_score_levels_outside_two_to_ten(self):
        for n in (1, 11):
            with self.assertRaisesRegex(ref.QuestionError, "2..10 level descriptions"):
                render({"type": "score", "instructions": "How bad?",
                        "criteria": ["level %d" % i for i in range(n)]})
        for n in (2, 10):
            r = render({"type": "score", "instructions": "How bad?", "criteria": ["level %d" % i for i in range(n)]})
            self.assertEqual(len(r.slots), n)

    def test_non_string_or_blank_instructions(self):
        for ins in ({"text": "ok?"}, ["ok?"], 3, "  "):
            with self.assertRaisesRegex(ref.QuestionError, "non-empty string"):
                render({"type": "noul", "instructions": ins})


if __name__ == "__main__":
    unittest.main()
