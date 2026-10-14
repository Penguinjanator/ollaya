"""Ollaya Bench scoring (bench_typed.py, bench_edition.py) on hand-made rows: which answer counts as right, what a
rejected request costs, and the chance-corrected score; no network, weights or model runtimes needed."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ollaya_convert import bench_edition, bench_typed


ROW = {
    "id": "r0",
    "workflow": "customer_service",
    "state": {"message": "Refund me"},
    "questions": {
        "intent": {"type": "choice", "instructions": "?", "criteria": {"refund": "", "billing": "", "other": ""}},
        "urgent": {"type": "noul", "instructions": "?"},
        "anger": {"type": "score", "instructions": "?", "criteria": ["calm", "annoyed", "angry", "furious"]},
    },
    "gold": {
        "intent": {"label": "refund", "probabilities": {"refund": 0.8, "billing": 0.15, "other": 0.05}},
        "urgent": {"label": "false", "probabilities": {"false": 0.7, "true": 0.3}},
        "anger": {"label": "2", "probabilities": {"0": 0.0, "1": 0.2, "2": 0.6, "3": 0.2}},
    },
}


class TypedScoring(unittest.TestCase):
    def test_argmax_of_each_answer_against_the_gold_label(self):
        answers = {
            "intent": {"type": "choice", "choice": "refund", "probabilities": {"refund": 0.6, "billing": 0.3, "other": 0.1}},
            "urgent": {"type": "noul", "noul": 0.9},  # says true, gold is false
            "anger": {"type": "score", "score": 2.1, "probabilities": {"0": 0.0, "1": 0.1, "2": 0.7, "3": 0.2}},
        }
        items = bench_typed.score_rows([ROW], [{"id": "r0", "answers": answers}])
        self.assertEqual([x["correct"] for x in items], [1, 0, 1])
        self.assertAlmostEqual(items[1]["top"], 0.9)
        self.assertTrue(all(x["answered"] for x in items))

    def test_a_rejected_request_is_wrong_on_every_question(self):
        items = bench_typed.score_rows([ROW], [{"id": "r0", "error": "HTTP 422: too many options"}])
        self.assertEqual([x["correct"] for x in items], [0, 0, 0])
        self.assertFalse(any(x["answered"] for x in items))

    def test_option_keys_follow_the_wire_order(self):
        self.assertEqual(bench_typed.option_keys(ROW["questions"]["intent"]), ["refund", "billing", "other"])
        self.assertEqual(bench_typed.option_keys(ROW["questions"]["urgent"]), ["false", "true"])
        self.assertEqual(bench_typed.option_keys(ROW["questions"]["anger"]), ["0", "1", "2", "3"])


class EditionScore(unittest.TestCase):
    def test_skill_is_zero_at_chance_and_one_hundred_when_all_right(self):
        noul = [(1, 0.5), (0, 0.5)]  # half right on yes/no questions: what guessing gets
        self.assertAlmostEqual(bench_edition.skill(noul), 0.0)
        self.assertAlmostEqual(bench_edition.skill([(1, 0.25)] * 4), 100.0)
        self.assertLess(bench_edition.skill([(0, 0.5)] * 4), 0)

    def test_the_score_weighs_each_task_equally(self):
        pub = {"a": [(1, 0.5)] * 10, "b": [(1, 0.5), (0, 0.5)] * 50}
        typed = [(row, 1, 0.25) for row in range(3)]
        tasks = bench_edition.task_skills(pub, typed)
        self.assertEqual(tasks, {"a": 100.0, "b": 0.0, "typed-decisions": 100.0})

    def test_a_model_trained_on_a_task_is_scored_on_the_others(self):
        tasks = {"a": 50.0, "b": 30.0, "typed-decisions": 90.0}
        self.assertAlmostEqual(bench_edition.score_of(tasks, ["typed-decisions"]), 40.0)
        self.assertAlmostEqual(bench_edition.score_of(tasks, []), 170.0 / 3)


if __name__ == "__main__":
    unittest.main()
