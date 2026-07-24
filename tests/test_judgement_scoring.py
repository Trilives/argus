import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation import summarize_judgement, to_detection_flag, to_gold_status


class LabelMappingTest(unittest.TestCase):
    def test_model_labels_map_into_gold_space(self) -> None:
        self.assertEqual(to_gold_status("compliant"), "compliant")
        self.assertEqual(to_gold_status("non_compliant"), "non_compliant")

    def test_both_abstentions_collapse_to_undetermined(self) -> None:
        self.assertEqual(to_gold_status("uncertain"), "undetermined")
        self.assertEqual(to_gold_status("need_review"), "undetermined")

    def test_unparsed_output_is_treated_as_abstention(self) -> None:
        self.assertEqual(to_gold_status(None), "undetermined")
        self.assertEqual(to_gold_status("garbage"), "undetermined")

    def test_only_non_compliant_is_flagged(self) -> None:
        self.assertEqual(to_detection_flag("non_compliant"), "flagged")
        self.assertEqual(to_detection_flag("compliant"), "not_flagged")
        self.assertEqual(to_detection_flag("undetermined"), "not_flagged")


class SummarizeJudgementTest(unittest.TestCase):
    def test_empty_input_returns_zeroed_report(self) -> None:
        report = summarize_judgement([])
        self.assertEqual(report["sample_count"], 0)
        self.assertEqual(report["detection"]["f1"], 0.0)

    def test_confusion_counts_map_model_into_gold_space(self) -> None:
        pairs = [
            ("non_compliant", "non_compliant"),  # true positive
            ("compliant", "compliant"),
            ("undetermined", "need_review"),     # abstention matches abstention
            ("compliant", "uncertain"),          # model abstains on a compliant gold
        ]
        report = summarize_judgement(pairs)
        conf = report["confusion"]
        self.assertEqual(conf["non_compliant"]["non_compliant"], 1)
        self.assertEqual(conf["undetermined"]["undetermined"], 1)
        self.assertEqual(conf["compliant"]["undetermined"], 1)

    def test_undetermined_collapses_to_not_flagged_in_detector_view(self) -> None:
        # A gold non_compliant that the model abstained on is a missed detection (FN),
        # never a true positive.
        report = summarize_judgement([("non_compliant", "need_review")])
        det = report["detection"]
        self.assertEqual((det["tp"], det["fp"], det["fn"], det["tn"]), (0, 0, 1, 0))
        self.assertEqual(det["recall"], 0.0)

    def test_detection_precision_recall(self) -> None:
        pairs = [
            ("non_compliant", "non_compliant"),  # TP
            ("non_compliant", "compliant"),      # FN
            ("compliant", "non_compliant"),      # FP
            ("compliant", "compliant"),          # TN
            ("undetermined", "non_compliant"),   # FP (undetermined gold is not-flagged)
        ]
        det = summarize_judgement(pairs)["detection"]
        self.assertEqual((det["tp"], det["fp"], det["fn"], det["tn"]), (1, 2, 1, 1))
        self.assertAlmostEqual(det["precision"], 1 / 3)
        self.assertAlmostEqual(det["recall"], 1 / 2)

    def test_selective_coverage_treats_undetermined_as_abstention(self) -> None:
        pairs = [
            ("compliant", "compliant"),          # committed + correct
            ("non_compliant", "non_compliant"),  # committed + correct
            ("compliant", "non_compliant"),      # committed + wrong
            ("non_compliant", "need_review"),    # abstained (not committed)
        ]
        sel = summarize_judgement(pairs)["selective"]
        self.assertEqual(sel["committed"], 3)
        self.assertAlmostEqual(sel["coverage"], 3 / 4)
        self.assertAlmostEqual(sel["committed_accuracy"], 2 / 3)

    def test_unknown_gold_status_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            summarize_judgement([("bogus", "compliant")])


if __name__ == "__main__":
    unittest.main()
