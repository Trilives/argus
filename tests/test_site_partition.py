"""Site folds must never split a site or a near-duplicate cluster."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from site_partition import assign_folds, partition, site_of, split_leaks, units  # noqa: E402


def confirmed(groups: dict[str, list[str]]) -> dict:
    return {"sites": [{"site_key": k, "images": [{"image_id": i} for i in v]}
                      for k, v in groups.items()]}


class SitePartitionTests(unittest.TestCase):
    def test_duplicate_assignment_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            site_of(confirmed({"a": ["i1"], "b": ["i1"]}))

    def test_cluster_across_sites_fuses_them(self) -> None:
        sites = site_of(confirmed({"a": ["i1", "i2"], "b": ["i3"], "c": ["i4"]}))
        rows = units(sites, [["i2", "i3"]])
        fused = [u for u in rows if u["fused_by_near_duplicate"]]
        self.assertEqual(len(rows), 2)
        self.assertEqual(fused[0]["sites"], ["a", "b"])

    def test_unsited_images_are_excluded_not_pooled(self) -> None:
        doc = confirmed({"a": ["i1"], "b": ["i2"], "c": ["i3"]})
        out = partition(doc, [], {"i1", "i2", "i3", "x9"}, k=2)
        self.assertNotIn("x9", out["folds"])
        self.assertEqual(out["counts"]["excluded_no_site"], 1)

    def test_folds_are_deterministic_and_balanced(self) -> None:
        doc = confirmed({f"s{j}": [f"i{j}_{m}" for m in range(j % 4 + 1)] for j in range(12)})
        population = {i["image_id"] for s in doc["sites"] for i in s["images"]}
        first = partition(doc, [], population, k=3)
        self.assertEqual(first["folds"], partition(doc, [], population, k=3)["folds"])
        sizes = first["counts"]["fold_sizes"]
        self.assertLessEqual(max(sizes) - min(sizes), 4)

    @unittest.skipUnless((ROOT / "data/splits/site_keys_confirmed.json").exists(),
                         "needs the restricted site keys and gold (available under a data-use agreement)")
    def test_no_leaks_on_the_confirmed_sites(self) -> None:
        doc = json.loads((ROOT / "data/splits/site_keys_confirmed.json").read_text())
        clusters = json.loads((ROOT / "results/data_audit/near_duplicates.json").read_text())["clusters"]
        gold = json.loads((ROOT / "data/annotations/image_rule_gold.json").read_text())
        population = {Path(i).stem for i in gold["images"]}
        out = partition(doc, clusters, population)
        sites = {i: s for i, s in site_of(doc).items() if i in population}
        self.assertEqual(split_leaks(out["folds"], sites, clusters), [])
        self.assertEqual(out["counts"]["assigned"], 346)

    def test_k_must_fit_the_units(self) -> None:
        with self.assertRaises(ValueError):
            assign_folds([{"unit": "a", "images": ["i"]}], k=2)


if __name__ == "__main__":
    unittest.main()
