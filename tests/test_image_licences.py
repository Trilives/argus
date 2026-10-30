"""Non-commercial image scope never withdraws the previously published grants."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_release import IMAGE_LICENCES, verify_image_licenses


class ImageLicenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for folder in ("gold_images", "research_images"):
            (self.root / "data" / folder).mkdir(parents=True)
        (self.root / "data/gold_images/LICENSE").write_text("CC BY 4.0")
        (self.root / "data/research_images/LICENSE").write_text("CC BY-NC 4.0")
        self.rows = [self.row("gold.jpg", "gold", "gold-source", "CC-BY-4.0", "gold"),
                     self.row("same.jpg", "research_unlabelled", "gold-source", "CC-BY-4.0", "gold"),
                     self.row("demo.jpg", "research_unlabelled", "demo-source", "Apache-2.0", "demonstration"),
                     self.row("new.jpg", "research_unlabelled", "new-source", "CC-BY-NC-4.0", "")]
        self.write_exceptions()

    @staticmethod
    def row(iid, subset, source, licence, prior):
        return {"image_id": iid, "subset": subset, "source_sha256": source,
                "license": licence, "license_url": IMAGE_LICENCES[licence], "prior_public_release": prior}

    def write_exceptions(self):
        keys = ["image_id", "license", "license_url", "prior_public_release"]
        with (self.root / "data/research_images/LICENSE_EXCEPTIONS.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
            writer.writeheader()
            writer.writerows({k: r[k] for k in keys} for r in self.rows
                            if r["subset"] != "gold" and r["license"] != "CC-BY-NC-4.0")

    def test_mixed_image_licences_are_verified(self):
        self.assertEqual(verify_image_licenses(self.root, self.rows),
                         {"CC-BY-4.0": 2, "Apache-2.0": 1, "CC-BY-NC-4.0": 1})

    def test_published_gold_alias_cannot_be_restricted(self):
        self.rows[1].update(license="CC-BY-NC-4.0", license_url=IMAGE_LICENCES["CC-BY-NC-4.0"])
        with self.assertRaisesRegex(ValueError, "gold permission was restricted"):
            verify_image_licenses(self.root, self.rows)

    def test_published_demo_grant_is_preserved(self):
        self.rows[2].update(license="CC-BY-NC-4.0", license_url=IMAGE_LICENCES["CC-BY-NC-4.0"])
        with self.assertRaisesRegex(ValueError, "demonstration permission was restricted"):
            verify_image_licenses(self.root, self.rows)

    def test_genuinely_new_file_requires_noncommercial_terms(self):
        self.rows[3].update(license="CC-BY-4.0", license_url=IMAGE_LICENCES["CC-BY-4.0"])
        with self.assertRaisesRegex(ValueError, "must be non-commercial"):
            verify_image_licenses(self.root, self.rows)

    def test_exception_list_cannot_omit_a_prior_grant(self):
        with (self.root / "data/research_images/LICENSE_EXCEPTIONS.csv").open("w") as handle:
            handle.write("image_id,license,license_url,prior_public_release\n")
        with self.assertRaisesRegex(ValueError, "exceptions differ"):
            verify_image_licenses(self.root, self.rows)

    def test_misleading_licence_url_is_rejected(self):
        self.rows[3]["license_url"] = IMAGE_LICENCES["CC-BY-4.0"]
        with self.assertRaisesRegex(ValueError, "invalid image licence"):
            verify_image_licenses(self.root, self.rows)


if __name__ == "__main__":
    unittest.main()
