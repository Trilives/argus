"""The public release checker catches tampering, broken navigation and private artifacts."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import verify_release as release


class ReleaseVerificationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "docs").mkdir()
        (self.root / "reproduce").mkdir()
        (self.root / "README.md").write_text("# Start\n\n[Guide](docs/guide.md#inference)\n")
        (self.root / "docs/guide.md").write_text("# Guide\n\n## Inference\n")
        for attribute, value in (("ROOT_FILES", ("README.md",)), ("PUBLIC_DIRS", ("docs",))):
            patch = mock.patch.object(release, attribute, value)
            patch.start()
            self.addCleanup(patch.stop)

    def test_manifest_detects_changed_files(self):
        release.write_manifest(self.root, "2.4.0")
        self.assertEqual(release.verify_manifest(self.root, "2.4.0"), 2)
        (self.root / "docs/guide.md").write_text("changed")
        with self.assertRaisesRegex(ValueError, "public files changed"):
            release.verify_manifest(self.root, "2.4.0")

    def test_manifest_detects_new_or_removed_files(self):
        release.write_manifest(self.root, "2.4.0")
        (self.root / "docs/unlisted.md").write_text("unlisted")
        with self.assertRaisesRegex(ValueError, "public files changed"):
            release.verify_manifest(self.root, "2.4.0")

    def test_links_validate_heading_fragments(self):
        self.assertEqual(release.verify_links(self.root), 1)
        (self.root / "docs/guide.md").write_text("# Guide\n")
        with self.assertRaisesRegex(ValueError, "broken heading"):
            release.verify_links(self.root)

    def test_links_reject_missing_destinations(self):
        (self.root / "README.md").write_text("[Missing](docs/missing.md)")
        with self.assertRaisesRegex(ValueError, "broken link"):
            release.verify_links(self.root)

    def test_links_cannot_depend_on_an_unbundled_local_file(self):
        (self.root / "unbundled.md").write_text("only in the working tree")
        (self.root / "README.md").write_text("[Local file](unbundled.md)")
        with self.assertRaisesRegex(ValueError, "not in public package"):
            release.verify_links(self.root)

    def test_private_records_are_not_packaged(self):
        private = self.root / "docs/Journal_response"
        private.mkdir()
        (private / "authorization.md").write_text("private")
        with self.assertRaisesRegex(ValueError, "private artifact"):
            release.public_files(self.root)

    def test_local_inference_outputs_are_excluded(self):
        private = self.root / "results/demo_rcasr"
        private.mkdir(parents=True)
        (private / "rcasr_runs.jsonl").write_text("local run output")
        with mock.patch.object(release, "PUBLIC_DIRS", ("docs", "results")):
            self.assertFalse(any(p.name == "rcasr_runs.jsonl" for p in release.public_files(self.root)))

    def test_symlink_and_path_traversal_are_rejected(self):
        (self.root / "docs/link.md").symlink_to(self.root / "README.md")
        with self.assertRaisesRegex(ValueError, "symlink"):
            release.public_files(self.root)
        with self.assertRaisesRegex(ValueError, "unsafe package path"):
            release.safe_path(self.root, "../outside.json")

    def test_image_metadata_and_hashes_are_checked(self):
        path = self.root / "image.jpg"
        Image.new("RGB", (20, 12), "red").save(path)
        row = {"sha256": release.digest(path), "width_px": "20", "height_px": "12"}
        release.verify_image(path, row, 2048)
        exif = Image.Exif()
        exif[315] = "private author"
        Image.new("RGB", (20, 12), "red").save(path, exif=exif)
        row["sha256"] = release.digest(path)
        with self.assertRaisesRegex(ValueError, "contains metadata"):
            release.verify_image(path, row, 2048)

    def test_snapshot_exceptions_do_not_allow_new_code_changes(self):
        entries = {}
        for n in range(279):
            p = self.root / f"docs/frozen_{n}.md"
            p.write_text(f"frozen {n}")
            entries[str(p.relative_to(self.root))] = release.digest(p)
        changed = self.root / "docs/public.md"
        changed.write_text("public copy")
        entries["docs/public.md"] = "historical-hash"
        entries["private.json"] = "private-input-hash"
        folder = self.root / "results/frozen"
        folder.mkdir(parents=True)
        (folder / "input_snapshot.json").write_text(json.dumps({"file_sha256": entries}))
        (self.root / "reproduce/snapshot_exceptions.json").write_text(json.dumps({
            "missing": {"private.json": {"reason": "private"}},
            "changed": {"docs/public.md": {"reason": "public copy", "public_sha256": release.digest(changed)}}}))
        counts = release.verify_snapshots(self.root)
        self.assertEqual(counts, {"matching": 279, "documented_public_difference": 1, "documented_missing": 1})
        (self.root / "docs/frozen_0.md").write_text("unexpected edit")
        with self.assertRaisesRegex(ValueError, "unexplained changed frozen input"):
            release.verify_snapshots(self.root)

    def test_version_drift_is_rejected(self):
        (self.root / "pyproject.toml").write_text('[project]\nname="argus"\nversion="2.4.0"\n')
        (self.root / "CITATION.cff").write_text('version: "2.4.0"\n')
        (self.root / ".zenodo.json").write_text(json.dumps({"version": "v2.4.0", "creators": [{}] * 5}))
        (self.root / "uv.lock").write_text('[[package]]\nname="argus"\nversion="2.4.0"\n')
        self.assertEqual(release.verify_versions(self.root), "2.4.0")
        (self.root / "CITATION.cff").write_text('version: "2.3.0"\n')
        with self.assertRaisesRegex(ValueError, "CITATION.cff version differs"):
            release.verify_versions(self.root)


if __name__ == "__main__":
    unittest.main()
