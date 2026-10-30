"""Relocating research plans preserves their original content hashes and snapshot checks."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from research_snapshot import sha256, verify_snapshot
from verify_release import verify_migrations


class SnapshotPathTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old = "docs/paper_v3/design/PLAN.md"
        self.new = "docs/research/design/PLAN.md"
        self.plan = self.root / self.new
        self.plan.parent.mkdir(parents=True)
        self.plan.write_text("frozen analysis plan\n")
        (self.root / "reproduce").mkdir()
        self.migrations = {"paths": {self.old: {"destination": self.new, "sha256": sha256(self.plan)}},
                           "code_updates": {}}
        self.frozen = {"file_sha256": {self.old: sha256(self.plan)}}
        self.write_migrations()

    def write_migrations(self):
        (self.root / "reproduce/path_migrations.json").write_text(json.dumps(self.migrations))

    def test_old_snapshot_resolves_the_relocated_plan(self):
        verify_snapshot(self.root, self.frozen)
        verify_migrations(self.root)
        self.assertFalse((self.root / self.old).exists())

    def test_changing_the_plan_payload_still_fails(self):
        self.plan.write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed or missing"):
            verify_snapshot(self.root, self.frozen)
        with self.assertRaisesRegex(ValueError, "changed relocated input"):
            verify_migrations(self.root)

    def test_only_recorded_path_reference_edits_are_allowed(self):
        source = self.root / "entry.py"
        source.write_text('PLAN="docs/paper_v3/design/PLAN.md"\n')
        before = sha256(source)
        source.write_text('PLAN="docs/research/design/PLAN.md"\n')
        self.migrations["code_updates"]["entry.py"] = {
            "original_sha256": before, "current_sha256": sha256(source)}
        self.write_migrations()
        verify_migrations(self.root)
        verify_snapshot(self.root, {"file_sha256": {"entry.py": before}})
        source.write_text('PLAN="docs/research/design/PLAN.md"\nchanged_logic=True\n')
        self.migrations["code_updates"]["entry.py"]["current_sha256"] = sha256(source)
        self.write_migrations()
        with self.assertRaisesRegex(ValueError, "more than a reference"):
            verify_migrations(self.root)

    def test_missing_private_inputs_are_not_silently_skipped(self):
        with self.assertRaisesRegex(ValueError, "changed or missing"):
            verify_snapshot(self.root, {"file_sha256": {"private.json": "hash"}})

    def test_destinations_cannot_escape_the_repository(self):
        self.migrations["paths"][self.old]["destination"] = "../outside.md"
        self.write_migrations()
        with self.assertRaises(ValueError):
            verify_snapshot(self.root, self.frozen)


if __name__ == "__main__":
    unittest.main()
