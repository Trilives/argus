"""Offline verification of the public ARGus package; no model calls or private inputs.

Run without arguments to verify. --write-manifest refreshes the public-file manifest only
after data, metadata, navigation and immutable research snapshots have passed their checks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tomllib
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "reproduce/release_manifest.json"
ROOT_FILES = ("README.md", "MODELS.md", "LICENSE", "NOTICE", "CITATION.cff", ".zenodo.json",
              ".gitignore", "pyproject.toml", "uv.lock", "run_demo.py", "run_rcasr.py", "data/README.md")
PUBLIC_DIRS = ("src", "Prompts_en", "configs", "data/rules", "data/rule_assets", "data/public_eval",
               "data/gold_images", "data/research_images", "data/image_pool", "docs", "experiments",
               "reproduce", "results", "tests", "scripts", "examples", "paper")
OMIT_PARTS = {"__pycache__", ".pytest_cache", ".venv", ".git", "public_images"}
LOCAL_OUTPUTS = ("results/demo/", "results/demo_rcasr/")
PRIVATE_PARTS = {"manuscript", "Journal_response", "review_freeze", "final_contact_sheets",
                 "final_blur_region_sheets", "review-private"}
IMAGE_LICENCES = {
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC-BY-NC-4.0": "https://creativecommons.org/licenses/by-nc/4.0/",
    "Apache-2.0": "https://www.apache.org/licenses/LICENSE-2.0",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    path = root / relative
    require(not Path(relative).is_absolute() and path.resolve().is_relative_to(root.resolve()),
            f"unsafe package path: {relative}")
    return path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def public_files(root: Path) -> list[Path]:
    paths = {root / name for name in ROOT_FILES}
    for directory in PUBLIC_DIRS:
        for path in (root / directory).rglob("*"):
            parts = path.relative_to(root).parts
            if set(parts) & OMIT_PARTS or path.suffix == ".pyc" or str(path.relative_to(root)).startswith(LOCAL_OUTPUTS):
                continue
            require(not (set(parts) & PRIVATE_PARTS) and not path.name.startswith((".env", "review_private")),
                    f"private artifact in public tree: {path.relative_to(root)}")
            require(not path.is_symlink(), f"symlink in public tree: {path.relative_to(root)}")
            if path.is_file() and str(path.relative_to(root)) != MANIFEST:
                paths.add(path)
    require(all(path.is_file() and not path.is_symlink() for path in paths), "required public file is missing or a symlink")
    return sorted(paths)


def verify_versions(root: Path) -> str:
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    version = project["version"]
    cff = re.search(r'^version:\s*["\']?([^"\'\s]+)', (root / "CITATION.cff").read_text(), re.M)
    zenodo = read_json(root / ".zenodo.json")
    lock = tomllib.loads((root / "uv.lock").read_text())
    packages = [p for p in lock["package"] if p["name"] == project["name"]]
    require(cff is not None and cff.group(1) == version, "CITATION.cff version differs")
    require(zenodo["version"].removeprefix("v") == version, "Zenodo version differs")
    require(len(packages) == 1 and packages[0]["version"] == version, "uv.lock project version differs")
    require(len(zenodo["creators"]) == 5, "author metadata changed unexpectedly")
    return version


def verify_image(path: Path, row: dict, max_side: int | None = None) -> None:
    require(path.is_file() and digest(path) == row["sha256"], f"image checksum differs: {path.name}")
    with Image.open(path) as image:
        require(image.format == "JPEG", f"not a JPEG: {path.name}")
        require(image.size == (int(row["width_px"]), int(row["height_px"])), f"dimensions differ: {path.name}")
        if max_side:
            require(max(image.size) <= max_side, f"image exceeds size limit: {path.name}")
        require(not dict(image.getexif()) and not any(image.info.get(k) for k in
                ("exif", "icc_profile", "comment", "xmp")), f"image contains metadata: {path.name}")
        image.verify()


def verify_images(root: Path) -> dict:
    rows = read_csv(root / "data/image_pool/manifest.csv")
    require(len(rows) == 1000 and len({r["image_id"] for r in rows}) == 1000, "expected 1,000 unique original IDs")
    require(Counter(r["subset"] for r in rows) == {"gold": 500, "research_unlabelled": 500}, "wrong original subsets")
    gold = set(read_json(root / "reproduce/gold_labels.json"))
    require({Path(r["image_id"]).stem for r in rows if r["subset"] == "gold"} == gold, "gold IDs changed")
    by_id = {r["image_id"]: r for r in rows}
    for row in rows:
        require(bool(re.fullmatch(r"img_\d{4}\.jpg", row["image_id"])), "invalid anonymous original ID")
        folder = "gold_images" if row["subset"] == "gold" else "research_images"
        require(row["release_relpath"] == f"data/{folder}/{row['image_id']}", "unexpected image destination")
        verify_image(safe_path(root, row["release_relpath"]), row, 2048)
        require(bool(re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"])), "invalid raw-input hash")
        alias = row["source_byte_alias_of"]
        if alias:
            require(alias in by_id and alias != row["image_id"] and
                    not by_id[alias]["source_byte_alias_of"] and
                    by_id[alias]["source_sha256"] == row["source_sha256"], "invalid source-byte alias")
    require(len({r["source_sha256"] for r in rows}) == 968, "source identity count changed")
    require(sum(bool(r["source_byte_alias_of"]) for r in rows) == 32, "source alias count changed")
    verify_subset_manifests(root, by_id)
    require(not (root / "data/research_crops").exists(), "cropped dataset must not be bundled")
    samples = read_csv(root / "examples/sample_images/manifest.csv")
    require(len(samples) == 40 and len({r["filename"] for r in samples}) == 40, "expected 40 unique demo images")
    for row in samples:
        require(bool(re.fullmatch(r"P\d{3}\.jpg", row["filename"])), "invalid demo filename")
        verify_image(root / "examples/sample_images" / row["filename"], row)
    require({p.name for p in (root / "examples/sample_images").glob("*.jpg")} ==
            {r["filename"] for r in samples}, "unlisted demo files")
    audit = read_json(root / "data/research_images/anonymisation.json")["images"]
    require(set(audit) == {Path(r["image_id"]).stem for r in rows if r["subset"] != "gold"}, "incomplete addition audit")
    require(all(r["pixels_changed_outside"] == 0 and r["exif_stripped"] for r in audit.values()), "failed pixel/metadata audit")
    return {"originals": len(rows), "gold": len(gold), "additions": len(audit), "demos": len(samples), "crops": 0}


def verify_subset_manifests(root: Path, by_id: dict) -> None:
    for folder in ("gold_images", "research_images"):
        per_folder = read_csv(root / f"data/{folder}/manifest.csv")
        require(len(per_folder) == 500 and len({r["filename"] for r in per_folder}) == 500, f"wrong {folder} count")
        for row in per_folder:
            combined = by_id[row["filename"]]
            require(all(row[k] == combined[k] for k in ("sha256", "width_px", "height_px")),
                    f"{folder} manifest disagrees")
            require(combined["release_relpath"] == f"data/{folder}/{row['filename']}", "wrong manifest subset")
            if folder == "research_images":
                require(row["license"] == combined["license"], "per-directory image licence differs")
        require({p.name for p in (root / f"data/{folder}").glob("*.jpg")} ==
                {r["filename"] for r in per_folder}, f"unlisted images in {folder}")


def verify_image_licenses(root: Path, rows: list[dict]) -> dict:
    gold_sources = {r["source_sha256"] for r in rows if r["subset"] == "gold"}
    by_source, counts = {}, Counter()
    for row in rows:
        licence, prior = row["license"], row["prior_public_release"]
        require(licence in IMAGE_LICENCES and row["license_url"] == IMAGE_LICENCES[licence],
                f"invalid image licence: {row['image_id']}")
        if row["source_sha256"] in gold_sources:
            require(licence == "CC-BY-4.0" and prior == "gold", "published gold permission was restricted")
        elif prior == "demonstration":
            require(licence == "Apache-2.0", "published demonstration permission was restricted")
        else:
            require(not prior and licence == "CC-BY-NC-4.0", "eligible additional image must be non-commercial")
        require(by_source.get(row["source_sha256"], licence) == licence, "identical source has inconsistent image licences")
        by_source[row["source_sha256"]] = licence
        counts[licence] += 1
    expected = [{k: r[k] for k in ("image_id", "license", "license_url", "prior_public_release")}
                for r in rows if r["subset"] == "research_unlabelled" and r["license"] != "CC-BY-NC-4.0"]
    require(read_csv(root / "data/research_images/LICENSE_EXCEPTIONS.csv") == expected,
            "image licence exceptions differ from the combined manifest")
    require("CC BY-NC" in (root / "data/research_images/LICENSE").read_text() and
            "CC BY 4.0" in (root / "data/gold_images/LICENSE").read_text(), "image notices disagree with licence policy")
    return dict(counts)


def verify_snapshots(root: Path) -> dict:
    exceptions = read_json(root / "reproduce/snapshot_exceptions.json")
    migrations = verify_migrations(root)
    counts = Counter()
    for snapshot in sorted((root / "results").glob("**/input_snapshot.json")):
        for relative, expected in read_json(snapshot).get("file_sha256", {}).items():
            moved = migrations.get("paths", {}).get(relative, {})
            edit = migrations.get("code_updates", {}).get(relative, {})
            path = safe_path(root, moved.get("destination", relative))
            if not path.is_file():
                require(relative in exceptions["missing"] and exceptions["missing"][relative]["reason"],
                        f"unexplained missing frozen input: {relative}")
                counts["documented_missing"] += 1
            elif edit.get("original_sha256") == expected and edit.get("current_sha256") == digest(path):
                counts["path_migrated"] += 1
            elif digest(path) != expected:
                entry = exceptions["changed"].get(relative, {})
                require(entry.get("reason") and entry.get("public_sha256") == digest(path),
                        f"unexplained changed frozen input: {relative}")
                counts["documented_public_difference"] += 1
            else:
                counts["path_migrated" if moved else "matching"] += 1
    require(counts["matching"] + counts.get("path_migrated", 0) >= 279, "expected frozen research inputs are missing")
    return dict(counts)


def verify_migrations(root: Path) -> dict:
    path = root / "reproduce/path_migrations.json"
    if not path.is_file():
        return {}
    migrations = read_json(path)
    for relative, entry in migrations.get("paths", {}).items():
        dest = safe_path(root, entry["destination"])
        require(dest.is_file() and digest(dest) == entry["sha256"], f"changed relocated input: {relative}")
    for relative, entry in migrations.get("code_updates", {}).items():
        source = safe_path(root, relative)
        require(digest(source) == entry["current_sha256"], f"changed path-migration code: {relative}")
        if entry.get("kind") != "snapshot_path_resolution":
            original = source.read_bytes().replace(b"docs/research/design/", b"docs/paper_v3/design/")
            require(hashlib.sha256(original).hexdigest() == entry["original_sha256"],
                    f"path migration changed more than a reference: {relative}")
    return migrations


def heading_anchors(text: str) -> set[str]:
    anchors, seen = set(), Counter()
    for line in text.splitlines():
        if not re.match(r"^#{1,6} ", line):
            continue
        heading = re.sub(r"[^\w -]", "", re.sub(r"^#+ ", "", line).lower())
        slug = re.sub(r"\s", "-", heading)
        anchors.add(slug if not seen[slug] else f"{slug}-{seen[slug]}")
        seen[slug] += 1
    return anchors


def verify_links(root: Path) -> int:
    count = 0
    files = public_files(root)
    packaged = {p.resolve() for p in files} | {(root / MANIFEST).resolve()}
    for path in files:
        relative = str(path.relative_to(root))
        if path.suffix != ".md" or relative.startswith(("docs/research/design/", "results/")):
            continue  # Immutable historical documents retain their original research-context references.
        text = re.sub(r"```[\s\S]*?```", "", path.read_text())
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            parsed = urlsplit(target.strip("<>"))
            if parsed.scheme or parsed.netloc:
                continue
            dest = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
            require(dest.is_relative_to(root.resolve()) and dest.exists(), f"broken link in {relative}: {target}")
            require(not dest.is_file() or dest in packaged, f"linked file is not in public package: {target}")
            if parsed.fragment and dest.suffix == ".md":
                require(unquote(parsed.fragment) in heading_anchors(dest.read_text()), f"broken heading in {relative}: {target}")
            count += 1
    return count


def write_manifest(root: Path, version: str) -> None:
    value = {"schema_version": 1, "version": version,
             "scope": "Public package only; this manifest excludes itself and local environments/caches.",
             "file_sha256": {str(p.relative_to(root)): digest(p) for p in public_files(root)}}
    (root / MANIFEST).write_text(json.dumps(value, indent=2) + "\n")


def verify_manifest(root: Path, version: str) -> int:
    manifest = read_json(root / MANIFEST)
    actual = {str(p.relative_to(root)): digest(p) for p in public_files(root)}
    require(manifest["version"] == version, "release manifest version differs")
    require(actual == manifest["file_sha256"], "public files changed; inspect and refresh the release manifest")
    return len(actual)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write-manifest", action="store_true", help="refresh after other checks pass")
    args = parser.parse_args()
    try:
        version = verify_versions(ROOT)
        report = {"version": version, "images": verify_images(ROOT), "snapshots": verify_snapshots(ROOT),
                  "original_image_licenses": verify_image_licenses(ROOT, read_csv(ROOT / "data/image_pool/manifest.csv")),
                  "local_document_links": verify_links(ROOT)}
        if args.write_manifest:
            write_manifest(ROOT, version)
        report["public_files"] = verify_manifest(ROOT, version)
        print(json.dumps(report, indent=2))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Release verification failed: {exc}\n")


if __name__ == "__main__":
    main()
