"""TP-07 metadata fixtures exercise the split protocol without reading images."""

import csv
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.rps import dataset


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture
def release(tmp_path):
    config = {"dataset_id": digest("release"), "inventory_version": digest("review"), "seed": 42}
    version = config["inventory_version"]
    source = tmp_path / "data" / "manifests" / config["dataset_id"] / "phase3-v1"
    inventory, groups = [], []
    for label in dataset.CLASSES:
        for i in range(21):
            image_id = f"{label}-{i}"
            excluded = i == 20
            row = dict(image_id=image_id, label=label, group_id=f"group-{i}",
                       disposition="excluded" if excluded else "eligible")
            groups.append(row)
            inventory.append(row | dict(
                dataset_id=config["dataset_id"], inventory_version=version,
                reason_code="reviewed_copy" if excluded else "", relative_path=f"{label}/{i}.png",
                file_sha256=digest(image_id), pixel_sha256=digest(f"pixels-{image_id}"),
            ))
    for name, records in (("inventory", inventory), ("groups", groups)):
        path = source / f"{name}-{version}.csv"
        dataset.write_csv(path, list(reversed(records)), list(records[0]))
        config[f"{name}_sha256"] = dataset.sha256_file(path)
    return tmp_path, config


def test_regenerate_review_and_freeze(release):
    root, config = release
    candidate = dataset.phase4_split(root, config, "split")
    path = dataset.filesystem_path(candidate["manifest"])
    original = path.read_bytes()
    repeated = dataset.phase4_split(root, config, "split", regenerate=True)
    assert candidate == repeated
    assert original == path.read_bytes()
    assert candidate["counts"] == {"train": 42, "validation": 9, "test": 9}
    assert all(min(classes.values()) > 0 for classes in candidate["per_class"].values())
    assert all(n == 0 for n in candidate["cross_split_overlap"].values())
    with pytest.raises(ValueError, match="audit evidence"):
        dataset.phase4_split(root, config, "freeze", candidate["sha256"])
    dataset.phase4_split(root, config, "audit")
    with pytest.raises(ValueError, match="frozen identity"):
        dataset.phase4_split(root, config, "frozen")
    with pytest.raises(ValueError, match="Reviewed manifest"):
        dataset.phase4_split(root, config, "freeze", "unreviewed")
    for _ in range(2):
        assert dataset.phase4_split(root, config, "freeze", candidate["sha256"]) == candidate
    assert dataset.phase4_split(root, config, "frozen") == candidate
    evidence = root / "artifacts" / "data" / "phase4" / candidate["split_version"]
    dataset.filesystem_path(evidence / "frozen.json").write_text("{}")
    with pytest.raises(ValueError, match="frozen identity"):
        dataset.phase4_split(root, config, "frozen")


@pytest.mark.parametrize("edit, message", [
    ("group", "Cross-split overlap in group_id"),
    ("source", "approved source"),
    ("duplicate", "Duplicate"),
    ("missing", "Missing eligible images"),
])
def test_audit_rejects_injected_conflict(release, edit, message):
    root, config = release
    candidate = dataset.phase4_split(root, config, "split")
    path = dataset.filesystem_path(candidate["manifest"])
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if edit == "group":
        rows[0]["split"] = next(name for name in dataset.SPLIT_PERCENT if name != rows[0]["split"])
    elif edit == "source":
        rows[0]["label"] = "unreviewed"
    elif edit == "duplicate":
        rows.append(rows[0])
    else:
        rows.pop()
    dataset.write_csv(path, rows, dataset.SPLIT_COLUMNS)
    with pytest.raises(ValueError, match=message):
        dataset.phase4_split(root, config, "audit")
    with pytest.raises(ValueError, match="Existing split.csv differs"):
        dataset.phase4_split(root, config, "split")


@pytest.mark.parametrize("field", ["file_sha256", "pixel_sha256"])
def test_audit_rejects_hash_crossing_group_boundary(release, field):
    root, config = release
    candidate = dataset.phase4_split(root, config, "split")
    with dataset.filesystem_path(candidate["manifest"]).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    foreign = next(row for row in rows if row["split"] != rows[0]["split"])
    foreign[field] = rows[0][field]
    sources = {row["image_id"]: {key: value for key, value in row.items()
                                 if key not in ("split", "split_version")} for row in rows}
    with pytest.raises(ValueError, match=f"Cross-split overlap in {field}"):
        dataset._audit_split(rows, sources, candidate["split_version"])


def test_frozen_use_rechecks_source_identity(release):
    root, config = release
    result = dataset.phase4_split(root, config, "split")
    dataset.phase4_split(root, config, "audit")
    dataset.phase4_split(root, config, "freeze", result["sha256"])
    source = root / "data" / "manifests" / config["dataset_id"] / "phase3-v1"
    dataset.filesystem_path(source / f"groups-{config['inventory_version']}.csv").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source SHA-256 mismatch"):
        dataset.phase4_split(root, config, "frozen")


def test_whole_groups_take_priority_over_ratio():
    eligible = {
        str((label, i)): {"label": label, "group_id": str(i)} for label in dataset.CLASSES for i in range(3)
    }
    assignments, _ = dataset._allocate_groups(eligible, 7)
    assert set(assignments.values()) == set(dataset.SPLIT_PERCENT)
    eligible.pop(str(("rock", 0)))
    with pytest.raises(ValueError, match="Too few independent groups"):
        dataset._allocate_groups(eligible, 7)


def test_atomic_csv_preserves_original_in_deep_directory(tmp_path):
    with TemporaryDirectory(dir=dataset.filesystem_path(tmp_path)) as temporary:
        folder = Path(temporary).name
        deep = tmp_path / folder / ("nested-" * 20) / ("release-" + digest("release"))
        path = deep / ("inventory-" + digest("review") + ".csv")
        assert len(str(path)) >= 260
        dataset.write_csv(path, [{"label": "rock"}], ["label"])
        assert dataset.sha256_file(path) == digest("label\nrock\n")

        with pytest.raises(ValueError, match="fields not in fieldnames"):
            dataset.write_csv(path, [{"label": "paper", "extra": "invalid"}], ["label"])

        assert dataset.sha256_file(path) == digest("label\nrock\n")
        assert list(dataset.filesystem_path(deep).iterdir()) == [dataset.filesystem_path(path)]
