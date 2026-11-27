import csv
import io
import json
import stat
import warnings
from zipfile import ZipFile, ZipInfo

from PIL import Image
import pytest

from src.rps.dataset import prepare_dataset


def png(color, *, compression=6, mode="RGB", size=(300, 200)):
    stream = io.BytesIO()
    Image.new(mode, size, color).save(stream, format="PNG", compress_level=compression)
    return stream.getvalue()


def archive_at(tmp_path, entries, name="source.zip"):
    archive = tmp_path / name
    with ZipFile(archive, "w") as handle:
        for member, content in entries:
            handle.writestr(member, content)
    return archive


def read_inventory(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["relative_path"]: row for row in csv.DictReader(handle)}


def test_inventory_preserves_labels_and_duplicate_lineage(tmp_path):
    red = png("red")
    blue = png("blue")
    archive = archive_at(tmp_path, [
        ("rps-cv-images/rock/z.png", red),
        ("rock/b.png", red),
        ("rock/a.png", png("red", compression=0)),
        ("scissors/z.png", png("green")),
        ("paper/a.png", blue),
        ("rock/c.png", blue),
        ("README.txt", b"acquisition notice"),
    ])
    assert red != png("red", compression=0)
    root = tmp_path / "workspace"
    run = prepare_dataset(archive, root)
    rows = read_inventory(run.inventory)
    assert len(rows) == 6
    assert len({r["image_id"] for r in rows.values()}) == 6
    representative = rows["rock/a.png"]
    assert representative["disposition"] == "eligible"
    for path in ("rock/b.png", "rps-cv-images/rock/z.png"):
        duplicate = rows[path]
        assert duplicate["disposition"] == "excluded"
        assert duplicate["reason_code"] == "exact_duplicate"
        assert duplicate["representative_id"] == representative["image_id"]
        assert duplicate["pixel_sha256"] == representative["pixel_sha256"]
        assert duplicate["group_id"] == representative["group_id"]
    assert rows["rock/b.png"]["file_sha256"] == rows["rps-cv-images/rock/z.png"]["file_sha256"]
    assert rows["rock/b.png"]["file_sha256"] != representative["file_sha256"]
    conflict = [rows["paper/a.png"], rows["rock/c.png"]]
    assert {r["source_label"] for r in conflict} == {"paper", "rock"}
    assert {r["group_id"] for r in conflict} == {conflict[0]["group_id"]}
    assert all(r["disposition"] == "pending" and r["reason_code"] == "conflicting_duplicate_labels" for r in conflict)
    assert run.summary["dispositions"] == {"eligible": 2, "excluded": 2, "pending": 2}
    assert run.summary["non_image_files"] == ["README.txt"]
    assert run.archive_copy.read_bytes() == archive.read_bytes()
    assert json.loads(run.provenance.read_text(encoding="utf-8"))["sha256"] == run.dataset_id
    first = [p.read_bytes() for p in (run.provenance, run.inventory, run.report)]
    rerun = prepare_dataset(archive, root)
    assert first == [p.read_bytes() for p in (rerun.provenance, rerun.inventory, rerun.report)]


def test_decode_and_label_anomalies_are_accounted_for(tmp_path):
    jpeg = io.BytesIO()
    Image.new("RGB", (300, 200), "orange").save(jpeg, format="JPEG")
    archive = archive_at(tmp_path, [
        ("rock/corrupt.png", b"invalid PNG"),
        ("paper/mode.png", png(128, mode="L")),
        ("scissors/size.png", png("black", size=(20, 30))),
        ("unknown/valid.png", png("yellow")),
        ("rock/mislabeled.png", jpeg.getvalue()),
        ("rock/unexpected.jpg", jpeg.getvalue()),
    ])
    run = prepare_dataset(archive, tmp_path / "workspace")
    rows = read_inventory(run.inventory)
    corrupt = rows["rock/corrupt.png"]
    assert corrupt["disposition"] == "excluded"
    assert corrupt["decode_status"] == "failed"
    assert corrupt["file_sha256"] and not corrupt["pixel_sha256"]
    assert rows["paper/mode.png"]["mode"] == "L"
    assert rows["scissors/size.png"]["width"] == "20"
    assert rows["unknown/valid.png"]["source_label"] == "unknown"
    assert not rows["unknown/valid.png"]["label"]
    assert rows["rock/mislabeled.png"]["format"] == "JPEG"
    assert rows["rock/unexpected.jpg"]["disposition"] == "pending"
    assert run.summary["dispositions"] == {"eligible": 0, "excluded": 1, "pending": 5}


def test_malformed_path_still_preserves_conflicting_source_claim(tmp_path):
    image = png("red")
    archive = archive_at(tmp_path, [("rock/a.png", image), ("paper/sub/a.png", image)])
    run = prepare_dataset(archive, tmp_path / "workspace")
    rows = read_inventory(run.inventory)
    assert rows["paper/sub/a.png"]["source_label"] == "paper"
    assert rows["paper/sub/a.png"]["label"] == ""
    assert {row["reason_code"] for row in rows.values()} == {"conflicting_duplicate_labels"}
    assert run.summary["dispositions"] == {"eligible": 0, "excluded": 0, "pending": 2}


def test_incomplete_extraction_can_resume_from_verified_members(tmp_path):
    archive = archive_at(tmp_path, [("rock/a.png", png("red")), ("paper/b.png", png("blue"))])
    root = tmp_path / "workspace"
    run = prepare_dataset(archive, root)
    expected = run.inventory.read_bytes()
    (run.extracted.parent / ".complete").unlink()
    (run.extracted / "paper/b.png").unlink()
    rerun = prepare_dataset(archive, root)
    assert rerun.inventory.read_bytes() == expected
    assert (rerun.extracted.parent / ".complete").is_file()


@pytest.mark.parametrize("name", ["../outside.png", "/absolute.png", "C:/drive.png", "rock\\escape.png", "rock/CON.png", "rock/x.png:stream", "rock/trim. /x.png"])
def test_unsafe_member_is_rejected_before_publication(tmp_path, name):
    member = ZipInfo("placeholder")
    member.filename = name  # Preserve the exact spelling on Windows.
    archive = archive_at(tmp_path, [("rock/valid.png", png("red")), (member, b"unsafe")])
    root = tmp_path / "workspace"
    with pytest.raises(ValueError, match="Unsafe ZIP"):
        prepare_dataset(archive, root)
    assert not (root / "data" / "manifests").exists()
    assert not (tmp_path / "outside.png").exists()


@pytest.mark.parametrize("paths", [("rock/a.png", "rock/a.png"), ("rock/a.png", "Rock/b.png"), ("rock", "rock/a.png")])
def test_duplicate_case_alias_and_parent_collision_fail(tmp_path, paths):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Duplicate name:.*", category=UserWarning)
        archive = archive_at(tmp_path, [(path, b"content") for path in paths])
    with pytest.raises(ValueError, match="ZIP path collision"):
        prepare_dataset(archive, tmp_path / "workspace")


def test_symlink_member_is_rejected(tmp_path):
    member = ZipInfo("rock/link.png")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive = archive_at(tmp_path, [(member, b"../../outside.png")])
    with pytest.raises(ValueError, match="Unsafe ZIP"):
        prepare_dataset(archive, tmp_path / "workspace")


@pytest.mark.parametrize("target", ["archive", "extracted"])
def test_tampered_raw_data_fails_on_rerun(tmp_path, target):
    archive = archive_at(tmp_path, [("rock/a.png", png("red"))])
    root = tmp_path / "workspace"
    run = prepare_dataset(archive, root)
    original = run.inventory.read_bytes()
    path = run.archive_copy if target == "archive" else run.extracted / "rock/a.png"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        prepare_dataset(archive, root)
    assert run.inventory.read_bytes() == original


def test_missing_source_produces_no_outputs(tmp_path):
    root = tmp_path / "workspace"
    with pytest.raises(FileNotFoundError):
        prepare_dataset(tmp_path / "missing.zip", root)
    assert not root.exists()
