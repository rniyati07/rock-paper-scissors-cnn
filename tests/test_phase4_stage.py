"""TP-08 staging and TP-09 integrity gates on temporary development fixtures."""

import io
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

from PIL import Image
import pytest

from src.rps import phase4
from src.rps.dataset import write_csv


@pytest.fixture
def workspace(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    config = phase4.load_json(repo / "configs/data.json")
    for name in (*phase4.SOURCES, "requirements.txt"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repo / name).read_bytes())
    raw = tmp_path / "data/raw" / config["dataset_id"] / "extracted"
    raw.mkdir(parents=True)
    rows = []
    for part in ("train", "validation", "test"):
        for label in config["contract"]["class_mapping"]:
            key = phase4.digest(f"{part}-{label}".encode())
            name = key + ".png"
            stream = io.BytesIO()
            Image.new("RGB", (6, 4), tuple(bytes.fromhex(key[:6]))).save(stream, format="PNG")
            data = stream.getvalue()
            if part != "test":
                (raw / name).write_bytes(data)
            rows.append(dict(image_id=key, split=part, relative_path=name, label=label, group_id=key,
                             file_sha256=phase4.digest(data)))
    manifest = tmp_path / "split.csv"
    write_csv(manifest, rows, list(rows[0]))
    split = {"split_version": "fixture", "sha256": phase4.digest(manifest.read_bytes()), "manifest": str(manifest),
             "counts": {"train": 3, "validation": 3, "test": 3}}
    directory = tmp_path / "artifacts/data/phase4/fixture"
    directory.mkdir(parents=True)
    (directory / "grid.png").write_bytes(data)
    identity = phase4.digest(phase4.json_bytes(config["contract"]))
    preview = {
        "contract": config["contract"], "contract_sha256": identity, "split_sha256": split["sha256"],
        "preprocessing_sha256": phase4.digest((tmp_path / phase4.SOURCES[1]).read_bytes()),
        "preview_files": {"grid.png": phase4.digest(data)},
    }
    (directory / "preview-summary.json").write_bytes(phase4.json_bytes(preview))
    return tmp_path, config, split, identity, rows


def test_contract_requires_matching_preview_and_owner_digest(workspace):
    root, config, split, identity, _ = workspace
    with pytest.raises(FileNotFoundError):
        phase4.contract_record(root, config, split)
    with pytest.raises(ValueError, match="Reviewed preview"):
        phase4.contract_record(root, config, split, "wrong")
    path, original = phase4.contract_record(root, config, split, identity)
    assert phase4.contract_record(root, config, split) == (path, original)
    (root / "artifacts/data/phase4/fixture/grid.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="figure identity"):
        phase4.contract_record(root, config, split, identity)


def test_verified_bundle_contains_only_development_data(workspace, monkeypatch):
    root, config, split, identity, rows = workspace
    phase4.contract_record(root, config, split, identity)
    monkeypatch.setattr(phase4, "subprocess", SimpleNamespace(check_output=lambda *args, **kwargs: "fixture-revision"))
    wrong_counts = split | {"counts": {"train": 4, "validation": 3}}
    with pytest.raises(ValueError, match="counts differ"):
        phase4.stage(root, config, wrong_counts)
    audit = phase4.stage(root, config, split)
    archive = Path(audit["archive"])
    hosted = root / "hosted"
    with ZipFile(archive) as bundle:
        bundle.extractall(hosted)
        names = set(bundle.namelist())
    assert {name for name in names if name.startswith("images/")} == {
        f"images/{r['image_id']}.png" for r in rows if r["split"] != "test"
    }
    assert audit["counts"] == {"train": 3, "validation": 3}
    with pytest.raises(ValueError, match="Uploaded ZIP"):
        phase4.verify_stage(hosted, archive, "wrong")
    report = phase4.verify_stage(hosted, archive, audit["archive_sha256"])
    assert report["status"] == "passed"
    assert report["shape"] == [config["contract"]["height"], config["contract"]["width"], 3]
    assert report["dtype"] == "float32"
    assert report["max_abs_difference"] <= 1e-5
    unexpected = hosted / "unfiltered.csv"
    unexpected.touch()
    with pytest.raises(ValueError, match="Unexpected extracted"):
        phase4.verify_stage(hosted, archive, audit["archive_sha256"])
    unexpected.unlink()
    (hosted / "development.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Content mismatch"):
        phase4.verify_stage(hosted, archive, audit["archive_sha256"])


def test_rejects_incompatible_hosted_dependencies(monkeypatch):
    monkeypatch.setattr(phase4, "version", lambda name: "incompatible")
    with pytest.raises(ValueError, match="Package mismatch"):
        phase4.runtime(dict.fromkeys(phase4.PACKAGES, "expected"))
