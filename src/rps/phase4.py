"""Phase 4 commands. Run locally from the repository root unless verifying a staged bundle."""

import argparse
from collections import Counter
import csv
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZipInfo


SOURCES = ("src/rps/dataset.py", "src/rps/preprocessing.py", "src/rps/phase4.py")
PACKAGES = ("tensorflow", "keras", "numpy", "Pillow")
STAGE_FIELDS = ("image_id", "relative_path", "label", "group_id", "file_sha256", "split")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def contract_record(root, config, split, expected=None):
    from .dataset import _publish_immutable, filesystem_path, sha256_file
    from .preprocessing import validate_contract

    contract = config["contract"]
    validate_contract(contract)
    identity = digest(json_bytes(contract))
    directory = root / "artifacts/data/phase4" / split["split_version"]
    source_hash = sha256_file(root / "src/rps/preprocessing.py")
    version_id = digest(json_bytes([split["sha256"], identity, source_hash]))
    path = directory / "contracts" / version_id / "frozen.json"
    if expected is not None:
        preview = load_json(filesystem_path(directory / "preview-summary.json"))
        if (identity != expected or preview["contract_sha256"] != identity or preview["contract"] != contract
                or preview["split_sha256"] != split["sha256"] or preview["preprocessing_sha256"] != source_hash):
            raise ValueError("Reviewed preview, configuration and frozen split must match")
        files = preview["preview_files"]
        if not files:
            raise ValueError("Missing preview evidence")
        for name, checksum in files.items():
            if Path(name).name != name or sha256_file(directory / name) != checksum:
                raise ValueError("Preview figure identity mismatch")
        _publish_immutable(path.parent / "approved-preview.json", json_bytes(preview))
        _publish_immutable(path, json_bytes({"contract": contract, "contract_sha256": identity, "version": version_id,
                                             "split_sha256": split["sha256"], "preprocessing_sha256": source_hash,
                                             "preview_sha256": digest(json_bytes(preview)), "preview_files": files}))
    record = load_json(filesystem_path(path))
    if (record["version"] != version_id or record["contract"] != contract or record["contract_sha256"] != identity
            or record["split_sha256"] != split["sha256"] or record["preprocessing_sha256"] != source_hash
            or record["preview_sha256"] != sha256_file(path.parent / "approved-preview.json")):
        raise ValueError("Frozen preprocessing contract mismatch")
    return path, record


def runtime(required):
    if sys.version_info[:2] != (3, 13):
        raise ValueError("This bundle requires Python 3.13")
    actual = {package: version(package) for package in PACKAGES}
    if actual != required:
        raise ValueError(f"Package mismatch. Expected {required}. Installed {actual}")
    return {"python": platform.python_version(), "packages": actual}


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=STAGE_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def reference_tensor(path, contract):
    import tensorflow as tf
    from .preprocessing import preprocess

    with tf.device("/CPU:0"):
        return preprocess(path, contract).numpy()


def stage(root, config, split):
    import numpy as np
    from .dataset import _publish_immutable, filesystem_path, sha256_file

    _, frozen = contract_record(root, config, split)
    pins = dict(line.split("==") for line in (root / "requirements.txt").read_text().splitlines()
                if "==" in line and not line.lstrip().startswith("#"))
    versions = runtime({name: pins[name] for name in PACKAGES})
    manifest = filesystem_path(split["manifest"]).read_bytes()
    if digest(manifest) != split["sha256"]:
        raise ValueError("Frozen manifest checksum mismatch")
    reader = csv.DictReader(io.StringIO(manifest.decode("utf-8"), newline=""))
    allowed = [row for row in reader if row["split"] in {"train", "validation"}]
    counts = Counter(row["split"] for row in allowed)
    if counts != {name: split["counts"][name] for name in ("train", "validation")}:
        raise ValueError("Staging counts differ from the frozen split audit")
    reference = min((row for row in allowed if row["split"] == "train"), key=lambda row: row["image_id"])
    raw = (root / "data/raw" / config["dataset_id"] / "extracted").resolve()
    output = root / "artifacts/data/phase4" / split["split_version"]
    rows = []
    checksums = {}
    with TemporaryDirectory(dir=filesystem_path(output)) as temp:
        archive = Path(temp) / "phase4-stage.zip"
        with ZipFile(archive, "w") as bundle:
            def add(name, content, expected=None):
                if name in checksums:
                    raise ValueError("Duplicate bundle member")
                checksum = digest(content)
                if expected is not None and checksum != expected:
                    raise ValueError(f"Staged image checksum mismatch for {name}")
                bundle.writestr(ZipInfo(name), content)
                checksums[name] = checksum

            for row in allowed:
                source = (raw / row["relative_path"]).resolve()
                if not source.is_relative_to(raw) or not re.fullmatch("[0-9a-f]{64}", row["image_id"]):
                    raise ValueError("Invalid source identity")
                name = f"images/{row['image_id']}.png"
                content = filesystem_path(source).read_bytes()
                add(name, content, row["file_sha256"])
                if row is reference:
                    reference_bytes = content
                rows.append({key: row[key] for key in STAGE_FIELDS} | {"relative_path": name})
            array = reference_tensor(io.BytesIO(reference_bytes), frozen["contract"])
            stream = io.BytesIO()
            np.save(stream, array, allow_pickle=False)
            add("reference.npy", stream.getvalue())
            add("development.csv", csv_bytes(rows))
            add("contract.json", json_bytes(frozen))
            for name in SOURCES:
                add(name, (root / name).read_bytes())
            if checksums["src/rps/preprocessing.py"] != frozen["preprocessing_sha256"]:
                raise ValueError("Preprocessing source changed during staging")
            index = {
                "dataset_id": config["dataset_id"], "split_sha256": split["sha256"],
                "contract_sha256": frozen["contract_sha256"], "runtime": versions,
                "git_revision": subprocess.check_output(
                    ["git", "-c", f"safe.directory={root.resolve().as_posix()}", "rev-parse", "HEAD"],
                    cwd=root, text=True).strip(),
                "reference_id": reference["image_id"], "counts": dict(counts),
                "files": checksums,
            }
            index_bytes = json_bytes(index)
            bundle.writestr(ZipInfo("stage.json"), index_bytes)
        with ZipFile(archive) as bundle:
            if set(bundle.namelist()) != set(checksums) | {"stage.json"} or bundle.testzip() is not None:
                raise ValueError("Staging archive integrity check failed")
        identity = digest(index_bytes)
        directory = output / "staging" / identity
        target = directory / archive.name
        _publish_immutable(target, archive.read_bytes())
    audit = {"archive": target.name, "archive_sha256": sha256_file(target), "counts": index["counts"],
             "split_sha256": split["sha256"], "contract_sha256": frozen["contract_sha256"],
             "manifest_sha256": checksums["development.csv"], "source_sha256": {s: checksums[s] for s in SOURCES}}
    _publish_immutable(directory / "staging-audit.json", json_bytes(audit))
    return audit | {"archive": str(target)}


def verify_stage(root, archive, expected):
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError("Uploaded ZIP differs from local staging audit")
    index = load_json(root / "stage.json")
    observed_runtime = runtime(index["runtime"]["packages"])
    expected_names = set(index["files"]) | {"stage.json"}
    with ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)) or set(names) != expected_names:
            raise ValueError("Unexpected archive membership")
        if bundle.read("stage.json") != (root / "stage.json").read_bytes():
            raise ValueError("Extracted index differs from archive")
    local_files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if local_files - {"colab-verification.json"} != expected_names:
        raise ValueError("Unexpected extracted file membership. Use a fresh directory and python -B.")
    for name, checksum in index["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path.read_bytes()) != checksum:
            raise ValueError(f"Content mismatch for {name}")
    with (root / "development.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    images = set()
    labels = {"train": set(), "validation": set()}
    counts = Counter()
    for row in rows:
        name = row["relative_path"]
        if (row["split"] not in labels or not re.fullmatch("[0-9a-f]{64}", row["image_id"])
                or name != f"images/{row['image_id']}.png" or name in images
                or row["file_sha256"] != index["files"].get(name)):
            raise ValueError("Invalid filtered assignment")
        images.add(name)
        labels[row["split"]].add(row["label"])
        counts[row["split"]] += 1
    if expected_names != images | set(SOURCES) | {"contract.json", "development.csv", "reference.npy", "stage.json"}:
        raise ValueError("Bundle contains extra data")
    frozen = load_json(root / "contract.json")
    contract = frozen["contract"]
    if (digest(json_bytes(contract)) != index["contract_sha256"] or frozen["split_sha256"] != index["split_sha256"]
            or frozen["contract_sha256"] != index["contract_sha256"]
            or frozen["preprocessing_sha256"] != index["files"]["src/rps/preprocessing.py"]
            or counts != index["counts"] or any(value != set(contract["class_mapping"]) for value in labels.values())):
        raise ValueError("Split, class or contract identity mismatch")
    import numpy as np
    reference = next(r for r in rows if r["image_id"] == index["reference_id"] and r["split"] == "train")
    actual = reference_tensor(root / reference["relative_path"], contract)
    local = np.load(root / "reference.npy", allow_pickle=False)
    shape = (contract["height"], contract["width"], 3)
    if actual.shape != shape or local.shape != shape or actual.dtype != np.float32 or local.dtype != np.float32:
        raise ValueError("Reference tensor shape or dtype mismatch")
    if not all(np.isfinite(t).all() and t.min() >= 0 and t.max() <= 1 for t in (actual, local)):
        raise ValueError("Reference tensor outside normalized range")
    delta = float(np.max(np.abs(actual - local)))
    if not np.isfinite(delta) or delta > 1e-5:
        raise ValueError(f"Preprocessing parity failed with maximum absolute difference {delta}")
    report = {"status": "passed", "runtime": observed_runtime, "counts": counts, "archive_sha256": expected,
              "split_sha256": index["split_sha256"], "contract_sha256": index["contract_sha256"],
              "reference_id": index["reference_id"], "shape": list(actual.shape), "dtype": actual.dtype.name,
              "max_abs_difference": delta, "tolerance": 1e-5}
    (root / "colab-verification.json").write_bytes(json_bytes(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("split", "audit", "freeze", "approve-contract", "stage"):
        subparser = commands.add_parser(name)
        subparser.add_argument("--config", type=Path, default=Path("configs/data.json"))
        if name in {"freeze", "approve-contract"}:
            subparser.add_argument("--expected-sha256", required=True)
        if name == "audit":
            subparser.add_argument("--regenerate", action="store_true")
    hosted = commands.add_parser("verify-stage")
    hosted.add_argument("--root", type=Path, default=Path.cwd())
    hosted.add_argument("--archive", required=True, type=Path)
    hosted.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    if args.command == "verify-stage":
        result = verify_stage(args.root.resolve(), args.archive, args.expected_sha256)
    else:
        from .dataset import phase4_split

        root = Path.cwd()
        config = load_json(args.config)
        action = args.command if args.command in {"split", "audit", "freeze"} else "frozen"
        result = phase4_split(root, config, action, expected_sha256=getattr(args, "expected_sha256", None)
                              if action == "freeze" else None, regenerate=getattr(args, "regenerate", False))
        if args.command == "approve-contract":
            path, record = contract_record(root, config, result, args.expected_sha256)
            result = {"frozen_contract": str(path), "contract_sha256": record["contract_sha256"]}
        if args.command == "stage":
            result = stage(root, config, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
