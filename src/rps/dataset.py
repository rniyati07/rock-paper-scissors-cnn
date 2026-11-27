"""Acquire and inventory one version of the approved DrGFreeman archive."""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
import hashlib
from io import BytesIO
import json
import ntpath
from pathlib import Path
from random import Random
import shutil
import stat
from tempfile import TemporaryDirectory
import unicodedata
import warnings
from zipfile import ZipFile

import numpy as np
from PIL import Image


CLASSES = ("rock", "paper", "scissors")
NESTED_ROOT = "rps-cv-images"
SOURCE_URL = "https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors"


@dataclass
class ImageRow:
    dataset_id: str
    image_id: str
    relative_path: str
    size_bytes: int
    source_label: str = ""
    label: str = ""
    format: str = ""
    width: int | None = None
    height: int | None = None
    mode: str = ""
    channels: int | None = None
    decode_status: str = ""
    file_sha256: str = ""
    pixel_sha256: str = ""
    session_id: str = ""
    session_evidence: str = ""
    group_id: str = ""
    representative_id: str = ""
    disposition: str = ""
    reason_code: str = ""
    review_ref: str = ""
    issues: str = ""


@dataclass(frozen=True)
class DatasetRun:
    dataset_id: str
    archive_copy: Path
    extracted: Path
    provenance: Path
    inventory: Path
    report: Path
    summary: dict


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_members(archive):
    """Preflight ZIP names and directory aliases with Windows path rules."""
    entries = {}
    paths = {}
    files = {}
    for member in archive.infolist():
        name = member.filename
        parts = name.removesuffix("/").split("/")
        kind = stat.S_IFMT(member.external_attr >> 16)
        expected_kind = stat.S_IFDIR if member.is_dir() else stat.S_IFREG
        if (
            name != member.orig_filename
            or "\\" in name
            or ":" in name
            or any(not part or part in (".", "..") or ntpath.isreserved(part) for part in parts)
            or member.external_attr & 0x400
            or kind not in (0, expected_kind)
            or member.flag_bits & 1
        ):
            raise ValueError(f"Unsafe ZIP member: {name!r}")

        for length in range(1, len(parts) + 1):
            spelling = "/".join(parts[:length])
            key = unicodedata.normalize("NFC", spelling).casefold()
            is_dir = length < len(parts) or member.is_dir()
            if paths.setdefault(key, (spelling, is_dir)) != (spelling, is_dir):
                raise ValueError(f"ZIP path collision: {name!r}")
        if key in entries:
            raise ValueError(f"ZIP path collision: {name!r}")
        entries[key] = member
        if not member.is_dir():
            files[spelling] = member
    return dict(sorted(files.items()))


def ensure_directory(path):
    if path.is_symlink() or path.is_junction() or (path.exists() and not path.is_dir()):
        raise ValueError(f"Unsafe working directory: {path}")


def publish_raw(source, release, dataset_id):
    ensure_directory(release)
    copy = release / "archive.zip"
    extracted = release / "extracted"
    complete = release / ".complete"
    ensure_directory(extracted)
    finalized = complete.is_file()
    release.mkdir(exist_ok=True)
    if copy.is_symlink() or complete.is_symlink():
        raise ValueError("Raw archive integrity check failed: unexpected link")
    if not copy.exists() and not finalized and not extracted.exists():
        with source.open("rb") as stream, copy.open("xb") as target:
            shutil.copyfileobj(stream, target)
    if not copy.is_file() or sha256_file(copy) != dataset_id:
        raise ValueError("Raw archive integrity check failed")
    if finalized and not extracted.is_dir():
        raise ValueError("Extracted data integrity check failed: missing directory")

    extracted.mkdir(exist_ok=True)
    extraction_root = extracted.resolve()
    with ZipFile(copy) as archive:
        members = safe_members(archive)
        seen = set()
        for directory, subdirs, names in extracted.walk():
            for name in subdirs:
                ensure_directory(directory / name)
            for name in names:
                path = directory / name
                relative = path.relative_to(extracted).as_posix()
                if path.is_symlink() or not path.is_file() or relative not in members:
                    raise ValueError(f"Extracted data integrity check failed: {relative}")
                with archive.open(members[relative]) as stream:
                    expected = hashlib.file_digest(stream, "sha256").hexdigest()
                if sha256_file(path) != expected:
                    raise ValueError(f"Extracted data integrity check failed: {relative}")
                seen.add(relative)

        missing = members.keys() - seen
        if missing and finalized:
            raise ValueError("Extracted data integrity check failed: missing members")
        if sum(members[name].file_size for name in missing) > shutil.disk_usage(extracted).free:
            raise ValueError("Insufficient free disk space for extraction")
        for relative in sorted(missing):
            path = extracted / relative
            if not path.resolve().is_relative_to(extraction_root):
                raise ValueError(f"Unsafe ZIP member: {relative!r}")
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(members[relative]) as stream, path.open("xb") as target:
                shutil.copyfileobj(stream, target)
    if not finalized:
        # Closing this marker publishes the raw release for subsequent inventory runs.
        complete.touch(exist_ok=False)
    return copy, extracted

def inspect_image(path, relative, dataset_id):
    parts = relative.split("/")
    label_path = parts[1:] if parts[0] == NESTED_ROOT else parts
    source_label = label_path[0] if len(label_path) > 1 else ""
    label = source_label if len(label_path) == 2 and source_label in CLASSES else ""
    image_id = hashlib.sha256(f"{dataset_id}/{relative}".encode("utf-8")).hexdigest()
    row = ImageRow(dataset_id, image_id, relative, path.stat().st_size, source_label, label)
    row.file_sha256 = sha256_file(path)
    issues = [] if label else ["unmapped_label"]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                row.format = image.format or ""
                row.width, row.height = image.size
                row.mode = image.mode
                row.channels = len(image.getbands())
                if image.format == "PNG":
                    image.verify()
            with Image.open(path) as image:
                image.load()
                # Palette indices are not colors. Preserve other modes without lossy conversion.
                pixels = image.convert("RGBA") if image.mode in ("P", "PA") else image
                convention = json.dumps([pixels.size, pixels.mode, pixels.getbands()]).encode("utf-8")
                digest = hashlib.sha256(convention + b"\0")
                digest.update(pixels.tobytes())
                row.pixel_sha256 = digest.hexdigest()
        row.decode_status = "ok"
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        row.decode_status = "failed"
        issues.append(f"decode_failed:{type(error).__name__}")
    if row.decode_status == "ok" and (row.format, row.mode, row.width, row.height) != ("PNG", "RGB", 300, 200):
        issues.append("unexpected_image_properties")
    row.issues = "|".join(issues)
    return row


def disposition_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        # Byte-identical decodable files share their pixel hash; failures have only a file hash.
        row.group_id = f"pixel:{row.pixel_sha256}" if row.pixel_sha256 else f"file:{row.file_sha256}"
        groups[row.group_id].append(row)

    for group in groups.values():
        labels = {row.source_label for row in group if row.source_label in CLASSES}
        conflicting = len(labels) > 1
        representative = next((row for row in group if not row.issues), None)
        for row in group:
            if conflicting:
                row.disposition = "pending"
                row.reason_code = "conflicting_duplicate_labels"
            elif row.decode_status == "failed":
                row.disposition = "excluded"
                row.reason_code = "decode_failed"
            elif row.issues:
                row.disposition = "pending"
                row.reason_code = "validation_review"
            else:
                row.representative_id = representative.image_id
                row.disposition = "eligible" if row is representative else "excluded"
                row.reason_code = "" if row is representative else "exact_duplicate"
    return groups


def duplicate_counts(rows, field):
    hashes = defaultdict(list)
    for row in rows:
        if value := getattr(row, field):
            hashes[value].append(row)
    repeats = [group for group in hashes.values() if len(group) > 1]
    tree_counts = [
        Counter(row.relative_path.startswith(NESTED_ROOT + "/") for row in group)
        for group in repeats
    ]
    return {
        "distinct": len(hashes),
        "repeated_groups": len(repeats),
        "additional_entries": sum(len(group) - 1 for group in repeats),
        "groups_across_trees": sum(len(counts) > 1 for counts in tree_counts),
        "groups_within_a_tree": sum(max(counts.values()) > 1 for counts in tree_counts),
    }


def summarize(rows, groups, metadata_files, dataset_id):
    states = ("eligible", "excluded", "pending")
    labels = (*CLASSES, "unmapped")
    class_counts = Counter((row.label or "unmapped", row.disposition) for row in rows)
    properties = Counter(
        f"{row.format} {row.mode} {row.width}x{row.height} ({row.channels} channels)"
        for row in rows if row.decode_status == "ok"
    )
    nested = {
        row.relative_path.removeprefix(NESTED_ROOT + "/"): row
        for row in rows if row.relative_path.startswith(NESTED_ROOT + "/")
    }
    pairs = [(row, nested[row.relative_path]) for row in rows if row.relative_path in nested]
    return {
        "dataset_id": dataset_id,
        "image_entries": len(rows),
        "non_image_files": metadata_files,
        "dispositions": {state: sum(class_counts[label, state] for label in labels) for state in states},
        "by_class": {label: {state: class_counts[label, state] for state in states} for label in labels},
        "root_class_counts": dict(Counter(r.label for r in rows if not r.relative_path.startswith(NESTED_ROOT + "/"))),
        "nested_class_counts": dict(Counter(r.label for r in nested.values())),
        "decoded_properties": dict(properties),
        "reason_counts": dict(Counter(row.reason_code for row in rows if row.reason_code)),
        "issue_counts": dict(Counter(issue for row in rows for issue in row.issues.split("|") if issue)),
        "file_hashes": duplicate_counts(rows, "file_sha256"),
        "pixel_hashes": duplicate_counts(rows, "pixel_sha256"),
        "exact_groups": len(groups),
        "same_path_tree_pairs": {
            "pairs": len(pairs),
            "identical_bytes": sum(a.file_sha256 == b.file_sha256 for a, b in pairs),
            "identical_pixels": sum(bool(a.pixel_sha256) and a.pixel_sha256 == b.pixel_sha256 for a, b in pairs),
        },
    }


def write_outputs(run, provenance, rows):
    for path in (run.provenance.parent, run.report.parent):
        path.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".inventory-", dir=run.inventory.parent) as temp:
        working = Path(temp)
        csv_path = working / run.inventory.name
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(ImageRow)], lineterminator="\n")
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
        json_path = working / run.provenance.name
        json_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        csv_path.replace(run.inventory)
        json_path.replace(run.provenance)
    run.summary["inventory_sha256"] = sha256_file(run.inventory)
    with TemporaryDirectory(prefix=".validation-", dir=run.report.parent) as temp:
        report = Path(temp) / run.report.name
        report.write_text(json.dumps(run.summary, indent=2) + "\n", encoding="utf-8", newline="\n")
        report.replace(run.report)


def prepare_dataset(source, workspace):
    source = Path(source)
    workspace = Path(workspace)
    source_stat = source.stat()
    dataset_id = sha256_file(source)
    with ZipFile(source) as archive:
        members = safe_members(archive)
    raw = workspace / "data" / "raw"
    manifests = workspace / "data" / "manifests" / dataset_id
    reports = workspace / "artifacts" / "data" / dataset_id
    for location in (workspace / "data", raw, manifests.parent, manifests, reports.parent.parent, reports.parent, reports):
        ensure_directory(location)
    raw.mkdir(parents=True, exist_ok=True)
    copy, extracted = publish_raw(source, raw / dataset_id, dataset_id)
    rows = []
    metadata_files = []
    image_extensions = Image.registered_extensions()
    for relative in members:
        if Path(relative).suffix.lower() in image_extensions:
            rows.append(inspect_image(extracted / relative, relative, dataset_id))
        else:
            metadata_files.append(relative)
    if not rows:
        raise ValueError("Archive contains no recognized image entries")
    groups = disposition_groups(rows)
    summary = summarize(rows, groups, metadata_files, dataset_id)
    run = DatasetRun(dataset_id, copy, extracted, manifests / "provenance.json", manifests / "inventory.csv",
                     reports / "validation.json", summary)
    provenance = {
        "source_url": SOURCE_URL,
        "source_evidence": "User-provided Kaggle download",
        "dataset": "Rock-Paper-Scissors Images",
        "publisher": "DrGFreeman",
        "attribution": "Julien de la Bruère-Terreault (DrGFreeman)",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "original_filename": source.name,
        "archive_size_bytes": source_stat.st_size,
        "sha256": dataset_id,
        "download_file_mtime_utc": datetime.fromtimestamp(source_stat.st_mtime, UTC).isoformat(),
        "retrieved_on": None,
        "published_release_id": None,
        "archived_metadata_files": metadata_files,
    }
    write_outputs(run, provenance, rows)
    return run


def load_validated_inventory(workspace, dataset_id, inventory_sha256):
    raw = workspace / "data" / "raw" / dataset_id
    manifests = workspace / "data" / "manifests" / dataset_id
    inventory = manifests / "inventory.csv"
    provenance = json.loads((manifests / "provenance.json").read_text(encoding="utf-8"))
    if provenance["sha256"] != dataset_id or sha256_file(raw / "archive.zip") != dataset_id:
        raise ValueError("Archive identity differs from the approved release")
    if sha256_file(inventory) != inventory_sha256:
        raise ValueError("Inventory identity differs from the approved inventory")
    with inventory.open(encoding="utf-8", newline="") as handle:
        rows = []
        for values in csv.DictReader(handle):
            for key in ("width", "height", "channels"):
                values[key] = int(values[key]) if values[key] else None
            values["size_bytes"] = int(values["size_bytes"])
            rows.append(ImageRow(**values))
    if {row.dataset_id for row in rows} != {dataset_id}:
        raise ValueError("Inventory contains unexpected dataset IDs")
    if len({row.image_id for row in rows}) != len(rows):
        raise ValueError("Duplicate image IDs in inventory")
    return raw / "extracted", rows


def sample_class_rows(rows, count, seed):
    rng = Random(seed)
    ordered = sorted(rows, key=lambda row: row.image_id)
    samples = {}
    for label in CLASSES:
        members = [row for row in ordered if row.label == label]
        samples[label] = rng.sample(members, min(count, len(members)))
    return samples


def difference_hash(image):
    gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    bits = np.packbits(gray[:, 1:] > gray[:, :-1])
    return int.from_bytes(bits.tobytes(), "big")


def screen_hashes(hashes, radius):
    counts = [0] * 65
    candidates = []
    ordered = sorted(hashes.items())
    for index, (left, first) in enumerate(ordered):
        for right, second in ordered[index + 1:]:
            distance = (first ^ second).bit_count()
            counts[distance] += 1
            if distance <= radius:
                candidates.append((left, right, distance))
    return counts, sorted(candidates, key=lambda pair: (pair[2], pair[0], pair[1]))


def border_mask(height, width, border_width):
    if not 0 < 2 * border_width < min(height, width):
        raise ValueError("Border width must leave a nonempty interior")
    mask = np.ones((height, width), dtype=bool)
    mask[border_width:-border_width, border_width:-border_width] = False
    return mask


def measure_image(path, border_width, expected_sha256=None):
    content = path.read_bytes()
    if expected_sha256 and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(f"Image differs from approved inventory: {path}")
    with Image.open(BytesIO(content)) as image:
        pixels = np.asarray(image.convert("RGB"))
        image_hash = difference_hash(image)
        metadata_keys = sorted(image.info)
        exif_tags = sorted(image.getexif())
    height, width, _ = pixels.shape
    border = pixels[border_mask(height, width, border_width)]
    red, green, blue = np.moveaxis(pixels, -1, 0)
    # Rec. 709 coefficients on stored RGB, used here as a brightness proxy.
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    luminance_histogram = np.bincount(np.rint(luminance).astype(np.uint8).ravel(), minlength=256)
    return {
        "rgb_histograms": np.stack([np.bincount(channel.ravel(), minlength=256) for channel in (red, green, blue)]),
        "luminance_histogram": luminance_histogram,
        "luminance_quantiles": np.quantile(luminance, [0.05, 0.5, 0.95]).tolist(),
        "brightness": float(luminance.mean()),
        "edge_strength": float((np.abs(np.diff(luminance, axis=0)).mean()
                                + np.abs(np.diff(luminance, axis=1)).mean()) / 2),
        "border_mean_rgb": border.mean(axis=0).tolist(),
        "border_green_share": float(((border[:, 1] > border[:, 0]) & (border[:, 1] > border[:, 2])).mean()),
        "dhash": image_hash,
        "metadata_keys": metadata_keys,
        "exif_tags": exif_tags,
    }


def connected_groups(rows, links):
    parents = {row.image_id: row.image_id for row in rows}

    def root(image_id):
        if image_id not in parents:
            raise ValueError(f"Unknown image in grouping: {image_id}")
        while parents[image_id] != image_id:
            parents[image_id] = parents[parents[image_id]]
            image_id = parents[image_id]
        return image_id

    exact_links = [(row.image_id, row.representative_id) for row in rows if row.representative_id]
    for left, right in [*exact_links, *links]:
        first, second = sorted((root(left), root(right)))
        parents[second] = first
    return {image_id: f"group:{root(image_id)}" for image_id in sorted(parents)}


def write_csv(path, records, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".csv-", dir=path.parent) as temp:
        staged = Path(temp) / path.name
        with staged.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
        staged.replace(path)


def sync_reviews(path, cases, choices):
    review_fields = ("decision", "reviewer", "rationale")
    columns = [*cases[0], *review_fields] if cases else ["case_id", *review_fields]
    if not path.exists():
        write_csv(path, [case | dict.fromkeys(review_fields, "") for case in cases], columns)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            raise ValueError(f"Review columns differ from generated schema: {path}")
        records = list(reader)
    expected = {case["case_id"]: case for case in cases}
    actual = {record["case_id"]: record for record in records}
    if len(records) != len(actual) or actual.keys() != expected.keys():
        raise ValueError(f"Review case IDs changed: {path}")
    for case_id, record in actual.items():
        if None in record or None in record.values():
            raise ValueError(f"Review row has the wrong number of fields: {case_id}")
        if any(record[key] != value for key, value in expected[case_id].items()):
            raise ValueError(f"Review case evidence changed: {case_id}")
        decision, reviewer, rationale = (record[key].strip() for key in review_fields)
        if decision and (decision not in choices or not reviewer or not rationale):
            raise ValueError(f"Review requires a valid decision, reviewer and rationale: {case_id}")
        record.update(zip(review_fields, (decision, reviewer, rationale)))
    return [actual[case["case_id"]] for case in cases]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Downloaded DrGFreeman ZIP file")
    args = parser.parse_args()
    run = prepare_dataset(args.archive, Path.cwd())
    print(f"Archive SHA-256: {run.dataset_id}")
    print(f"Inventory: {run.inventory}")
    print(f"Validation report: {run.report}")
    print(f"Image entries: {run.summary['image_entries']} · {run.summary['dispositions']}")


if __name__ == "__main__":
    main()
