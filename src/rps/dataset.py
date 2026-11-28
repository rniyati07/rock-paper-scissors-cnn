"""Acquire and inventory one version of the approved DrGFreeman archive."""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
import hashlib
from io import BytesIO, StringIO
import json
import ntpath
import os
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


def filesystem_path(path):
    """Use extended-length Windows paths for filesystem I/O."""
    path = Path(path)
    if os.name != "nt":
        return path
    absolute = os.path.abspath(path)
    if absolute.startswith(("\\\\?\\", "\\\\.\\")):
        return Path(absolute)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)


def sha256_file(path):
    with filesystem_path(path).open("rb") as stream:
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
    inventory = filesystem_path(run.inventory)
    provenance_path = filesystem_path(run.provenance)
    validation = filesystem_path(run.report)
    for path in (inventory, provenance_path, validation):
        path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".inventory-", dir=inventory.parent) as temp:
        working = Path(temp)
        csv_path = working / run.inventory.name
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(ImageRow)], lineterminator="\n")
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
        json_path = working / run.provenance.name
        json_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        csv_path.replace(inventory)
        json_path.replace(provenance_path)
    run.summary["inventory_sha256"] = sha256_file(inventory)
    with TemporaryDirectory(prefix=".validation-", dir=validation.parent) as temp:
        report = Path(temp) / run.report.name
        report.write_text(json.dumps(run.summary, indent=2) + "\n", encoding="utf-8", newline="\n")
        report.replace(validation)


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
    manifests = filesystem_path(workspace / "data" / "manifests" / dataset_id)
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
    path = filesystem_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".csv-", dir=path.parent) as temp:
        staged = Path(temp) / "rows.csv"
        with staged.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
        staged.replace(path)


def sync_reviews(path, cases, choices):
    path = filesystem_path(path)
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


SPLIT_PERCENT = {"train": 70, "validation": 15, "test": 15}
SPLIT_COLUMNS = ("dataset_id", "inventory_version", "split_version", "image_id", "relative_path",
                 "label", "group_id", "file_sha256", "pixel_sha256", "split")
SPLIT_ALGORITHM = "group-count-squared-deviation-v1"
SPLIT_ATTEMPTS = 16


def _identity_csv(path, digest):
    """Hash and parse the same snapshot even if a file is replaced concurrently."""
    content = filesystem_path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"Source SHA-256 mismatch: {path.name}")
    rows = list(csv.DictReader(StringIO(content.decode("utf-8"), newline="")))
    indexed = {row["image_id"]: row for row in rows}
    if not rows or len(indexed) != len(rows) or "" in indexed:
        raise ValueError(f"Missing or duplicate source image IDs: {path.name}")
    return indexed


def _eligible_split_rows(source, identity):
    version = identity["inventory_version"]
    inventory = _identity_csv(source / f"inventory-{version}.csv", identity["inventory_sha256"])
    groups = _identity_csv(source / f"groups-{version}.csv", identity["groups_sha256"])
    if inventory.keys() != groups.keys():
        raise ValueError("Inventory and group image IDs differ")
    eligible = {}
    paths = set()
    for image_id, row in sorted(inventory.items()):
        if row["dataset_id"] != identity["dataset_id"] or row["inventory_version"] != version:
            raise ValueError("Source release or inventory version mismatch")
        if any(row[key] != groups[image_id][key] for key in ("group_id", "label", "disposition")):
            raise ValueError("Reviewed group records disagree with inventory")
        if row["disposition"] == "excluded" and row["reason_code"]:
            continue
        if row["disposition"] != "eligible" or row["label"] not in CLASSES or not row["group_id"]:
            raise ValueError("Unresolved disposition, class or group")
        relative = row["relative_path"]
        if ("\\" in relative or ntpath.splitdrive(relative)[0] or
                any(part in ("", ".", "..") for part in relative.split("/")) or relative.casefold() in paths):
            raise ValueError("Unsafe or duplicate source path")
        paths.add(relative.casefold())
        if not all(_is_sha256(row[key]) for key in ("pixel_sha256", "file_sha256")):
            raise ValueError("Missing or invalid content hash")
        eligible[image_id] = {key: row[key] for key in SPLIT_COLUMNS if key not in ("split", "split_version")}
    return eligible


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _allocate_groups(eligible, seed):
    by_group = defaultdict(Counter)
    totals = Counter()
    for row in eligible.values():
        by_group[row["group_id"]][row["label"]] += 1
        totals[row["label"]] += 1
    if any(sum(label in counts for counts in by_group.values()) < len(SPLIT_PERCENT) for label in CLASSES):
        raise ValueError("Too few independent groups to cover each class in every split")

    for attempt in range(SPLIT_ATTEMPTS):
        rng = Random(seed + attempt)
        ordered = sorted(by_group)
        rng.shuffle(ordered)
        ordered.sort(key=lambda group: -by_group[group].total())
        counts = {split: Counter() for split in SPLIT_PERCENT}
        assigned = {}
        remaining_groups = Counter(label for group in ordered for label in by_group[group])
        for group in ordered:
            members = by_group[group]
            remaining_groups.subtract(members.keys())
            # The change in squared error is (error + group size)^2 - error^2.
            # Normalize each term by its population to balance total and class counts.
            def cost(split):
                fraction = SPLIT_PERCENT[split] / 100
                current = counts[split]
                size = members.total()
                delta = size * (2 * (current.total() - fraction * len(eligible)) + size) / len(eligible)
                for label, n in members.items():
                    delta += n * (2 * (current[label] - fraction * totals[label]) + n) / totals[label]
                return delta

            # Leave enough groups to cover the partitions still missing each class.
            candidates = [
                split for split in SPLIT_PERCENT
                if all(sum(counts[other][label] == 0 for other in SPLIT_PERCENT if other != split)
                       <= remaining_groups[label] for label in members)
            ]
            if not candidates:
                break
            rng.shuffle(candidates)
            split = min(candidates, key=cost)
            assigned[group] = split
            counts[split].update(members)
        if len(assigned) == len(by_group) and all(set(counts[split]) == set(CLASSES) for split in counts):
            return assigned, seed + attempt
    raise ValueError(f"Class coverage failed after {SPLIT_ATTEMPTS} seeded attempts; review group constraints")


def _split_csv(records):
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=SPLIT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return stream.getvalue().encode("utf-8")


def _audit_split(records, eligible, version):
    seen = set()
    by_class = {split: dict.fromkeys(CLASSES, 0) for split in SPLIT_PERCENT}
    owners = {key: {} for key in ("group_id", "file_sha256", "pixel_sha256")}
    for row in records:
        image_id, split = row["image_id"], row["split"]
        if image_id in seen or image_id not in eligible or split not in by_class:
            raise ValueError("Duplicate, unknown or invalid split assignment")
        source = eligible[image_id]
        if row["split_version"] != version or any(row[key] != value for key, value in source.items()):
            raise ValueError("Split row differs from its approved source")
        seen.add(image_id)
        for key, values in owners.items():
            if values.setdefault(row[key], split) != split:
                raise ValueError(f"Cross-split overlap in {key}")
        by_class[split][row["label"]] += 1
    if seen != eligible.keys() or any(0 in counts.values() for counts in by_class.values()):
        raise ValueError("Missing eligible images or incomplete class coverage")
    return by_class


def _publish_immutable(path, content):
    """Atomically publish complete bytes without replacing an existing identity."""
    path = filesystem_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        staged = Path(directory) / "publish"
        staged.write_bytes(content)
        try:
            if os.name == "nt":
                os.rename(staged, path)  # Windows refuses to rename over an existing file.
            else:
                os.link(staged, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError(f"Existing {path.name} differs; preserve it and review the discrepancy") from None


def phase4_split(root, config, action, expected_sha256=None, regenerate=False):
    """Generate, audit, explicitly freeze, or verify the reviewed local split."""
    if action not in ("split", "audit", "freeze", "frozen"):
        raise ValueError("Unknown split action")
    root = Path(root).resolve()
    identity = {key: config[key] for key in ("dataset_id", "inventory_version", "inventory_sha256", "groups_sha256")}
    if not all(_is_sha256(value) for value in identity.values()) or type(config["seed"]) is not int:
        raise ValueError("Expected approved SHA-256 identities and an integer seed")
    identity.update(seed=config["seed"], algorithm=SPLIT_ALGORITHM, target_percent=SPLIT_PERCENT,
                    max_attempts=SPLIT_ATTEMPTS)
    version = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    base = root / "data" / "manifests" / identity["dataset_id"]
    eligible = _eligible_split_rows(base / "phase3-v1", identity)
    assigned, used_seed = _allocate_groups(eligible, identity["seed"])
    canonical = _split_csv(row | {"split_version": version, "split": assigned[row["group_id"]]}
                           for row in eligible.values())
    manifest = base / "splits" / version / "split.csv"
    if action == "split" and not regenerate:
        _publish_immutable(manifest, canonical)
    content = filesystem_path(manifest).read_bytes()
    reader = csv.DictReader(StringIO(content.decode("utf-8"), newline=""))
    if tuple(reader.fieldnames or ()) != SPLIT_COLUMNS:
        raise ValueError("Split manifest schema differs")
    per_class = _audit_split(reader, eligible, version)
    if content != canonical:
        raise ValueError("Manifest differs from canonical regeneration")
    digest = hashlib.sha256(content).hexdigest()
    counts = {split: sum(classes.values()) for split, classes in per_class.items()}
    fractions = {split: count / len(eligible) for split, count in counts.items()}
    report = dict(identity, split_version=version, sha256=digest, selected_seed=used_seed,
                  counts=counts, per_class=per_class, actual_ratio=fractions,
                  deviation={split: fractions[split] - percent / 100 for split, percent in SPLIT_PERCENT.items()},
                  eligible_count=len(eligible), cross_split_overlap=dict.fromkeys(
                      ("image_id", "group_id", "file_sha256", "pixel_sha256"), 0))
    report_bytes = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
    evidence = root / "artifacts" / "data" / "phase4" / version
    audit, marker = (filesystem_path(evidence / name) for name in ("audit.json", "frozen.json"))
    if action == "audit":
        _publish_immutable(audit, report_bytes)
    if action in ("freeze", "frozen"):
        if not audit.is_file() or audit.read_bytes() != report_bytes:
            raise ValueError("Matching audit evidence required before freeze or downstream use")
        if action == "freeze":
            if digest != expected_sha256:
                raise ValueError("Reviewed manifest SHA-256 required for freeze")
            _publish_immutable(marker, report_bytes)
        if not marker.is_file() or marker.read_bytes() != report_bytes:
            raise ValueError("Matching frozen identity required before downstream use")
    return report | {"manifest": str(manifest), "raw_root": str(root / "data" / "raw" / identity["dataset_id"] / "extracted")}


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
