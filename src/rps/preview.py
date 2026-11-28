"""Render repeatable Phase 4 previews of frozen training examples."""

import csv
import hashlib
import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from matplotlib.figure import Figure
from PIL import Image, ImageOps

from .dataset import filesystem_path, phase4_split, sha256_file
from .preprocessing import augment_image, preprocess


def _examples(manifest, mapping, seed):
    with filesystem_path(manifest).open(encoding="utf-8", newline="") as stream:
        training = [row for row in csv.DictReader(stream) if row["split"] == "train"]
    rng = random.Random(seed)
    selected = []
    for label in sorted(mapping, key=mapping.get):
        members = sorted(
            (row for row in training if row["label"] == label),
            key=lambda row: row["image_id"],
        )
        if not members:
            raise ValueError(f"Training split has no examples for {label}")
        selected.extend(rng.sample(members, min(2, len(members))))
    return selected


def _panel(axis, pixels, title):
    axis.imshow(pixels, interpolation="nearest")
    axis.set_title(title, fontsize=9)
    axis.set_axis_off()


def _isolated(image, key, value, seed, step):
    # For the optional flip panel, show the applied branch of the 50% draw.
    for attempt in range(64):
        pair = [seed, step + attempt]
        output = augment_image(image, {key: value}, pair).numpy()
        if key != "horizontal_flip" or np.array_equal(output, image.numpy()[:, ::-1, :]):
            return output, pair
    raise ValueError("Could not reproduce the horizontal flip for this preview seed")


def run_preview(root, config_path):
    """Save training grids and their review record, then return the record."""
    root = Path(root).resolve()
    preprocessing_sha256 = sha256_file(root / "src" / "rps" / "preprocessing.py")
    config = json.loads((root / config_path).read_text(encoding="utf-8"))
    split = phase4_split(root, config, "frozen")
    contract = config["contract"]
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    seed = config["seed"]
    examples = _examples(split["manifest"], contract["class_mapping"], seed)
    alternative = {**contract, "height": 128, "width": 128}
    settings = contract["augmentation"]
    transforms = [
        ("rotation_degrees", "Rotation"), ("translation_fraction", "Translation"),
        ("zoom_fraction", "Zoom"), ("brightness_delta", "Brightness"),
        ("horizontal_flip", "Flip trial"),
    ]
    size_grid = Figure(figsize=(9, 2.25 * len(examples)), layout="constrained")
    aug_grid = Figure(figsize=(16, 2.25 * len(examples)), layout="constrained")
    sizes = size_grid.subplots(len(examples), 3, squeeze=False)
    aug = aug_grid.subplots(len(examples), len(transforms) + 1, squeeze=False)
    records = []
    raw_root = Path(split["raw_root"]).resolve()
    for i, row in enumerate(examples):
        path = (raw_root / row["relative_path"]).resolve()
        if not path.is_relative_to(raw_root) or sha256_file(path) != row["file_sha256"]:
            raise ValueError(f"Training source verification failed for {row['image_id']}")
        with Image.open(filesystem_path(path)) as source:
            original = ImageOps.exif_transpose(source).convert("RGB")
        image = preprocess(path, contract)
        label = f"{row['label']} · {row['image_id'][:8]}"
        _panel(sizes[i, 0], original, label + " · source")
        _panel(sizes[i, 1], image, f"Candidate {contract['width']} × {contract['height']}")
        _panel(sizes[i, 2], preprocess(path, alternative), "Comparison 128 × 128")
        _panel(aug[i, 0], image, label + " · baseline")
        seeds = {}
        for j, (key, title) in enumerate(transforms, start=1):
            value = True if key == "horizontal_flip" else settings[key]
            pixels, pair = _isolated(image, key, value, seed, (i * len(transforms) + j) * 64)
            _panel(aug[i, j], pixels, title + f" ({value})")
            seeds[key] = pair
        records.append({
            "image_id": row["image_id"], "label": row["label"],
            "file_sha256": row["file_sha256"], "transform_seeds": seeds,
            "output_shape": image.shape.as_list(), "output_dtype": image.dtype.name,
            "output_range": [float(np.min(image)), float(np.max(image))],
        })

    directory = root / "artifacts" / "data" / "phase4" / split["split_version"]
    output = filesystem_path(directory)
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    with TemporaryDirectory(dir=output) as temp:
        stage = Path(temp)
        for name, figure in [("preprocessing", size_grid), ("augmentation", aug_grid)]:
            path = stage / f"{name}.png"
            figure.savefig(path, dpi=110)
            checksum = sha256_file(path)
            filename = f"{name}-{checksum}.png"
            files[filename] = checksum
            path.replace(output / filename)
        summary = {
            "dataset_id": config["dataset_id"], "split_version": split["split_version"],
            "split_sha256": split["sha256"], "contract_sha256": digest, "contract": contract,
            "preprocessing_sha256": preprocessing_sha256,
            "seed": seed, "comparison_size": [128, 128], "preview_files": files,
            "selection": "Up to two training images per class, sampled from sorted IDs.",
            "examples": records,
        }
        record = stage / "preview-summary.json"
        record.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        record.replace(output / record.name)
    print(f"Training previews: {directory}")
    print(f"Candidate contract SHA-256 for approve-contract: {digest}")
    return summary
