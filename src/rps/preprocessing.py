"""Shared deterministic input conversion and the Phase 4 development pipeline."""

import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

from .dataset import CLASSES, filesystem_path


def _image_geometry(contract):
    policies = {
        "schema_version": 1, "resize": "bilinear", "color_mode": "RGB",
        "exif_transpose": True, "scale_divisor": 255, "padding": "center_constant",
    }
    for key, value in policies.items():
        if contract.get(key) != value:
            raise ValueError(f"Unsupported preprocessing policy for {key}: expected {value!r}")
    height, width, pad = (contract[key] for key in ("height", "width", "pad_value"))
    if any(type(size) is not int or size <= 0 for size in (height, width)):
        raise ValueError("height and width must be positive integers")
    if type(pad) is not int or not 0 <= pad <= 255:
        raise ValueError("pad_value must be an integer from 0 to 255")
    return height, width, pad


def _read_image(path, height, width, pad):
    image_input = filesystem_path(path) if isinstance(path, (str, os.PathLike)) else path
    with Image.open(image_input, formats=("PNG", "JPEG")) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
    ratio = min(width / rgb.width, height / rgb.height)
    size = (max(1, round(rgb.width * ratio)), max(1, round(rgb.height * ratio)))
    # Pillow bilinear resampling widens the filter footprint when downsampling.
    resized = rgb.resize(size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (width, height), (pad,) * 3)
    canvas.paste(resized, ((width - size[0]) // 2, (height - size[1]) // 2))
    return np.asarray(canvas, dtype=np.float32) / np.float32(255)


def preprocess(path, contract):
    """Decode original file bytes and return one float32 [height,width,3] tensor."""
    return tf.convert_to_tensor(_read_image(path, *_image_geometry(contract)))


def _validate_augmentation(settings):
    maxima = {"rotation_degrees": 180, "translation_fraction": 1,
              "zoom_fraction": 1, "brightness_delta": 1}
    unknown = settings.keys() - (maxima.keys() | {"horizontal_flip"})
    if unknown:
        raise ValueError(f"Unknown augmentation settings: {sorted(unknown)}")
    for key, maximum in maxima.items():
        value = settings.get(key, 0)
        if type(value) not in (int, float) or not 0 <= value <= maximum:
            raise ValueError(f"{key} must be in [0, {maximum}]")
    zoom = settings.get("zoom_fraction", 0)
    if zoom == 1 or type(settings.get("horizontal_flip", False)) is not bool:
        raise ValueError("zoom_fraction must be < 1 and horizontal_flip must be boolean")


def validate_contract(contract):
    _image_geometry(contract)
    mapping = contract.get("class_mapping", {})
    if (not isinstance(mapping, dict) or set(mapping) != set(CLASSES)
            or any(type(i) is not int for i in mapping.values())
            or set(mapping.values()) != {0, 1, 2}):
        raise ValueError("class_mapping must map rock, paper and scissors to distinct IDs 0..2")
    _validate_augmentation(contract["augmentation"])


def augment_image(image, settings, seed):
    """Resample once with bilinear interpolation and nearest-edge extension.

    An optional flip and clipped brightness adjustment follow the geometry.
    A scalar seed or [seed, step] repeats a preview.
    """
    _validate_augmentation(settings)
    zoom = settings.get("zoom_fraction", 0)
    seed = tf.convert_to_tensor(seed, dtype=tf.int32)
    if seed.shape.rank == 0:
        seed = tf.stack([seed, 0])
    seed = tf.ensure_shape(seed, [2])
    draws = tf.random.stateless_uniform([6], seed, minval=-1., maxval=1.)
    rotation = settings.get("rotation_degrees", 0)
    translation = settings.get("translation_fraction", 0)
    if rotation or translation or zoom:
        height, width = tf.unstack(tf.cast(tf.shape(image)[:2], tf.float32))
        cx, cy = (width - 1) / 2, (height - 1) / 2
        angle = draws[0] * (rotation * math.pi / 180)
        scale = 1 + draws[1] * zoom
        cos, sin = tf.cos(angle) / scale, tf.sin(angle) / scale
        dx, dy = draws[2] * translation * width, draws[3] * translation * height
        # ProjectiveTransform maps output coordinates back to the input.
        tx = cx - cos * (cx + dx) - sin * (cy + dy)
        ty = cy + sin * (cx + dx) - cos * (cy + dy)
        transform = tf.stack([cos, sin, tx, -sin, cos, ty, 0., 0.])
        image = tf.raw_ops.ImageProjectiveTransformV3(
            images=image[None], transforms=transform[None], output_shape=tf.shape(image)[:2],
            interpolation="BILINEAR", fill_mode="NEAREST", fill_value=0.,
        )[0]
    if settings.get("horizontal_flip", False):
        image = tf.cond(draws[4] < 0, lambda: tf.image.flip_left_right(image), lambda: image)
    if settings.get("brightness_delta", 0):
        image = tf.clip_by_value(image + draws[5] * settings["brightness_delta"], 0., 1.)
    return image


def make_dataset(rows, root, contract, split, batch_size=32, seed=42, augment=False):
    """Build batches from filtered development CSV dictionaries in manifest order."""
    if split not in {"train", "validation"} or (augment and split != "train"):
        raise ValueError("Choose train or validation; augmentation requires train")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    validate_contract(contract)
    mapping = contract["class_mapping"]
    height, width, pad = _image_geometry(contract)
    root = Path(root).resolve()
    paths, labels = [], []
    for row in rows:
        if row["split"] not in {"train", "validation"}:
            raise ValueError("Development input requires a filtered train/validation manifest")
        if row["split"] != split:
            continue
        label = row["label"]
        if label not in mapping:
            raise ValueError(f"Unknown label {label!r} for {row['image_id']}")
        if "class_id" in row and str(row["class_id"]) != str(mapping[label]):
            raise ValueError(f"Conflicting class_id for {row['image_id']}")
        relative = Path(row["relative_path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError(f"Expected an image path within the development root: {relative}")
        paths.append(str(path))
        labels.append(mapping[label])
    if not paths:
        raise ValueError(f"No {split} rows supplied")
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if split == "train":
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    # These callbacks build TensorFlow ops directly and need no AutoGraph AST conversion.
    @tf.autograph.experimental.do_not_convert
    def load(path, label):
        def read(path_bytes):
            return _read_image(os.fsdecode(np.asarray(path_bytes).item()), height, width, pad)
        image = tf.numpy_function(read, [path], tf.float32)
        return tf.ensure_shape(image, [height, width, 3]), label

    dataset = dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    if augment:
        @tf.autograph.experimental.do_not_convert
        def transform(step, pair):
            image, label = pair
            image_seed = tf.stack([tf.cast(seed, tf.int32), tf.cast(step, tf.int32)])
            return augment_image(image, contract["augmentation"], image_seed), label
        dataset = dataset.enumerate().map(
            transform, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True,
        )
    return dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
