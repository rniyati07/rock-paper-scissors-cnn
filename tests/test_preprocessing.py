"""TP-09 through TP-11 unit checks with temporary decoder fixtures."""

import numpy as np
from PIL import Image
import pytest
import tensorflow as tf

from src.rps.preprocessing import augment_image, make_dataset, preprocess


@pytest.fixture
def contract():
    return {
        "schema_version": 1, "resize": "bilinear", "color_mode": "RGB",
        "exif_transpose": True, "scale_divisor": 255, "padding": "center_constant",
        "height": 5, "width": 4, "pad_value": 127,
        "class_mapping": {"rock": 0, "paper": 1, "scissors": 2},
        "augmentation": {"brightness_delta": 0.1},
    }


def test_grayscale_rgb_scaling_and_centered_odd_padding(tmp_path, contract):
    path = tmp_path / "gray.png"
    gray = np.array([[0, 63, 128, 255], [255, 128, 63, 0]], dtype=np.uint8)
    Image.fromarray(gray).save(path)
    image = preprocess(path, contract)
    assert image.dtype == tf.float32
    assert tuple(image.shape) == (5, 4, 3)
    expected = np.full((5, 4, 3), 127, dtype=np.float32)
    expected[1:3] = gray[:, :, None]
    np.testing.assert_array_equal(image, expected / 255)
    np.testing.assert_array_equal(image, preprocess(path, contract))


def test_bilinear_downsampling_smooths_high_frequency_pattern(tmp_path, contract):
    path = tmp_path / "stripes.png"
    stripes = np.tile(np.array([0, 255] * 16, dtype=np.uint8), (16, 1))
    Image.fromarray(stripes).save(path)
    result = preprocess(path, contract).numpy()
    assert np.all((result[1:3] > 0.45) & (result[1:3] < 0.55))
    np.testing.assert_allclose(result[[0, 3, 4]], 127 / 255)


@pytest.mark.parametrize("suffix", ["jpg", "jpeg"])
def test_jpeg_exif_rotation_before_aspect_fit(tmp_path, contract, suffix):
    pixels = np.full((16, 32, 3), [255, 0, 0], dtype=np.uint8)
    pixels[:, 16:] = [0, 0, 255]
    path = tmp_path / f"oriented.{suffix}"
    exif = Image.Exif()
    exif[274] = 6  # 90 degrees clockwise: original left edge becomes top.
    Image.fromarray(pixels).save(path, exif=exif, quality=100, subsampling=0)
    image = preprocess(path, contract).numpy()
    np.testing.assert_allclose(image[:, [0, 3]], 127 / 255)
    assert image[0, 1, 0] > 0.9
    assert image[-1, 1, 2] > 0.9


def test_rejects_unapproved_decoder_and_unknown_contract(tmp_path, contract):
    path = tmp_path / "wrong.png"
    Image.new("RGB", (3, 2)).save(path, format="GIF")
    with pytest.raises(OSError):
        preprocess(path, contract)
    with pytest.raises(ValueError, match="resize"):
        preprocess(path, {**contract, "resize": "nearest"})


@pytest.fixture
def dev_rows(tmp_path):
    rows = []
    for split in ("train", "validation"):
        for index in range(7):
            label = ("paper", "scissors", "rock")[index % 3]
            image_id = f"{split}-{index}"
            name = f"{image_id}.png"
            Image.new("RGB", (4, 5), ((index + 1) * 20, 60, 80)).save(tmp_path / name)
            rows.append(dict(image_id=image_id, relative_path=name, split=split, label=label))
    return rows


def test_order_alignment_partial_batches_and_seeded_shuffle(tmp_path, contract, dev_rows):
    def records(split):
        dataset = make_dataset(dev_rows, tmp_path, contract, split, batch_size=3, seed=19)
        aligned, sizes = [], []
        for images, labels in dataset.as_numpy_iterator():
            sizes.append(len(labels))
            ids = np.rint(images[:, 0, 0, 0] * 255 / 20).astype(int) - 1
            aligned.extend(zip(ids.tolist(), labels.tolist()))
        assert sizes == [3, 3, 1]
        return aligned

    ordered = [(i, [1, 2, 0][i % 3]) for i in range(7)]
    assert records("validation") == records("validation") == ordered
    train = records("train")
    assert sorted(train) == ordered
    assert train == records("train")
    assert train != ordered


def test_rejects_incompatible_labels_mapping_and_reserved_rows(tmp_path, contract, dev_rows):
    row = dev_rows[0]
    for invalid, message in [
        ({**row, "label": "unknown"}, "Unknown label"),
        ({**row, "class_id": 9}, "Conflicting class_id"),
        ({**row, "split": "test"}, "filtered"),
        ({**row, "relative_path": "../outside.png"}, "within"),
    ]:
        with pytest.raises(ValueError, match=message):
            make_dataset([invalid], tmp_path, contract, "train")
    with pytest.raises(ValueError, match="class_mapping"):
        make_dataset(dev_rows, tmp_path, {**contract, "class_mapping": {"rock": 0}}, "train")
    with pytest.raises(ValueError, match="Choose train or validation"):
        make_dataset(dev_rows, tmp_path, contract, "test")


def test_augmentation_confined_to_training(tmp_path, contract, dev_rows):
    def first(split, augment):
        ds = make_dataset(dev_rows, tmp_path, contract, split, seed=7, augment=augment)
        return next(ds.as_numpy_iterator())

    original, labels = first("train", False)
    modified, modified_labels = first("train", True)
    np.testing.assert_array_equal(labels, modified_labels)
    assert np.max(np.abs(original - modified)) > 1e-4
    with pytest.raises(ValueError, match="augmentation requires train"):
        first("validation", True)
    clean, _ = first("validation", False)
    for image, row in zip(clean, [r for r in dev_rows if r["split"] == "validation"]):
        np.testing.assert_array_equal(image, preprocess(tmp_path / row["relative_path"], contract))


def test_preview_seed_repeats_and_inactive_transforms_are_identity():
    image = tf.reshape(tf.linspace(0., 1., 60), (5, 4, 3))
    settings = {"rotation_degrees": 6, "translation_fraction": .05, "zoom_fraction": .05,
                "horizontal_flip": True, "brightness_delta": .05}
    result = augment_image(image, settings, [42, 7]).numpy()
    np.testing.assert_array_equal(result, augment_image(image, settings, [42, 7]))
    np.testing.assert_array_equal(image, augment_image(image, {}, 42))
    assert result.shape == image.shape and result.min() >= 0 and result.max() <= 1
    assert np.max(np.abs(result - image.numpy())) > 1e-4
