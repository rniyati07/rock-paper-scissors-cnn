import hashlib
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf


SMOKE_INPUT = [[1.0, 2.0]]


def test_python_and_tensorflow_can_use_local_cpu():
    assert sys.version_info[:2] == (3, 13)
    assert tf.config.list_physical_devices("CPU")
    assert tf.keras.version() == version("keras")


def test_keras_round_trip_preserves_prediction(tmp_path):
    inputs = tf.constant(SMOKE_INPUT, dtype=tf.float32)
    with tf.device("/CPU:0"):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(2,)),
            tf.keras.layers.Dense(2, activation="softmax"),
        ])
        model.layers[0].set_weights([
            np.array([[0.25, 0.5], [-0.5, 0.25]], dtype=np.float32),
            np.array([0.25, -0.25], dtype=np.float32),
        ])
        before = model(inputs, training=False).numpy()
        path = tmp_path / "phase1_smoke.keras"
        model.save(path)
        after = tf.keras.models.load_model(path)(inputs, training=False).numpy()

    assert before.shape == after.shape == (1, 2)
    assert np.isfinite(before).all() and np.isfinite(after).all()
    np.testing.assert_allclose(after, before, rtol=0, atol=1e-6)
    print(f"TP-01 maximum absolute score difference: {np.max(np.abs(after - before)):.9g}")


def test_colab_export_loads_on_windows_cpu():
    artifact_dir = os.environ.get("RPS_COLAB_SMOKE_DIR")
    if not artifact_dir:
        pytest.skip("Set RPS_COLAB_SMOKE_DIR to the downloaded smoke artifact directory")

    model_path = Path(artifact_dir) / "phase1_smoke.keras"
    reference = json.loads(
        (model_path.parent / "phase1_reference.json").read_text(encoding="utf-8")
    )
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == reference["model_sha256"]
    assert reference["python"].split(".")[:2] == ["3", "13"]
    for dependency in ("tensorflow", "keras", "numpy"):
        assert reference[dependency] == version(dependency)

    with tf.device("/CPU:0"):
        model = tf.keras.models.load_model(model_path, compile=False)
        local_scores = model(tf.constant(SMOKE_INPUT, dtype=tf.float32), training=False).numpy()

    reference_scores = np.asarray(reference["scores"], dtype=np.float32)
    assert local_scores.shape == reference_scores.shape == (1, 2)
    assert np.isfinite(local_scores).all() and np.isfinite(reference_scores).all()
    np.testing.assert_allclose(local_scores, reference_scores, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(local_scores.argmax(axis=1), reference_scores.argmax(axis=1))
    print(f"TP-01 portability maximum absolute score difference: {np.max(np.abs(local_scores - reference_scores)):.9g}")
