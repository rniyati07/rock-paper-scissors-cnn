import csv
import json

import numpy as np
from PIL import Image
import pytest

from src.rps import dataset
from src.rps.eda import build_group_evidence


def test_seeded_gallery_ignores_inventory_order():
    rows = [dataset.ImageRow("release", str(i), f"rock/{i}.png", 0, label="rock") for i in range(8)]
    first = dataset.sample_class_rows(rows, 3, 42)
    assert first == dataset.sample_class_rows(list(reversed(rows)), 3, 42)
    assert len({row.image_id for row in first["rock"]}) == 3


def test_dhash_and_pair_distances_include_cross_label_candidates():
    gradient = np.tile(np.arange(9, dtype=np.uint8), (8, 1))
    assert dataset.difference_hash(Image.fromarray(gradient)) == 2**64 - 1
    assert dataset.difference_hash(Image.fromarray(gradient[:, ::-1])) == 0
    counts, pairs = dataset.screen_hashes({"c": 7, "a": 0, "b": 1}, 2)
    assert counts[1:4] == [1, 1, 1]
    assert pairs == [("a", "b", 1), ("b", "c", 2)]


def test_image_stats_use_unique_border_pixels_and_rgb(tmp_path):
    pixels = np.zeros((4, 6, 3), dtype=np.uint8)
    pixels[:, :, 1] = 80
    pixels[1:-1, 1:-1, 1] = 0
    path = tmp_path / "image.png"
    Image.fromarray(pixels).save(path)
    result = dataset.measure_image(path, 1)
    assert result["rgb_histograms"][1, 80] == 16
    assert result["rgb_histograms"][1, 0] == 8
    assert result["border_mean_rgb"] == [0, 80, 0]
    assert result["border_green_share"] == 1
    assert result["brightness"] == pytest.approx(80 * 0.7152 * 16 / 24)


def test_relationships_through_excluded_copy_and_quarantine_are_transitive():
    rows = [
        dataset.ImageRow("release", "a", "rock/a.png", 0, source_label="rock", disposition="eligible"),
        dataset.ImageRow("release", "copy", "nested/a.png", 0, representative_id="a", disposition="excluded"),
        dataset.ImageRow("release", "b", "paper/b.png", 0, source_label="paper", disposition="eligible"),
        dataset.ImageRow("release", "c", "rock/c.png", 0, source_label="rock", disposition="eligible"),
    ]
    groups = dataset.connected_groups(rows, [("copy", "b")])
    assert groups["a"] == groups["copy"] == groups["b"]
    assert groups["c"] != groups["a"]
    assert groups == dataset.connected_groups(rows[::-1], [("b", "copy")])
    with pytest.raises(ValueError, match="Unknown image"):
        dataset.connected_groups(rows, [("copy", "missing")])


def test_review_records_preserve_decisions_and_require_attribution(tmp_path):
    path = tmp_path / "reviews.csv"
    cases = [{"case_id": "a-b", "distance": "2"}]
    records = dataset.sync_reviews(path, cases, {"distinct", "confirmed"})
    assert not records[0]["decision"]
    records[0].update(decision="distinct", reviewer="Reviewer", rationale="Different fingers")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    assert dataset.sync_reviews(path, cases, {"distinct", "confirmed"}) == records
    with pytest.raises(ValueError, match="changed"):
        dataset.sync_reviews(path, [{"case_id": "a-b", "distance": "3"}], {"distinct", "confirmed"})
    path.write_text(path.read_text().replace("Reviewer,", ","), encoding="utf-8")
    with pytest.raises(ValueError, match="reviewer"):
        dataset.sync_reviews(path, cases, {"distinct", "confirmed"})


def test_confirmed_component_with_corrected_conflict_is_quarantined(tmp_path):
    rows = [
        dataset.ImageRow("release", key, f"rock/{key}.png", 0, source_label="rock", label="rock",
                         representative_id=representative, disposition=state)
        for key, representative, state in [("a", "a", "eligible"), ("copy", "a", "excluded"),
                                           ("b", "b", "eligible"), ("c", "c", "eligible")]
    ]
    pairs = [
        {"left": left, "right": right, "left_label": "rock", "right_label": "rock", "decision": "confirmed"}
        for left, right in [("copy", "b"), ("b", "c")]
    ]
    correction = {"case_id": "I001", "image_id": "c", "decision": "correct_paper"}
    result = build_group_evidence(rows, [correction], pairs, tmp_path)
    assert result["by_disposition"] == {"pending": 3, "excluded": 1}
    assert result["components"] == 1
    assert not result["review_complete"]
    assert (tmp_path / result["review_snapshot"]).is_file()


def test_trusted_session_can_include_distinct_gestures(tmp_path):
    rows = [dataset.ImageRow("release", label, f"{label}/a.png", 0, source_label=label, label=label,
                             disposition="eligible") for label in ("paper", "rock")]
    result = build_group_evidence(rows, [], [], tmp_path, [("paper", "rock")])
    assert result["review_complete"]
    assert result["by_disposition"] == {"eligible": 2}
    assert result["eligible_component_sizes"] == {2: 1}


def test_conservative_policy_preserves_unreviewed_evidence_and_restores_distinct_pair(tmp_path):
    rows = [
        dataset.ImageRow("release", key, f"{label}/{key}.png", 0, source_label=label, label=label,
                         representative_id=key, disposition="eligible", file_sha256=key)
        for key, label in (("a", "rock"), ("b", "rock"), ("c", "paper"))
    ]
    images = [{"case_id": "I001", "image_id": "a", "decision": ""}]
    pairs = [
        {"case_id": "P001", "left": "a", "right": "b", "left_label": "rock", "right_label": "rock", "decision": ""},
        {"case_id": "P002", "left": "a", "right": "c", "left_label": "rock", "right_label": "paper", "decision": ""},
    ]
    policy = {"mode": "conservative", "approved_by": "Project owner"}
    pending = build_group_evidence(rows, images, pairs, tmp_path, review_policy=policy)
    assert not pending["review_complete"]
    assert pending["by_disposition"]["pending"] == 3  # Quarantine propagates through the conservative link.
    assert "inventory_file" not in pending
    pairs[1]["decision"] = "distinct"
    final = build_group_evidence(rows, images, pairs, tmp_path, review_policy=policy)
    assert final["review_complete"]
    assert final["eligible_component_sizes"] == {1: 1, 2: 1}
    snapshot = json.loads((tmp_path / final["review_snapshot"]).read_text(encoding="utf-8"))
    assert snapshot["pair_reviews"][0]["decision"] == ""
    assert snapshot["policy_links"] == [{"case_id": "P001", "left": "a", "right": "b"}]
    assert snapshot["retained_by_policy_case_ids"] == ["I001"]
    with (tmp_path / final["inventory_file"]).open(newline="", encoding="utf-8") as handle:
        inventory = {row["image_id"]: row for row in csv.DictReader(handle)}
    assert inventory["a"]["file_sha256"] == "a"
    assert inventory["a"]["group_id"] == inventory["b"]["group_id"]
    assert inventory["c"]["label"] == "paper"
