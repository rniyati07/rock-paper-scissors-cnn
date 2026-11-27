"""Reproduce Phase 3 summaries and maintain local visual review records."""

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import re
from zipfile import ZipFile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, __version__ as pillow_version

from . import dataset


DATASET_ID = "3ba1e5daf5beba5358afe6f9c2e43ff9040789603e0405a81a50353a3ae2d508"
INVENTORY_SHA256 = "3ce3874c12e586aa4d29677be44e6a16155cbde82713ed980c8a0ed1acd1ff50"
SETTINGS = {
    "protocol": "phase3-v1",
    "seed": 42,
    "examples_per_class": 12,
    "border_width": 10,
    "review_radius": 2,
    "cross_label_radius": 4,
    "calibration_radius": 5,
    "pairs_per_sheet": 8,
}
COLORS = {"rock": "#4c78a8", "paper": "#d57b29", "scissors": "#458a64"}
IMAGE_DECISIONS = {"accept", "flag", "exclude", *(f"correct_{label}" for label in dataset.CLASSES)}
PAIR_DECISIONS = {"distinct", "confirmed", "co_group", "flag"}


def save_figure(fig, path):
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_json(path, values):
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def plot_inventory(rows, output):
    fig = plt.figure(figsize=(10, 5), layout="constrained")
    subplots = fig.add_gridspec(3, 2)
    counts = fig.add_subplot(subplots[:, 0])
    positions = np.arange(len(dataset.CLASSES))
    for index, (name, state) in enumerate((("Raw", None), ("Eligible", "eligible"), ("Excluded", "excluded"))):
        totals = [sum(r.source_label == label and (state is None or r.disposition == state) for r in rows)
                  for label in dataset.CLASSES]
        counts.bar(positions + (index - 1) * 0.25, totals, 0.25, label=name)
    counts.set(xticks=positions, xticklabels=dataset.CLASSES, ylabel="Image entries", title="Phase 2 dispositions")
    counts.legend()
    values = [("Width (px)", "width"), ("Height (px)", "height"), ("Width / height", None)]
    decoded = [row for row in rows if row.decode_status == "ok"]
    for index, (label, field) in enumerate(values):
        axis = fig.add_subplot(subplots[index, 1])
        frequencies = Counter(getattr(r, field) if field else r.width / r.height for r in decoded)
        axis.bar([str(value) for value in sorted(frequencies)], [frequencies[v] for v in sorted(frequencies)])
        axis.set(ylabel="Entries", xlabel=label)
    save_figure(fig, output / "inventory.png")


def plot_distributions(eligible, metrics, output):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout="constrained")
    for label in dataset.CLASSES:
        values = [metrics[row.image_id] for row in eligible if row.label == label]
        rgb = np.sum([item["rgb_histograms"] for item in values], axis=0)
        luminance = np.sum([item["luminance_histogram"] for item in values], axis=0)
        for axis, histogram in zip(axes[0], [*rgb, luminance]):
            axis.plot(histogram / histogram.sum(), label=label, color=COLORS[label])
        axes[1, 0].hist([item["brightness"] for item in values], bins=np.arange(0, 257, 4),
                        density=True, histtype="step", color=COLORS[label], label=label)
    for axis, title in zip(axes[0], ("Red", "Green", "Blue", "Brightness proxy")):
        axis.set(title=title, xlabel="Level (0–255)", ylabel="Share of class pixels")
    axes[0, 0].legend()
    axes[1, 0].set(title="Mean brightness per image", xlabel="Mean level (0–255)", ylabel="Probability density")
    quantiles = [[metrics[row.image_id]["luminance_quantiles"] for row in eligible if row.label == label]
                 for label in dataset.CLASSES]
    for index, axis in enumerate(axes[1, 1:]):
        axis.boxplot([np.asarray(values)[:, index] for values in quantiles], tick_labels=dataset.CLASSES,
                     showfliers=False)
        axis.set(title=f"Per-image P{(5, 50, 95)[index]}", ylabel="Brightness proxy (0–255)")
    save_figure(fig, output / "distributions.png")

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), layout="constrained")
    for index, (axis, title) in enumerate(zip(axes, ("Red", "Green", "Blue", "Green dominance"))):
        values = [
            [metrics[row.image_id]["border_mean_rgb"][index] if index < 3
             else metrics[row.image_id]["border_green_share"] for row in eligible if row.label == label]
            for label in dataset.CLASSES
        ]
        axis.boxplot(values, tick_labels=dataset.CLASSES, showfliers=False)
        axis.set(title=title, ylabel="Mean border level (0–255)" if index < 3 else "Share of border pixels")
    save_figure(fig, output / "border-color.png")


def image_panel(axis, root, row, title, *, border_only=False):
    with Image.open(root / row.relative_path) as image:
        pixels = np.array(image.convert("RGB"))
    if border_only:
        mask = dataset.border_mask(*pixels.shape[:2], SETTINGS["border_width"])
        pixels[~mask] = 128
    axis.imshow(pixels)
    axis.set_title(title, fontsize=8)
    axis.set_axis_off()


def image_sheets(root, directory, cases, rows_by_id, name, border_only=False):
    for offset in range(0, len(cases), 12):
        page = cases[offset:offset + 12]
        columns = 2 if border_only else 4
        fig, axes = plt.subplots((len(page) + columns - 1) // columns, 4, squeeze=False,
                                 figsize=(12, 2.6 * ((len(page) + columns - 1) // columns)), layout="constrained")
        for axis in axes.flat:
            axis.set_axis_off()
        for index, case in enumerate(page):
            row = rows_by_id[case["image_id"]]
            title = f"{case['case_id']} · {row.source_label}\n{case['selection']}"
            image_panel(axes.flat[index * (2 if border_only else 1)], root, row, title)
            if border_only:
                image_panel(axes.flat[2 * index + 1], root, row, "Border only", border_only=True)
        save_figure(fig, directory / f"{name}-{offset // 12 + 1:02}.png")


def pair_sheets(root, directory, cases, rows_by_id, name):
    page_size = SETTINGS["pairs_per_sheet"]
    for offset in range(0, len(cases), page_size):
        page = cases[offset:offset + page_size]
        fig, axes = plt.subplots((len(page) + 1) // 2, 4, figsize=(12, 2.6 * ((len(page) + 1) // 2)),
                                 squeeze=False, layout="constrained")
        for axis in axes.flat:
            axis.set_axis_off()
        for index, case in enumerate(page):
            first, second = (rows_by_id[case[key]] for key in ("left", "right"))
            title = f"{case['case_id']} · distance {case['distance']}"
            image_panel(axes.flat[2 * index], root, first, f"{title}\n{first.source_label}")
            image_panel(axes.flat[2 * index + 1], root, second, f"{title}\n{second.source_label}")
        save_figure(fig, directory / f"{name}-{offset // page_size + 1:02}.png")


def select_image_cases(eligible, samples, metrics, pairs, rows_by_id):
    selections = {}

    def add(row, reason):
        selections.setdefault(row.image_id, []).append(reason)

    for label in dataset.CLASSES:
        for row in samples[label]:
            add(row, "seeded gallery")
        members = [row for row in eligible if row.label == label]
        brightness = sorted(members, key=lambda row: (metrics[row.image_id]["brightness"], row.image_id))
        edges = sorted(members, key=lambda row: (metrics[row.image_id]["edge_strength"], row.image_id))
        for row in brightness[:3]:
            add(row, "dark tail")
        for row in brightness[-3:]:
            add(row, "bright tail")
        for row in edges[:3]:
            add(row, "low edge strength")
    for case in pairs:
        if case["selection"] == "cross-label priority":
            for key in ("left", "right"):
                add(rows_by_id[case[key]], "cross-label pair")
    return [
        {"case_id": f"I{index:03}", "image_id": image_id, "source_label": rows_by_id[image_id].source_label,
         "selection": " / ".join(reasons)}
        for index, (image_id, reasons) in enumerate(selections.items(), 1)
    ]


def select_pair_cases(pairs, rows_by_id):
    review_cases = []
    examples = []
    for left, right, distance in pairs:
        cross_label = rows_by_id[left].label != rows_by_id[right].label
        evidence = {"left": left, "right": right, "distance": str(distance),
                    "left_label": rows_by_id[left].label, "right_label": rows_by_id[right].label}
        if distance <= SETTINGS["review_radius"] or (cross_label and distance <= SETTINGS["cross_label_radius"]):
            selection = "cross-label priority" if cross_label else "review radius"
            review_cases.append({"case_id": f"P{len(review_cases) + 1:03}", **evidence, "selection": selection})
        if distance <= SETTINGS["calibration_radius"]:
            examples.append((distance, cross_label, evidence))
    example_counts = Counter()
    calibration_cases = []
    for distance, cross_label, evidence in sorted(examples, key=lambda item: (item[0], not item[1])):
        if example_counts[distance] < 2:
            example_counts[distance] += 1
            calibration_cases.append({"case_id": f"D{len(calibration_cases) + 1:02}", **evidence})
    return review_cases, calibration_cases


def inspect_sessions(raw, rows, metrics, directory):
    with ZipFile(raw.parent / "archive.zip") as archive:
        dates = Counter(member.date_time for member in archive.infolist() if not member.is_dir())
    keys = Counter(key for values in metrics.values() for key in values["metadata_keys"])
    exif = Counter(tag for values in metrics.values() for tag in values["exif_tags"])
    names = Counter("16-character alphanumeric" if re.fullmatch(r"[A-Za-z0-9]{16}", Path(row.relative_path).stem)
                    else "other" for row in rows)
    evidence = {
        "filename_patterns": dict(names),
        "png_metadata_key_image_counts": dict(keys),
        "exif_tag_image_counts": dict(exif),
        "zip_entry_timestamp_frequency": {"-".join(map(str, date)): count for date, count in sorted(dates.items())},
        "trusted_session_links": [],
        "disposition": "No corroborated capture-session identifier. ZIP timestamps describe archive members and "
                       "filenames have no verified session semantics. Similarity links may reduce, but cannot "
                       "eliminate, capture-session leakage risk.",
    }
    write_json(directory / "session-evidence.json", evidence)
    return evidence


def build_group_evidence(rows, image_reviews, pair_reviews, directory, session_links=(), *, review_policy=None):
    near_links = []
    confirmed_links = []
    policy_links = []
    session_links = list(session_links)
    disputed = set()
    image_by_id = {case["image_id"]: case for case in image_reviews}
    decisions = {image_id: case["decision"] for image_id, case in image_by_id.items()}
    labels = {
        row.image_id: decisions[row.image_id].removeprefix("correct_")
        if decisions.get(row.image_id, "").startswith("correct_") else row.label
        for row in rows
    }
    for case in pair_reviews:
        left, right = case["left"], case["right"]
        if case["decision"] in {"confirmed", "co_group"}:
            near_links.append((left, right))
        if case["decision"] == "confirmed":
            confirmed_links.append((left, right))
        if (review_policy and not case["decision"] and case["left_label"] == case["right_label"]
                and labels[left] == labels[right]):
            policy_links.append({"case_id": case["case_id"], "left": left, "right": right})
        if case["left_label"] != case["right_label"] and case["decision"] != "distinct":
            resolved = case["decision"] in {"confirmed", "co_group"} and (
                "exclude" in (decisions.get(left), decisions.get(right)) or labels[left] == labels[right]
            )
            if not resolved:
                disputed.update((left, right))
    # Trusted sessions may contain different gestures. Effective labels must agree within duplicate components.
    duplicate_groups = dataset.connected_groups(rows, confirmed_links)
    duplicate_labels = {}
    for row in rows:
        if row.disposition == "eligible" and decisions.get(row.image_id) != "exclude":
            duplicate_labels.setdefault(duplicate_groups[row.image_id], set()).add(labels[row.image_id])
    disputed.update(row.image_id for row in rows if len(duplicate_labels.get(duplicate_groups[row.image_id], ())) > 1)
    links = [*session_links, *near_links, *((case["left"], case["right"]) for case in policy_links)]
    groups = dataset.connected_groups(rows, links)
    pending_groups = {groups[image_id] for image_id in disputed}
    pending_groups.update(groups[case["image_id"]] for case in image_reviews if case["decision"] == "flag")
    records = []
    for row in sorted(rows, key=lambda item: item.image_id):
        review = image_by_id.get(row.image_id, {})
        decision = review.get("decision", "")
        label = labels[row.image_id]
        state = row.disposition
        reason = row.reason_code
        if state == "eligible":
            if decision == "exclude":
                state, reason = "excluded", "visual_review_exclusion"
            elif groups[row.image_id] in pending_groups:
                state, reason = "pending", "visual_review_quarantine"
            elif label != row.label:
                reason = "reviewed_label_correction"
        records.append({
            "image_id": row.image_id, "source_label": row.source_label, "label": label,
            "phase2_disposition": row.disposition, "disposition": state, "reason": reason,
            "phase2_group_id": row.group_id, "representative_id": row.representative_id,
            "group_id": groups[row.image_id], "image_review_case": review.get("case_id", ""),
        })
    retained_by_policy = [
        record["image_review_case"] for record in records
        if review_policy and record["disposition"] == "eligible" and record["image_id"] in image_by_id
        and not decisions[record["image_id"]]
    ]
    snapshot = {
        "inventory_sha256": INVENTORY_SHA256, "settings": SETTINGS, "image_reviews": image_reviews,
        "pair_reviews": pair_reviews, "near_links": near_links, "session_links": session_links,
        "review_policy": review_policy, "policy_links": policy_links,
        "retained_by_policy_case_ids": retained_by_policy,
    }
    version = hashlib.sha256(json.dumps([snapshot, records], sort_keys=True).encode()).hexdigest()
    output = directory / f"groups-{version}.csv"
    dataset.write_csv(output, records, list(records[0]))
    snapshot_path = directory / f"reviews-{version}.json"
    write_json(snapshot_path, snapshot)
    eligible_groups = Counter(record["group_id"] for record in records if record["disposition"] == "eligible")
    policy_case_ids = {case["case_id"] for case in policy_links} | set(retained_by_policy)
    review_complete = (
        all((case["decision"] and case["decision"] != "flag") or case.get("case_id") in policy_case_ids
            for case in [*image_reviews, *pair_reviews])
        and all(record["disposition"] != "pending" for record in records)
        and all(record["label"] in dataset.CLASSES for record in records if record["disposition"] == "eligible")
    )
    inventory_identity = {}
    if review_complete:
        inventory = directory / f"inventory-{version}.csv"
        rows_by_id = {row.image_id: row for row in rows}
        inventory_rows = [
            asdict(rows_by_id[record["image_id"]]) | {
                "label": record["label"], "disposition": record["disposition"],
                "reason_code": record["reason"], "group_id": record["group_id"],
                "review_ref": record["image_review_case"] or rows_by_id[record["image_id"]].review_ref,
                "phase2_disposition": record["phase2_disposition"], "phase2_group_id": record["phase2_group_id"],
                "inventory_version": version,
            }
            for record in records
        ]
        dataset.write_csv(inventory, inventory_rows, list(inventory_rows[0]))
        inventory_identity = {"inventory_file": inventory.name, "inventory_sha256": dataset.sha256_file(inventory)}
    return {
        **inventory_identity,
        "file": output.name, "sha256": dataset.sha256_file(output), "version": version,
        "review_snapshot": snapshot_path.name, "review_snapshot_sha256": dataset.sha256_file(snapshot_path),
        "rows": len(records), "components": len(set(groups.values())),
        "accepted_non_exact_links": len(links),
        "policy_co_group_links": len(policy_links),
        "retained_by_policy": len(retained_by_policy),
        "exact_copy_links": sum(bool(row.representative_id and row.representative_id != row.image_id) for row in rows),
        "by_disposition": dict(Counter(record["disposition"] for record in records)),
        "by_disposition_and_class": {
            state: dict(Counter(record["label"] for record in records if record["disposition"] == state))
            for state in ("eligible", "excluded", "pending")
        },
        "eligible_component_sizes": dict(Counter(eligible_groups.values())),
        "review_complete": review_complete,
    }


def run_eda(workspace):
    raw, rows = dataset.load_validated_inventory(workspace, DATASET_ID, INVENTORY_SHA256)
    directory = workspace / "data" / "manifests" / DATASET_ID / SETTINGS["protocol"]
    output = workspace / "artifacts" / "data" / DATASET_ID / SETTINGS["protocol"]
    directory.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    by_id = {row.image_id: row for row in rows}
    eligible = sorted((row for row in rows if row.disposition == "eligible"), key=lambda row: row.image_id)
    metrics = {row.image_id: dataset.measure_image(raw / row.relative_path, SETTINGS["border_width"], row.file_sha256)
               for row in eligible}
    counts, pairs = dataset.screen_hashes({key: values["dhash"] for key, values in metrics.items()},
                                         SETTINGS["calibration_radius"])
    pair_cases, calibration = select_pair_cases(pairs, by_id)
    queued = {(case["left"], case["right"]) for case in pair_cases}
    boundary_cases = [
        case for case in calibration if (case["left"], case["right"]) not in queued
        and case["left_label"] == case["right_label"]
    ]
    samples = dataset.sample_class_rows(eligible, SETTINGS["examples_per_class"], SETTINGS["seed"])
    image_cases = select_image_cases(eligible, samples, metrics, pair_cases, by_id)
    session = inspect_sessions(raw, rows, metrics, directory)
    image_reviews = dataset.sync_reviews(directory / "image-review.csv", image_cases, IMAGE_DECISIONS)
    pair_reviews = dataset.sync_reviews(directory / "pair-review.csv", pair_cases, PAIR_DECISIONS)
    boundary_reviews = dataset.sync_reviews(directory / "calibration-review.csv", boundary_cases, PAIR_DECISIONS)
    policy_path = directory / "review-policy.json"
    review_policy = None
    if policy_path.exists():
        review_policy = json.loads(policy_path.read_text(encoding="utf-8"))
        if (review_policy["mode"] != "conservative" or review_policy["inventory_sha256"] != INVENTORY_SHA256
                or not review_policy["approved_by"] or not review_policy["approval_ref"]):
            raise ValueError("Review policy does not identify an approved conservative waiver for this inventory")
    groups = build_group_evidence(rows, image_reviews, [*pair_reviews, *boundary_reviews], directory,
                                  session["trusted_session_links"], review_policy=review_policy)

    plot_inventory(rows, output)
    plot_distributions(eligible, metrics, output)
    for label, selected in samples.items():
        gallery = [case for case in image_cases if case["image_id"] in {row.image_id for row in selected}]
        image_sheets(raw, output, gallery, by_id, f"representative-{label}")
        image_sheets(raw, output, gallery[:3], by_id, f"border-{label}", border_only=True)
    image_sheets(raw, output, image_cases, by_id, "image-review")
    pair_sheets(raw, output, pair_cases, by_id, "pair-review")
    pair_sheets(raw, output, calibration, by_id, "distance-examples")
    fig, (axis, low) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axis.bar(range(65), counts)
    low.bar(range(6), counts[:6])
    low.axvline(SETTINGS["review_radius"] + 0.5, color="black", linestyle="--", label="Review cutoff")
    low.legend()
    for panel, title in ((axis, "All representative pairs"), (low, "Low-distance detail")):
        panel.set(title=title, xlabel="dHash Hamming distance (bits)", ylabel="Pairs")
    save_figure(fig, output / "distances.png")
    dataset.write_csv(directory / "distance-examples.csv", calibration, list(calibration[0]))
    columns = ("image_id", "label", "brightness", "p05", "p50", "p95", "border_r", "border_g", "border_b",
               "border_green_share", "edge_strength", "dhash")
    dataset.write_csv(directory / "image-metrics.csv", (
        dict(zip(columns, (row.image_id, row.label, m["brightness"], *m["luminance_quantiles"],
                           *m["border_mean_rgb"], m["border_green_share"], m["edge_strength"], f"{m['dhash']:016x}")))
        for row in eligible for m in [metrics[row.image_id]]
    ), columns)

    summary = {
        "dataset_id": DATASET_ID, "inventory_sha256": INVENTORY_SHA256, "settings": SETTINGS,
        "source_sha256": {
            name: dataset.sha256_file(workspace / name)
            for name in ("src/rps/dataset.py", "src/rps/eda.py", "notebooks/01_eda.ipynb")
        },
        "runtime": {"python": platform.python_version(), "pillow": pillow_version, "numpy": np.__version__,
                    "matplotlib": matplotlib.__version__},
        "raw_counts": dict(Counter(row.source_label for row in rows)),
        "eligible_counts": dict(Counter(row.label for row in eligible)),
        "phase2_dispositions": dict(Counter(row.disposition for row in rows)),
        "decoded_properties": dict(Counter(f"{r.format} {r.mode} {r.width}x{r.height} bands={r.channels}"
                                          for r in rows if r.decode_status == "ok")),
        "brightness_range": [min(m["brightness"] for m in metrics.values()),
                             max(m["brightness"] for m in metrics.values())],
        "class_statistics": {
            label: {
                "median_brightness": float(np.median([m["brightness"] for m in selected])),
                "median_border_green_share": float(np.median([m["border_green_share"] for m in selected])),
                "median_border_rgb": np.median([m["border_mean_rgb"] for m in selected], axis=0).tolist(),
            }
            for label in dataset.CLASSES
            for selected in [[metrics[row.image_id] for row in eligible if row.label == label]]
        },
        "gallery_ids": {label: [row.image_id for row in selected] for label, selected in samples.items()},
        "pair_count": sum(counts), "distance_counts": counts,
        "review_counts": {
            "images": dict(Counter(case["decision"] or "unreviewed" for case in image_reviews)),
            "pairs": dict(Counter(case["decision"] or "unreviewed" for case in [*pair_reviews, *boundary_reviews])),
            "pair_selection": dict(Counter(case["selection"] for case in pair_cases)),
            "boundary_pairs": [case["case_id"] for case in boundary_cases],
        },
        "session": session, "grouping": groups,
    }
    write_json(output / "summary.json", summary)
    return summary, output, directory
