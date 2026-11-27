# Dataset Design

**Project** · Rock–Paper–Scissors Image Classification using CNN  
**Status** · Phase 0 draft for human review  
**Foundation** · [PRD](01_PRD.md) · [TRD](02_TRD.md)

This design defines the data contracts and review gates for Phases 2–4. The sole approved source, three-class task, local data preparation, Colab training boundary, and reserved local test split come from the PRD and TRD. Paths, algorithms, and preprocessing defaults below are proposals. Dataset-dependent settings become effective after the relevant phase review.

## 1. Dataset overview and published evidence

Use Julien de la Bruère-Terreault’s DrGFreeman Rock–Paper–Scissors Images release on Kaggle. The photographs came from the creator’s Rock–Paper–Scissors computer-vision game. The three labeled classes and modest dataset size make an end-to-end custom CNN study feasible with student resources. The green background and relatively consistent capture conditions limit the range of environments represented. [Kaggle release](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors) · [Creator’s project](https://github.com/DrGFreeman/rps-cv) · [Creator’s dataset overview](https://github.com/DrGFreeman/rps-cv-data-science#dataset)

| Property | Published specification or expectation | Local confirmation in Phase 2 |
| --- | --- | --- |
| Total and labels | 2,188 real photographs across `rock`, `paper`, and `scissors`. | Inventory the archive and reconcile counts before and after exclusions. |
| Per-class counts | Creator’s project overview reports Rock 710, Paper 726, and Scissors 752. A reproduction of the original README reports Rock 726, Paper 710, and Scissors 752. | Record measured folder counts and the discrepancy. Use validated, eligible counts for split allocation. |
| Image properties | RGB PNG, 300 × 200 pixels in width × height order. | Decode and report format, dimensions, original mode, and channel count before RGB conversion. |
| License | CC BY-SA 4.0 as published by the creator. | Retain the applicable release notice and record attribution and any changes when publishing dataset material. |

Local verification is pending. A source discrepancy requires a recorded disposition based on the acquired release. [Creator’s dataset overview](https://github.com/DrGFreeman/rps-cv-data-science#dataset) · [Reproduced original README excerpt](https://github.com/ToddKerpelman/SmallRPSDataset) · [CC BY-SA 4.0 deed](https://creativecommons.org/licenses/by-sa/4.0/)

## 2. Acquisition, versioning, and storage

1. In local VS Code, acquire the approved Kaggle archive through the browser or an authenticated Kaggle client. Record the publisher, source URL, release identifier if available, retrieval date, original filename, size, SHA-256, license, and attribution. Keep credentials outside the repository.
2. Retain the archive byte-for-byte under `data/raw/`. Extract to a version-specific directory under `data/raw/`, checking that all archive entries resolve within the extraction directory and that file paths are unique. Inventory extracted files before inclusion decisions. Separate curation records from the raw source so every decision can be audited.
3. Use the archive SHA-256 as the immutable dataset identifier, even if the published version label remains the same. Assign a new derived version when inventory, curation, grouping, or split assignments change. Tie downstream evidence to those versions and hash the canonical split manifest.
4. Plan `.gitignore` coverage for raw and staged data, generated manifests with restricted test labels, bulk plots and models, credentials, and local caches. Track code, configs, and shareable documentation. Publish reviewed summaries and licensed samples selectively. Verify ignore rules and inspect staged changes before committing. Keep the test partition and its per-image labels local.

Verify the archive checksum before reuse, match the version record to the actual files, and keep a record of which manifests were generated from each archive. Proposed derived locations are `data/manifests/` for inventories and assignments and `artifacts/data/` for validation reports, plots, and integrity evidence.

## 3. Inventory schema and validation

Use CSV for row-level inventories and JSON for source and summary metadata. Store normalized archive-relative paths with portable separators and stable ordering. Use a stable image ID derived from the dataset identity and relative path, with hashes recorded separately for content identity. Retain one row for each discovered image candidate, including unreadable candidates. Track other files in the reconciliation report with their disposition.

| Fields | Meaning or validation |
| --- | --- |
| `dataset_id`, `image_id`, `relative_path`, `size_bytes` | Dataset identity, unique row identity, safe relative path, and source size. |
| `source_label`, `label` | Original folder label and reviewed canonical class. Unresolved labels receive no eligible class assignment. |
| `format`, `width`, `height`, `mode`, `channels`, `decode_status` | Observed properties, recorded only when available. Force a full decode to catch truncated pixel data. |
| `file_sha256`, `pixel_sha256` | Raw-byte hash and canonical decoded-pixel hash when decodable. The pixel hash includes shape and channel convention and is computed before resizing or any lossy transform. |
| `session_id`, `session_evidence`, `group_id` | Nullable capture-group provenance and the final indivisible split group. Record known relationships for audit. |
| `disposition`, `reason_code`, `review_ref` | Eligible, excluded, or pending review, with a reason and a link or identifier for any manual decision. |

Derive candidate labels from the expected class folders. Validate decoded format and image properties against published expectations and review examples for visual label agreement. Record deviations even when a reviewer accepts a decodable image. Reconcile discovered, eligible, excluded, and pending counts per class and document non-image files. Eligibility requires a readable image, a valid reviewed label, and a documented disposition for every anomaly. Block split freeze while any pending anomaly or material source discrepancy remains unresolved. Verify uniqueness, field completeness for eligible rows, and count reconciliation against the inventory.

## 4. Duplicate detection and anomaly policy

Compare file SHA-256 values for byte-identical files and canonical decoded-pixel SHA-256 values for identical pixels in different encodings. For near duplicates, a provisional perceptual hash with Hamming distance can produce candidate pairs for manual review. Use a contact sheet to check gesture geometry and hand details, since the green background can create false matches. Record algorithm, settings, pair decisions, and evidence. Keep the threshold provisional until candidate review establishes a useful operating range. Recheck reviewed groups after changes to decoding or screening.

| Finding | Disposition and verification |
| --- | --- |
| Corrupt or truncated image | Record the decoder failure and exclude it from eligible inputs. Review the inventory against the raw file and confirm there is no split assignment. |
| Unexpected file, missing class, or unsupported source property | Record the discrepancy. List non-image metadata separately. Exclude unmapped images unless trusted source evidence and a human decision establish a canonical label. Review unusual but decodable images against the input contract. |
| Same-label exact duplicates | Prefer a single representative chosen by stable path order after eligibility checks. Record links to excluded copies so exact counts and lineage are recoverable. |
| Conflicting labels within a duplicate group | Quarantine implicated images while reviewing source labels and pixels. Preserve original labels. Default to excluding the disputed component if the label conflict cannot be supported and resolved explicitly. Record every accepted correction with evidence and approval. |
| Ambiguous gesture | Flag for visual review against the three-class task. Exclude when one canonical label remains indeterminate and retain the reason and review record. |
| Confirmed near duplicates | Keep eligible members in one split group, or exclude redundant members under a documented reviewer decision. Keep pair judgments and flag unresolved high-risk pairs for conservative co-grouping or exclusion before the split freezes. |
| Capture-session evidence | Group images using reliable source metadata whose meaning and coverage can be verified. Treat filenames or timestamps without corroboration as review clues. Record gaps or unreliable metadata and use duplicate-based grouping where session identity cannot be established. |

Take connected components over confirmed duplicate links and trusted session links, including informative links through excluded copies. Give remaining eligible images singleton group IDs when they have no links. Verify that every accepted relationship resolves to one group and disclose residual risk where evidence is incomplete.

For the approved Phase 3 release, the owner waived exhaustive visual review. Apply the recorded pair judgments and add conservative co-grouping links for the remaining unreviewed same-label candidates. Keep policy links separate from visually confirmed near duplicates. Retain unreviewed image cases when the available evidence supports the existing disposition. Resolve cross-label or flagged evidence before finalizing the derived inventory. [Phase 3 decision and evidence](05_TEST_PLAN.md#phase-3-execution-evidence)

## 5. Exploratory data analysis

Create a local EDA notebook driven by the inventory. Before the split, use archive-wide inspection for curation, distribution summaries, and group discovery. Once the split is frozen, use training rows for model-facing previews and preprocessing decisions. Use validation for planned comparisons and reserve test examples for final error analysis. Record selection rules for example galleries and label manual observations as such.

| Analysis and question | Proposed evidence | Downstream decision |
| --- | --- | --- |
| Class distribution and representative gestures | Raw and eligible class-count bars, exclusion counts, and a seeded contact sheet per class. | Class-aware group allocation and interpretation of per-class support. |
| Dimensions, aspect ratio, original channels and modes | Frequency tables and width, height, and aspect-ratio plots. | Validate published properties and choose resize, padding, and RGB handling. |
| Pixel intensity and RGB distributions | Per-channel histograms, luminance proxy histograms, and per-image quantiles by class. | Verify scaling assumptions and investigate class-associated color cues. |
| Brightness and lighting | Per-image brightness spread, dark or bright tails, and representative lighting or shadow examples. | Decide whether modest brightness changes merit a controlled comparison. |
| Green background and shortcut risk | Sample border color statistics by class and compare border-only visual cues. Document that borders are an approximate background proxy. | Inspect padding choices and propose a validation diagnostic for sensitivity to background cues. |
| Gesture position, orientation, and framing | Reviewed sample grid for hand location, rotation, scale, handedness where discernible, and clipped fingers. | Review resize fidelity and plausible bounds for translation, zoom, rotation and flips. |
| Quality, ambiguity, and repetition | Galleries and disposition counts for blur, artifacts, ambiguous labels, duplicates, and candidate session clusters. | Refine eligibility, duplicate thresholds, and the group-preserving split. |

Save key figures and a short findings report with inventory identity, counts, inclusion filters, and decisions or questions for Phase 4. Compare figures against manifest totals and spot-check unusual rows against the approved source. Record the observed ranges after execution.

## 6. Split design and leakage audit

Target 70:15:15 by eligible image count. Allocate indivisible groups using per-group class counts and a recorded seed. Provisional allocation orders large or class-constraining groups first, uses reproducible tie-breaking, and chooses placements that reduce total and per-class deviations. A bounded retry with documented seeds or review of group assignments is acceptable when class coverage fails. Group integrity and class coverage take priority over exact proportions. If no feasible reviewed assignment covers all three classes in each split, stop before training and review the curation and allocation constraints with the team. When session evidence is absent, use the approved duplicate and conservative groups and disclose the remaining capture-session uncertainty.

Write the canonical split CSV with `dataset_id`, `inventory_version`, `split_version`, `image_id`, `relative_path`, `label`, `group_id`, `file_sha256`, `pixel_sha256`, and `split`. Record seed, algorithm and version, target and actual ratios, counts by split and class, and linked inventory and grouping hashes in JSON. Regenerate the exact assignments from the same inputs. Audit total eligible coverage, one assignment per image ID, zero cross-split group or content-hash overlap, and three-class support per partition. Freeze the reviewed manifest and quarantine test images and labels from downstream development. A material correction requires a new split version and comparable reruns before selection. If final outcomes have already been seen, further claims require a new evaluation plan.

## 7. Preprocessing contract

Keep deterministic transforms in `src/rps/preprocessing.py` and record the contract in the resolved configuration and model bundle. A provisional model boundary accepts float32 RGB in `[0, 1]` with shape `[batch, height, width, 3]`. Training and evaluation also consume integer class IDs from one explicit, versioned mapping, with three IDs used consistently for manifest labels, model outputs, and metrics. Freeze exact height, width, class index order, interpolation and padding after EDA.

| Step | Proposed behavior and rationale | Verification |
| --- | --- | --- |
| Decode and normalize color | Validate the declared format using decodable content. Normalize documented orientation consistently, including JPEG EXIF orientation, and convert to RGB under a single policy. Set supported mode and size limits before release. | Inspect images with different orientations and color modes. Compare CLI and Gradio decoded results with the shared pipeline. |
| Resize and pad | Preserve aspect ratio. Bilinear interpolation with antialiasing for downsampling and constant padding are provisional. Inspect edges and finger visibility and select padding using training previews. | Check shape, aspect-ratio behavior, padding placement, and visual preservation on reviewed training samples. |
| Convert pixel range | Cast and scale the 0–255 representation to float32 in `[0, 1]` once in the shared preprocessing path. The CNN accepts the scaled tensor directly. | Compare values before and after preprocessing and check a fixed sample across local and Colab. Verify the exported model expects the same tensor contract. |
| Batch and feed | Build `tf.data` from explicit split rows. Seed training shuffling, retain evaluation ordering and final partial batches, and prefetch. Consider caching deterministic work only when measured resources permit. Keep stochastic operations downstream of any cache. | Check image-to-label alignment, batch and sample counts, deterministic evaluation, reproducible ordering controls, and absence of augmentation in evaluation or inference. |

Both local inference surfaces plan to support PNG and JPEG (`.jpg` and `.jpeg`) via one decoder. Transfer the original upload bytes or equivalent file path from Gradio. This supplementary refinement resolves the PRD §7 and TRD §16 optional-format decision for initial inference under FR-09, FR-10, and AC-04. The source dataset validation still expects the published PNG format. Additional inference formats remain optional and outside the first release plan.

## 8. Training augmentation

Keep the baseline unaugmented. Candidate comparisons include mild translations, zoom, and rotation for framing variation, modest brightness changes for lighting variation, and horizontal flips if gesture meaning and framing are preserved. Derive candidate ranges from training observations. Preview before-and-after grids per class using fixed preview seeds. Reject settings that hide fingers, clip the hand, distort gesture cues, or introduce strong border artifacts. Record the random seed policy, transformation order, and active settings per run. Apply stochastic transforms only to training batches and compare a reviewed transform or group of related settings against the fixed baseline on the same validation split. Verify disabled augmentation reproduces deterministic preprocessing and that validation inputs remain identical across runs. [TensorFlow data augmentation guide](https://www.tensorflow.org/tutorials/images/data_augmentation)

## 9. Local–Colab transfer and integrity

Build a simple staging directory from the frozen train and validation manifest rows. Include only referenced train and validation image files, filtered assignments, the class mapping, preprocessing and run configuration, and the approved shared code revision. Record the full split hash by reference and a checksum for the filtered manifest. Exclude test images, test-label records, and any unfiltered archive or inventory. On Colab, verify the intended Python 3.13 and compatible TensorFlow/Keras versions, source revision, file membership, available classes, and checksums before training. Compare a fixed development sample’s preprocessed tensor or summary within a documented tolerance. Stop on missing files, mismatched mappings, or version incompatibility. Full training, training-time validation, and training-related experiments run in Colab. Evaluation, report writing, and inference run locally using the same codebase.

## 10. Outputs, acceptance, and requirement traceability

Raw decoded images plus frozen assignments and shared input code are sufficient to regenerate model-ready tensors. A cache is disposable and may be reused only with matching dataset, split, and preprocessing identities. Plan lightweight CSV, JSON, notebook, and Markdown outputs under the proposed paths in §2.

| Output and acceptance evidence | PRD requirements and criteria |
| --- | --- |
| Source metadata, inventory, validation and anomaly reports. Counts reconcile, excluded rows have reasons, and eligible records have reviewed labels and valid images. | FR-01, NFR-01 · AC-01, AC-05 |
| EDA notebook, figures, and decision notes. Each image analysis is reproducible from recorded inputs and tied to a decision or open question. | FR-02, NFR-02, NFR-04 · AC-05 |
| Frozen split manifest, grouping review, and leakage audit. Repeated generation agrees and the audit confirms coverage and zero prohibited overlap. | FR-03, NFR-01, NFR-04 · AC-01, AC-05 |
| Shared preprocessing contract, model-ready batch checks, and augmentation previews. All paths agree within the approved tolerance, with stochastic behavior confined to training. | FR-04, NFR-01, NFR-02 · AC-02, AC-04, AC-05 |
| Colab staging manifest, integrity results, and a documented local prediction input contract. Both inference interfaces accept PNG and JPEG and agree on shared examples. | FR-05, FR-07, FR-09, FR-10, NFR-03, NFR-04 · AC-02, AC-04 |

## 11. Residual risks and decision gates

The green background, similar lighting, limited environment diversity, and possible related captures constrain interpretation of validation and test results. Report the scope of the measured capture conditions. Near-duplicate screening can miss pairs, and unavailable session metadata leaves residual dependence even after audit. Record what evidence supported grouping and which checks were inconclusive. Reserve claims about real-world backgrounds for evidence that captures those conditions.

Phase 2 approves acquisition and inventory dispositions. Phase 3 approves EDA findings and grouping evidence, including the screening radius used for this release. Phase 4 approves the split, class mapping, preprocessing contract, and train/validation-only staging. Exact split counts, model input resolution, interpolation and padding details, augmentation ranges, and supported inference image limits remain pending review. Numerical quality targets, latency budgets, and repeatability tolerances are finalized using development evidence at the pre-test gate in the [Implementation Plan](04_IMPLEMENTATION_PLAN.md#phase-8--controlled-model-improvement).
