# Repository instructions for Codex

## Project and sources of truth

This project teaches the full image-classification lifecycle through a custom TensorFlow/Keras CNN trained from scratch on DrGFreeman’s Rock–Paper–Scissors Images, the sole approved dataset. Use Python 3.13, Jupyter, Pytest, Git and GitHub, and `.keras` artifacts. The final deliverables include local evaluation and local CLI and Gradio still-image prediction.

- Use the [PRD](docs/01_PRD.md) for scope, requirement IDs, and acceptance criteria. Use the [TRD](docs/02_TRD.md) for technical constraints and architecture. Consult the [Dataset Design](docs/03_DATASET_DESIGN.md) for data contracts, the [Implementation Plan](docs/04_IMPLEMENTATION_PLAN.md) for phase dependencies, and the [Test Plan](docs/05_TEST_PLAN.md) for verification evidence and gates.
- The current supplementary decisions move the Test Plan, this file, and the initial README into Phase 0 for review before implementation, and add JPEG/JPG to the PNG inference requirement on both interfaces. Record later refinements and raise conflicts for review before changing approved decisions or evidence.

## Workflow for each phase

1. Confirm authorization for the current phase. Read the relevant specifications and inspect the affected code, repository state, and existing evidence.
2. Explain the prerequisite concepts, proposed approach, significant decisions and alternatives, and expected outputs. List intended file changes, dependencies, ordered implementation tasks, and acceptance IDs.
3. Wait for human approval of the phase implementation plan before changing code. Implement in small, reviewable steps.
4. Explain important code, tensor shapes, mathematical operations, and the design rationale. Give exact execution commands and run available checks from the Test Plan.
5. Report the phase and status, files changed, decisions and evidence by requirement and check ID, commands and actual results, and unresolved issues or unrun checks. Update relevant documentation and stop for review. A phase is complete only when required evidence exists and a human approves it. Start each subsequent phase only with explicit approval.

Integrate self-directed learning and practical exercises throughout development. In Phase 1, use the [prerequisite notebook](notebooks/00_prerequisite_learning.ipynb) to cover NumPy arrays, indexing and dimensions, image loading and visualization, RGB representation, pixel normalization, tensor creation and shapes, batch dimensions, a simple forward pass, and basic loss. Verify notebook execution for TP-02. Exercise inputs stay outside the project dataset and model evaluation.

## Implementation and verification

- Use simple, readable modules under `src/rps/`. Keep reusable logic in modules, and let notebooks explain concepts, explore data, call shared code, and record findings. Use local VS Code for development, data preparation, testing, offline evaluation, and inference. Run all full epoch-based training, training-time validation, and training experiments in Colab from the same code revision and compatible runtime.
- Preserve the raw archive and original labels. Record anomalies, reviewed exclusions, and evidence-backed label decisions. Use SHA-256 file and decoded-pixel hashes, straightforward perceptual hashes, manual candidate review, and simple group assignment. Preserve group integrity and three-class split coverage. Record any reviewed change to a frozen split or approved model artifact under a new identity.
- Before splitting, limit archive-wide inspection to curation, validation, descriptive inventory and group discovery. After freeze, use train for model-facing EDA, preprocessing and augmentation previews, and validation for comparisons and selection. Reserve test data for approved final evaluation and subsequent error analysis. Keep test images, per-image test labels, and unfiltered manifests local. Apply augmentation to training only. Share deterministic preprocessing and the class mapping across all paths.
- Establish an unaugmented baseline, register controlled changes and tie rules, and record failed and neutral runs. For both checkpointing and model selection, compute macro F1 over the entire validation split or an equivalent accumulated confusion matrix. Batch-level F1 averages are invalid. Freeze the selected artifact and obtain approval for quality targets, latency budgets, repeatability and interface parity tolerances before test access.
- Use Pytest for suitable deterministic checks and manual review for visual and statistical evidence. Record each phase’s tolerance and rationale before comparison. Preserve failures and investigate before proposing a reviewed protocol change. Validate `.keras` export and local CPU reload, verify PNG/JPG/JPEG and EXIF handling through the shared predictor, and compare interface scores before display rounding.
- Keep configuration, CSV history, JSON metadata and Markdown evidence simple and versioned where appropriate. Record code, data, split, runtime and artifact identities. Verify dependencies in both runtimes before adopting or changing pins. Keep raw data, unfiltered manifests, credentials and bulk artifacts out of Git. Commit or push only with explicit approval and preserve unrelated user changes.
- After establishing context, read the files relevant to the current phase, reuse stable logic, and avoid broad rewrites or repeating whole files in reports. Keep explanations focused on decisions and concepts the learner needs. Update the README, tests, and phase evidence as their owning work changes.

## Prohibited actions

- Introducing alternate datasets, synthetic data for model development, pretrained models, or unnecessary tooling. Changes to finalized decisions or project scope require explicit approval.
- Beginning an unapproved phase, opening the reserve for development, running full training locally, or silently changing raw data, labels, frozen splits, or approved artifacts.
- Fabricating results, concealing failed checks, exposing credentials, committing disallowed files, or marking incomplete or unapproved work as complete.
