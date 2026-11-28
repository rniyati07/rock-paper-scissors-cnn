# Implementation Plan

**Project** · Rock–Paper–Scissors Image Classification using CNN  
**Status** · Phase 0 draft for human review  
**Foundation** · [PRD](01_PRD.md) · [TRD](02_TRD.md) · [Dataset Design](03_DATASET_DESIGN.md)

## How to use this plan

The roadmap follows Phases 0–11 in order. Each phase starts after the team has reviewed and explicitly approved the previous phase’s exit evidence. Record the reviewer, decision, evidence references, and any required follow-up in the phase record. The practical exercises and reflection questions in each phase are for self-directed learning. The paths below are proposed additions or updates for their scheduled stages. Phase 0 ends at review of the planning documents.

From Phase 4 onward, Codex implements approved deliverables and prepares execution instructions, and the project owner runs the phase procedures and verification checks. Codex requires explicit authorization to run them. Phase conclusions and closure depend on actual execution evidence and the existing approval gates. [User-executed workflow](../AGENTS.md#user-executed-workflow-from-phase-4-onward)

Local VS Code is the primary development workspace. Google Colab is the exclusive execution environment for full epoch-based training, training-time validation, and training-related experiments. Local smoke checks, data workflows, offline validation analysis, final evaluation, and inference use the shared code and approved artifacts. Preserve the TRD §4 `src/rps/` package structure and use notebooks as readers, orchestrators, or explanations of shared logic.

## Milestones and dependencies

| Milestone | Phases | Dependency and evidence needed to advance |
| --- | --- | --- |
| Planning and setup | 0–1 | Approved documentation and a verified Python 3.13 workflow locally and in Colab. |
| Data readiness | 2–4 | Validated source, reviewed EDA and grouping, leakage audit, frozen splits and preprocessing contract. |
| Baseline readiness and evidence | 5–7 | Reviewed design, successful Colab training and export, full validation metrics, and prioritized error hypotheses. |
| Model freeze | 8 | Comparable experiments, a selected artifact, and approved numerical targets, budgets, and tolerances before any test access. |
| Final evaluation and presentation | 9–11 | Held-out report, local PNG/JPEG prediction parity, and integration and reproducibility evidence. |

The critical path is the reviewed environment → validated inventory → EDA and groups → frozen split and input contract → baseline → validation diagnosis → controlled comparison and pre-test sign-off → final evaluation → local demo → integration. Development reports and review may be iterative within each approved phase.

## Phase 0 — Project planning and documentation

- **Objective and entry** · Define the project, data lifecycle, and roadmap from the approved project brief and technical choices. Use local VS Code. Learn how product requirements, technical design, and acceptance evidence relate.
- **Ordered work** · Review the PRD and TRD. Design acquisition, curation, split, and input contracts. Derive the dependency chain and assign future documents and checks to their owning phases. Keep data- and experiment-dependent values provisional so later decisions can cite evidence.
- **Files and deliverables** · Four Phase 0 planning documents under `docs/`. This planning increment adds `03_DATASET_DESIGN.md` and `04_IMPLEMENTATION_PLAN.md`.
- **Verification and traceability** · Check referenced requirements and acceptance IDs against the PRD. Verify all 12 phases and six refinements are covered. This phase prepares evidence for FR-01–FR-10, NFR-01–NFR-04, and AC-01–AC-05.
- **Self-directed reflection** · How does an acceptance criterion differ from a provisional design choice? Which actions could compromise a held-out test before code exists?
- **Exit and approval** · Team reviews the four documents, confirms supplementary refinements, and explicitly approves Phase 1.

## Phase 1 — Prerequisite learning and environment setup

- **Objective and entry** · Establish a reproducible local CPU and Colab setup after Phase 0 approval. Learn Python environments, Python packaging, notebook kernels, tensors, and Git basics.
- **Ordered work** · Check Git and VS Code setup, create the Python 3.13 environment and Jupyter kernel, then resolve a minimal compatible TensorFlow/Keras dependency set. Check versions, imports, TensorFlow device visibility, and kernel identity in Colab. Record commands and versions, prepare ignore rules, and capture verification evidence in the Test Plan. Run the prerequisite notebook and use a small tensor forward pass and `.keras` save/reload smoke to identify runtime incompatibility before dataset work.
- **Files and deliverables** · Root `AGENTS.md`, `README.md`, `.gitignore`, a proposed root `requirements.txt`, `docs/05_TEST_PLAN.md`, and `notebooks/00_prerequisite_learning.ipynb`. Deliver recorded local and Colab compatibility and prerequisite notebook execution results. A tensor-level model smoke check is sufficient here.
- **Verification and traceability** · Capture Python and dependency versions, kernel and device checks, and model reload evidence. Confirm prerequisites can be repeated from the README. NFR-01, NFR-02, NFR-03 · AC-04, AC-05.
- **Self-directed reflection** · How can a notebook use a different Python interpreter from the terminal? What does a successful import establish about portability?
- **Exit and approval** · Team reviews the working local instructions, TP-01 compatibility evidence, TP-02 notebook execution evidence, and the Test Plan, then explicitly approves closure. Resolve incompatible runtimes and obtain approval before Phase 2.

## Phase 2 — Dataset acquisition and validation

- **Objective and entry** · Acquire and validate the approved source locally after setup approval. Learn dataset provenance, content hashes, image encoding, and label integrity.
- **Ordered work** · Download the Kaggle archive locally, preserve it, and record its identity and license. Extract safely, inventory images, compare observed properties to published expectations, then run exact file and pixel checks. Review problematic labels and files under Dataset Design §§2–4 and record every disposition or open item for EDA.
- **Files and deliverables** · `src/rps/dataset.py`, `tests/test_dataset.py`, source JSON and inventory CSV under `data/manifests/`, and a validation report under `artifacts/data/`.
- **Verification and traceability** · Verify the archive hash, reconcile counts, review every excluded file and reason, and confirm no material source discrepancy is left without an owner. FR-01, NFR-01, NFR-04 · AC-01, AC-05.
- **Self-directed reflection** · Why can two encodings of the same pixels have different file hashes? What evidence warrants a label correction?
- **Exit and approval** · Team approves the provenance, validated inventory, and anomaly dispositions for EDA. Candidate near duplicates or unresolved examples may advance only as explicit review items and must be resolved before split freeze.

## Phase 3 — Exploratory data analysis

- **Objective and entry** · Use the approved inventory locally to explain dataset characteristics and identify grouping and preprocessing implications. Learn distribution summaries, sampling bias, similarity screening, and shortcut learning.
- **Ordered work** · Use the EDA matrix in Dataset Design §5. Plot class counts, dimensions, aspect ratios, channels, pixel and RGB distributions, and brightness variation. Review border color and framing, orientation and gesture position, and quality and label ambiguity. Screen similar images, review candidate pairs and session evidence, then record grouping decisions and proposed preprocessing implications. Keep pre-split galleries focused on dataset curation and group discovery.
- **Files and deliverables** · `notebooks/01_eda.ipynb`, updates to `src/rps/dataset.py` and data tests, grouping records and EDA figures and findings under the established data paths.
- **Verification and traceability** · Reconcile plots with the inventory, reproduce sample selections, and review the rationale for each approved group and exclusion. FR-01, FR-02, FR-03, NFR-04 · AC-01, AC-05.
- **Self-directed reflection** · How might green border pixels correlate with labels? What evidence distinguishes a near duplicate from two different captures of the same gesture?
- **Exit and approval** · Team approves the EDA evidence, anomaly dispositions, grouping rationale, and constraints for splitting and preprocessing.
- **Approved review protocol** · The owner accepted limited visual review and conservative co-grouping for the remaining same-label candidates. The versioned inventory, component audit, and residual capture-session risk are recorded in the [Phase 3 execution evidence](05_TEST_PLAN.md#phase-3-execution-evidence).

## Phase 4 — Dataset splitting and preprocessing

- **Objective and entry** · Turn the reviewed eligible inventory into frozen, reproducible inputs. Work locally and use Colab only for staging and preprocessing compatibility smoke checks. Learn stratification, group leakage, transform order, labels, and `tf.data`.
- **Ordered work** · Allocate groups toward 70:15:15, audit inventories and overlap, and freeze split versions. Isolate the reserve, then use training previews to settle fixed input shape, RGB conversion, resize, padding, scaling, and class mapping. Implement shared deterministic processing and a configurable training-only augmentation path. Stage train and validation files and compare a fixed development example in both environments.
- **Files and deliverables** · `src/rps/dataset.py`, `src/rps/preprocessing.py`, relevant tests, `configs/data.yaml` or equivalent simple config, versioned split and staging records, leakage audit, and contract evidence under `artifacts/data/`.
- **Verification and traceability** · Regenerate identical assignments, check zero prohibited overlaps and all three classes per split, reconcile transferred rows, and verify shape, range, label alignment, train-only augmentation and local/Colab parity. FR-03, FR-04, NFR-01, NFR-04 · AC-01, AC-04, AC-05.
- **Self-directed reflection** · Why does group integrity take priority over exact split proportions? Where would an accidental second scaling step show up?
- **Exit and approval** · Team signs off on split identity, leakage audit, test isolation, data contract, and transfer checks. Resolve group-feasibility failures before Phase 5.
- **Recorded outcome** · Phase 4 closed with owner approval after TP-07–TP-11 passed. Use the frozen split and preprocessing contract recorded in the Test Plan for later approved work. [Phase 4 execution evidence](05_TEST_PLAN.md#phase-4-execution-evidence)

## Phase 5 — CNN architecture design

- **Objective and entry** · Design and implement a small, untrained baseline locally after the data contract is approved. Learn convolution, receptive fields, pooling, softmax, loss, and parameter counts.
- **Ordered work** · Draft the baseline design using the TRD’s candidate stack. Trace spatial shapes and parameter count, select a compact configuration that fits the inputs and expected resources, then implement a straightforward model builder. Record provisional optimizer and loss choices and check the three-output contract with a local forward pass.
- **Files and deliverables** · `docs/06_MODEL_DESIGN.md`, `src/rps/model.py`, `configs/baseline.yaml`, and `tests/test_model.py`. Deliver a reviewed shape and parameter summary and an untrained baseline ready for Colab.
- **Verification and traceability** · Check output shape, class ordering assumptions, finite scores and loss, and save/reload behavior. Review how the architecture supports the learning objectives. FR-04, FR-05, NFR-02, NFR-03 · AC-02, AC-05.
- **Self-directed reflection** · How do pooling and the dense head affect the number of parameters? Why does sparse categorical cross-entropy fit integer class IDs?
- **Exit and approval** · Team approves the model design, agreed training budget, and model contract before full training begins.

## Phase 6 — Baseline model training

- **Objective and entry** · Establish the unaugmented reference run from the approved design, pinned runtime, and verified train/validation staging. Learn optimization, generalization, checkpointing, and macro averaging.
- **Ordered work** · Locally implement the training entry point and prepare the experiment protocol and log. Declare the within-run checkpoint tie rule and evaluation ordering. Verify the Colab runtime and filtered inputs, then run full epochs in Colab and log loss, accuracy, and validation macro F1. Preserve the preferred checkpoint during training, export it with the resolved run record, and verify local CPU loading.
- **Files and deliverables** · `src/rps/train.py`, `src/rps/evaluate.py`, thin `src/rps/artifacts.py`, `notebooks/02_colab_train.ipynb`, baseline config and tests, `docs/07_EXPERIMENT_PLAN.md`, and `docs/08_EXPERIMENT_LOG.md`. Deliver `.keras` model and checkpoints, CSV history, JSON metadata, and a recorded reload check.
- **Verification and traceability** · Accumulate one confusion matrix over the entire validation split or compute from all aligned predictions and labels. Derive F1 per class as `2TP / (2TP + FP + FN)` in fixed order and take the unweighted mean with documented zero-division handling. Never average independently computed batch F1 scores. Verify full manifest coverage, including a partial last batch, and check that macro F1 identifies the saved checkpoint. FR-05, FR-07, NFR-01, NFR-03 · AC-02, AC-04, AC-05.
- **Self-directed reflection** · Why is the mean of per-batch F1 values generally different from corpus-level macro F1? What does the gap between training and validation curves suggest?
- **Exit and approval** · Team accepts a completed baseline with recoverable artifacts and auditable validation evidence. Preserve failed or interrupted attempts in the run log.

## Phase 7 — Validation and error analysis

- **Objective and entry** · Diagnose the approved baseline on local development data. Learn confusion matrices, per-class support, error taxonomies, and the distinction between diagnosis and causal evidence.
- **Ordered work** · Load the approved baseline bundle and evaluate the complete validation split locally. Compare with the checkpoint’s recorded metrics and review curves, per-class errors, and representative failures. Check potential background or framing shortcuts using development evidence. Prioritize a small set of controlled hypotheses with explicit expected effects.
- **Files and deliverables** · `notebooks/03_validation_analysis.ipynb`, updates to `src/rps/evaluate.py` and its tests, validation plots and report under the baseline run directory, and updates to the experiment plan and log.
- **Verification and traceability** · Compare validation sample count and label order to the manifest. Reconcile macro F1 under a documented development tolerance and link examples and plots to the checkpoint. Use observed differences as input to the Phase 8 tolerance decision. FR-05, FR-06, NFR-03, NFR-04 · AC-02, AC-05.
- **Self-directed reflection** · Which confusion pairs are backed by enough samples to discuss? Which change would isolate the proposed cause of an error pattern?
- **Exit and approval** · Team approves baseline findings, proposed controls, and the experiments to fund in Phase 8.

## Phase 8 — Controlled model improvement

- **Objective and entry** · Compare approved hypotheses against the frozen baseline and prepare the final test gate. Learn control variables, seed variation, model selection, and practical significance. Design and compare locally. Run all training and training-time validation in Colab.
- **Ordered work** · Register candidate changes, budgets, comparison precision, and tie rules before running comparisons. Run at least one controlled comparison with the same split, evaluation rules, and relevant fixed conditions. Export and log every completed or unsuccessful attempt. Calculate full-split validation macro F1 for each checkpoint and select across comparable runs by validation macro F1. Use the approved tie rule, with fewer parameters and then stable run ID as the TRD’s provisional order. Optionally repeat close results across seeds within the budget.
- **Files and deliverables** · Versioned experiment overrides under `configs/`, targeted module and test updates, run bundles, `docs/07_EXPERIMENT_PLAN.md`, `docs/08_EXPERIMENT_LOG.md`, and an approved selection record under `artifacts/` with exact model hash, split, class order, preprocessing contract, and comparison evidence.
- **Verification and traceability** · Audit run identities, controls and negative results. Benchmark representative development inputs on the intended local CPU. Verify local model reload and compare score or tensor outputs across local and Colab paths using development examples. Use this evidence to define the tolerances for Phase 10 interface checks. FR-04, FR-06, FR-07, NFR-01, NFR-03, NFR-04 · AC-02, AC-03, AC-04, AC-05.
- **Self-directed reflection** · How large is the observed gain relative to run variation? Which target is supported by development evidence and which still needs a documented trade-off?
- **Exit and approval** · Mandatory pre-test gate. Team signs off on the frozen candidate and the exact measurable model-quality targets, latency budgets with hardware and timing protocol, and repeatability tolerances with comparison scope. Record values, units, methods and pass conditions from development evidence. Hold Phase 9 until all are approved, even if the baseline remains the selected candidate.

## Phase 9 — Final performance evaluation

- **Objective and entry** · Locally assess the frozen model on the reserved test split after pre-test approval. Learn held-out estimation, metric support, latency measurement, and limits of generalization.
- **Ordered work** · Verify the selection and audit identities. Evaluate the frozen model on the full test split, generate labeled metrics and confusion matrix, and link its training curves. Analyze representative test errors after scoring. Measure parameter count, `.keras` size, and latency using the frozen protocol. Compare outcomes to the approved targets, report misses and limitations, and log the result before starting the demo.
- **Files and deliverables** · `src/rps/evaluate.py`, `notebooks/04_final_evaluation.ipynb`, metrics and plots in `artifacts/`, and the first `docs/10_FINAL_REPORT.md` draft.
- **Verification and traceability** · Reconcile prediction count and class support with the test manifest, confirm scores derive from the recorded model hash, check all TRD §13 metrics and benchmark conditions, and document interpretation of failures. FR-08, NFR-03, NFR-04 · AC-03, AC-05.
- **Self-directed reflection** · What population does this score describe? How should a missed target be reported once the test has been observed?
- **Exit and approval** · Team accepts the evaluation evidence and any target misses as explicit findings. Any proposed test-informed change requires a separate approved evaluation plan before a fresh held-out claim.

## Phase 10 — Local inference and Gradio demonstration

- **Objective and entry** · Expose the approved evaluated bundle on a local CLI and Gradio interface. Learn artifact contracts, image decoding, error handling, and model score interpretation.
- **Ordered work** · Build a shared predictor with startup artifact checks. Use the same byte-path decoder, RGB and orientation handling, and scaling for both interfaces. Support PNG and JPEG (`.jpg`, `.jpeg`) in each, return top label and score, and surface invalid file or invalid bundle errors. Run the Gradio demo on a local-only endpoint. Update usage instructions with setup, supported inputs, and limitations.
- **Files and deliverables** · `src/rps/inference.py`, `src/rps/cli.py`, `src/rps/app.py`, relevant tests, and `README.md` updates. Deliver both local entry points and acceptance evidence.
- **Verification and traceability** · Exercise valid PNG, `.jpg`, and `.jpeg` inputs through both interfaces, using approved development examples, including JPEG conversions of those examples for interface checks. Confirm label equality and unrounded score agreement within the frozen tolerance. Verify missing, corrupt, unsupported, or oversized input errors and invalid artifact handling. FR-04, FR-07, FR-09, FR-10, NFR-03 · AC-04.
- **Self-directed reflection** · Why can upload-time conversion cause a disagreement between interfaces? What does a high score on an unfamiliar image establish?
- **Exit and approval** · Team approves working local entry points, clear errors, documented parity, and use of the evaluated artifact.

## Phase 11 — Integration testing and final documentation

- **Objective and entry** · Verify the end-to-end handoff locally after demo approval. Use Colab only for a separately approved training-side reproduction when needed. Learn regression testing, reproducibility, and communicating limitations.
- **Ordered work** · Run Pytest in a clean supported environment, reconstruct the run record from recorded inputs, and repeat local bundle and interface smoke checks. Reconcile all PRD requirements and acceptance evidence, finalize the README and report, and summarize technical decisions and results. Review repository changes and identify the exact revision and artifact bundle for handoff.
- **Files and deliverables** · `tests/`, `README.md`, `docs/05_TEST_PLAN.md`, `docs/08_EXPERIMENT_LOG.md`, and `docs/10_FINAL_REPORT.md` updates. Deliver verification results and a completion record.
- **Verification and traceability** · Check data lineage, artifact provenance, baseline and comparison evidence, final metric coverage, PNG/JPEG parity, and clean instructions. FR-01–FR-10, NFR-01–NFR-04 · AC-01–AC-05.
- **Self-directed reflection** · Which evidence lets another student rerun the workflow? What findings would change if the capture environment changed?
- **Exit and approval** · Team reviews the final checklist and signs off on the completed project and documented limitations.

## Repository and environment workflow

- Implement the smallest reviewable slice needed for each phase. Work from an approved revision, use focused Git changes, run the relevant Pytest and notebook or entry-point checks, and review staged paths and diffs before committing. Use GitHub for revision review and track source and configurations. Keep raw data, test labels, bulk artifacts, and credentials outside Git history.
- From Phase 1, record OS, Python interpreter, package versions, Jupyter kernel, and available accelerator. Verify installs and a forward-pass/serialization smoke in both environments after dependency changes. Pin a jointly compatible TensorFlow/Keras version set and record any approved platform-specific dependency differences. If the hosted Colab runtime differs from Python 3.13 or the agreed core ML package versions, record the discrepancy and pause for team review before changing the runtime or pins. Verify imports and CPU reload again on every export-consuming change.
- Keep one folder per run or attempt with `.keras` artifacts when produced, resolved config, CSV history, JSON metadata and a small file/checksum manifest. Use Markdown for the experimental hypothesis, decisions, failures, and outcomes. Record Git revision, archive and split hashes, seeds, runtime and device, and checkpoint lineage when relevant. Shared file I/O in `artifacts.py` stays limited to reading, writing, and validation needed by the actual workflow.
- Use the same validation scoring rule for baseline, best checkpoint, and cross-run comparison. Retain incomplete runs and record whether a retry resumed a verified checkpoint with valid optimizer state or restarted from a new run ID. Persist exports outside ephemeral Colab storage.

## Documentation schedule

| File | First created | Updates and ownership |
| --- | --- | --- |
| `docs/05_TEST_PLAN.md` | Phase 1 | Extend as data, training, evaluation, and inference contracts become testable. Record final checks in Phase 11. |
| `AGENTS.md` | Phase 1 | Update when approved repository workflows or verification commands change. |
| `README.md` | Phase 1 | Add data and training instructions as they work. Add demo commands in Phase 10 and validate from a clean environment in Phase 11. |
| `docs/06_MODEL_DESIGN.md` | Phase 5 | Record approved architecture rationale and revisions arising from controlled evidence. |
| `docs/07_EXPERIMENT_PLAN.md` | Phase 6, before the baseline run | Define protocol and controls, then update for hypotheses in Phase 7 and approved comparisons and pre-test decisions in Phase 8. |
| `docs/08_EXPERIMENT_LOG.md` | Phase 6, before the baseline run | Record all attempts, outputs, and validation selection during Phases 6–8. Reference the final evaluation record in Phase 9. |
| `docs/10_FINAL_REPORT.md` | Phase 9 | Draft from final evaluation evidence, add inference checks in Phase 10, and finalize in Phase 11. |

## Review findings and specification relationship

| Review finding | Planned resolution and verification |
| --- | --- |
| Inference image formats | Dataset Design §7 and Phase 10 add initial JPEG/JPG support alongside required PNG under PRD §7 and TRD §16. Check parity for both formats. |
| Simplicity | Follow TRD §§4–5 and 17 with lightweight Python modules, configs, CSV histories, JSON metadata and checksums, and Markdown evidence. Phase 6 validates the local reload workflow. |
| Image-specific EDA | Dataset Design §5 maps each required image analysis to Phase 3 figures and follow-on decisions. |
| Anomaly policies | Dataset Design §§3–4 and Phases 2–4 require recorded review, exclusions or evidence-backed label corrections, and duplicate/session-aware grouping. |
| Numerical targets | PRD §10 and TRD §13 are made actionable by the Phase 8 approval gate for quality targets, latency budgets, and repeatability tolerances. Phase 9 uses the frozen protocol. |
| Macro F1 | TRD §§10 and 13 are enforced in Phases 6–8 with full-split accumulation for checkpoint selection and model comparison. Audit against aligned validation predictions and labels. |

## Risk and fallback decisions

| Risk | Response and review point |
| --- | --- |
| Source discrepancy or missing session metadata | Verify the acquired archive in Phase 2, record conflicts, and use confirmed duplicate grouping with explicit residual risk in Phases 3–4. |
| Group integrity prevents class coverage | Preserve groups, check documented assignment candidates, and pause at the Phase 4 gate if coverage cannot be obtained. |
| Colab accelerator or runtime interruption | Verify compatible versions and device each session, persist checkpoints and logs, and reduce experiment count or use Colab CPU only if feasible and approved. Retain at least the baseline and one comparable experiment. |
| Overfitting or validation instability | Keep comparisons small, predeclare controls, use seed repeats when feasible, and report uncertainty. Phase 8 can select the baseline if it wins under the declared rule. |
| Green-background shortcut or domain shift | Link Phase 3 observations and Phase 7 diagnostics to the Phase 9 interpretation and the Phase 10 demo limitations. |
| Artifact mismatch or poor local portability | Validate checksums, class order, input contract and runtime versions. Re-export the same checkpoint when packaging fails and repeat parity checks before acceptance. |

## Requirement coverage and final completion checklist

The PRD and TRD remain the sources for the requirement text and original acceptance checks. The phase references above identify where the evidence is produced. Phase 11 verifies that the evidence is complete and readable.

- [ ] **AC-01** · Validated inventory, exclusions, EDA, frozen split, and leakage evidence account for the approved dataset and preserve class coverage and groups. FR-01–FR-03, NFR-04.
- [ ] **AC-02** · Baseline and at least one controlled comparison are reproducible from saved inputs. Checkpoint and model selection use full-validation macro F1 and the recorded tie rules. FR-04–FR-07, NFR-01.
- [ ] **AC-03** · Confirm pre-test approval and record final performance, error analysis, resource measurements, and any missed targets against the frozen identity. FR-08, NFR-03, NFR-04.
- [ ] **AC-04** · Verified local CPU reload, clear errors, and deterministic CLI/Gradio label and score parity for PNG and JPEG under the approved tolerance. FR-04, FR-07, FR-09, FR-10, NFR-03.
- [ ] **AC-05** · A reviewer can locate the code, dataset and split identities, run settings, interface instructions, verification results, and limits on generalization. NFR-01, NFR-02, NFR-04.
- [ ] Human approval and evidence are recorded for all phases. The final report and README identify the selected artifact and its intended capture conditions.
