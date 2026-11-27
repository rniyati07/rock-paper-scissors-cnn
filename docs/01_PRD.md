# Product Requirements Document

**Project** · Rock–Paper–Scissors Image Classification using CNN  
**Phase** · 0 — Requirements  
**Status** · Draft for team review

## 1. Project overview and problem statement

Build a system that accepts a still photograph of one hand showing Rock, Paper, or Scissors and returns the predicted gesture and a model score. The project covers the lifecycle from acquiring and inspecting the approved dataset through training a custom convolutional neural network (CNN), evaluating it, and making predictions through a local command-line interface and Gradio demo.

## 2. Motivation and objectives

The primary objective is to learn how data quality, image representation, CNN design, and experimental choices affect classification. The final demo gives students and reviewers a way to inspect the resulting model on individual images. The evaluation must establish how the system performs under the dataset's capture conditions and explain what evidence would be needed to assess performance in other settings.

## 3. Learning objectives and expected outcomes

By the end of the project, a student should be able to:

- Audit labels and image properties, identify likely leakage paths, and explain how the split protects the evaluation.
- Explain how convolution, activation, pooling, and the classifier head transform an image into class scores.
- Read learning curves and per-class results, diagnose plausible underfitting or overfitting, and test one hypothesis at a time.
- Explain why macro F1 is the primary comparison metric and use a confusion matrix and example errors to interpret it.
- Reconstruct a run from its recorded inputs and settings, load the selected model locally, and explain where its predictions are likely to be unreliable.

## 4. Scope

**In scope** · Dataset acquisition, validation, exploratory data analysis (EDA), splitting, preprocessing, training-only augmentation, a custom CNN baseline, controlled improvements, final evaluation, model export, local still-image inference, and tests and documentation for those workflows.

**Outside the initial scope** · Alternative or synthetic datasets, multi-dataset integration, pretrained models, real-time webcam recognition, gesture localization, multi-gesture recognition, a complex frontend or backend, cloud deployment, and advanced MLOps infrastructure.

## 5. Functional requirements

| ID | Requirement |
| --- | --- |
| FR-01 | Acquire the DrGFreeman dataset, record its provenance and license, validate image and label properties, and report how discrepancies and unusable files are handled. |
| FR-02 | Produce EDA showing image properties, class distribution, representative examples, and evidence of capture bias or repeated images. |
| FR-03 | Produce reproducible train, validation, and test assignments. Keep accepted duplicate groups and identifiable capture sessions within a single split, retain all three classes in each split, and record any deviations from the proposed proportions. |
| FR-04 | Apply equivalent deterministic image preprocessing across training, evaluation, and inference. Apply random augmentation only to training examples. |
| FR-05 | Train a custom CNN from scratch in Google Colab and record its configuration, learning curves, and validation results as the baseline. |
| FR-06 | Run and document controlled model-improvement experiments. Select the model using validation macro F1 and a recorded tie-breaking rule. |
| FR-07 | Export loadable `.keras` models and checkpoints with the class mapping, histories, configuration, and metadata required to reproduce or audit a run locally. |
| FR-08 | Evaluate the frozen selection on the reserved test set. Report accuracy, precision, recall, macro F1, per-class metrics, a confusion matrix, learning curves, misclassification analysis, and model efficiency. |
| FR-09 | Accept a supported image path through a local CLI and return the predicted label and score or an actionable error. |
| FR-10 | Accept a supported still image through a simple local Gradio interface and return the predicted label and score or an actionable error. |

## 6. Non-functional requirements

| ID | Requirement |
| --- | --- |
| NFR-01 | **Reproducibility.** Record the code revision, dataset identity, split identity, seeds, configuration, and runtime versions for every reported run. |
| NFR-02 | **Maintainability.** Use shared Python modules for reusable behavior and notebooks for exploration, explanation, and analysis. Document the supported entry points. |
| NFR-03 | **Portability and resource use.** Verify that exported models load and run on a supported local CPU environment. Report parameter count, `.keras` file size, and measured single-image latency with the benchmark conditions. |
| NFR-04 | **Evidence integrity.** Keep the test split reserved for final evaluation. Qualify performance claims by the measured capture conditions and disclose residual leakage or domain-shift risks. |

## 7. Input and output specifications

| Surface | Input | Output |
| --- | --- | --- |
| Dataset pipeline | Approved dataset archive containing labeled gesture photos | Validation and EDA reports, an image inventory, and versioned split assignments |
| Training and evaluation | Split assignments, deterministic preprocessing settings, and run configuration | Model and checkpoints, epoch history, validation comparison, and final test report |
| CLI and Gradio | One supported, decodable still image containing a single Rock, Paper, or Scissors gesture. PNG support is required. Additional format and image-size limits are pending. | Top predicted label (`Rock`, `Paper`, or `Scissors`) and its score on a 0–1 scale, or an actionable error for an invalid input or unavailable model |

The prediction score is a model output. Its calibration has yet to be established. A valid photo outside the intended capture conditions can still receive a high score.

## 8. High-level system workflow

1. Acquire and validate the approved dataset locally, then inspect class balance and capture conditions.
2. Resolve data-quality issues, freeze split assignments, and review the leakage audit.
3. Develop the CNN locally and run the baseline and controlled training experiments in Colab using the same repository code.
4. Export candidate models and run evidence. Compare candidates using validation results and freeze the selected model and preprocessing contract.
5. Run final evaluation locally on the reserved test split. Publish the findings and expose the selected model through the local CLI and Gradio demo.

## 9. Expected deliverables

- Phase 0 PRD and TRD for review.
- Dataset provenance, validation and EDA outputs, split manifest, and leakage audit.
- Educational notebooks, reusable Python modules, Colab training entry point, configuration files, and Pytest suite.
- Baseline and experiment histories, reproducible summaries, selected `.keras` bundle, and validation-based selection record.
- Final evaluation report with figures and efficiency measurements, plus operating instructions for local CLI and Gradio inference.

## 10. Success and acceptance criteria

| ID | Check | Evidence |
| --- | --- | --- |
| AC-01 | Every discovered image is accounted for as included or excluded with a reason. Every included image has a valid class and exactly one split. All three classes appear in each split, and the audit finds zero overlap among file IDs, exact duplicates, and confirmed duplicate or session groups. | Validation report, split manifest, and leakage audit |
| AC-02 | The baseline and at least one controlled comparison have saved settings, histories, and validation macro F1. The selection record identifies the winner using validation evidence only. | Run bundles and comparison report |
| AC-03 | The frozen candidate has a complete final test report with all required predictive metrics, class support, diagnostic plots, an error analysis, and measured efficiency. | Evaluation report and artifact identity |
| AC-04 | The Colab export loads on the supported local CPU setup. The CLI and Gradio produce the same label and agree within a documented score tolerance for the same supported image and bundle. Each returns a clear error for invalid input. | Pytest and interface smoke-check results |
| AC-05 | A reviewer can identify the code and data used for a reported run and explain the split, baseline, selected change, and limits on generalization from the delivered materials. | Run metadata, notebooks, and documentation |

Numerical model-quality targets, latency budgets, and repeatability tolerances remain pending. The team will assess feasibility using dataset validation and development evidence and record any adopted target before opening the test split for final evaluation.

## 11. Assumptions and constraints

- **Approved** · Python 3.13, TensorFlow/Keras, local VS Code development, Colab for full epoch-based training and training validation, Jupyter and Python modules, a CNN trained from scratch, and controlled experiments. Persist models in `.keras`, track experiments with configuration files plus CSV, JSON, and Markdown, test with Pytest, use Git and GitHub, and deliver CLI and local Gradio inference.
- **Approved** · The sole source is DrGFreeman's Rock–Paper–Scissors Images dataset. The project starts from a published total of 2,188 and expects real RGB PNG photographs at 300 × 200 pixels with a CC BY-SA 4.0 license. Confirm the actual archive properties during data validation. [Kaggle dataset](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors) · [Dataset creator’s overview](https://github.com/DrGFreeman/rps-cv-data-science#dataset)
- **Provisional** · Target a 70:15:15 split by image count after cleaning, subject to class coverage and group integrity. Capture-session metadata may be incomplete or unavailable.
- **Pending** · Exact split counts, model input resolution, layer configuration, hyperparameters, augmentation settings, supported optional image formats, and numerical targets. Confirm local and Colab dependency versions against both runtimes.

## 12. Risks and limitations

| Risk | Impact and response |
| --- | --- |
| Uniform green background and limited environmental diversity | The model may learn background or lighting cues. Report the limits of held-out results and use validation diagnostics to examine shortcut behavior. |
| Repeated or related captures | Shared images or sessions can inflate measured performance. Audit exact and near duplicates, group known sessions, and disclose the remaining risk where provenance is incomplete. |
| Small dataset and repeated validation comparisons | Overfitting or unstable rankings can drive selection. Keep the experiment set focused, retain comparable controls, and document uncertainty and any seed repeats. |
| Colab availability and local runtime compatibility | GPU access and session duration can vary. Preserve checkpoints and histories, verify Python and framework compatibility, and reduce experiment scope if needed while retaining the agreed acceptance evidence. |

## 13. Potential future enhancements

Subject to review, extend error analysis by capture condition where metadata supports it, study calibrated confidence and a low-confidence warning using development data, or explore a smaller custom CNN for CPU inference. Review these requirements before starting the next documentation stage. Review the dataset split before training and the selection record before final test evaluation.
