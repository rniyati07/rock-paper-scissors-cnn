# Rock–Paper–Scissors Image Classification with a CNN

This learning project will train a custom convolutional neural network to classify a still image of one hand as **Rock**, **Paper**, or **Scissors**. It follows the full deep learning workflow so students can explain how data quality, CNN architecture, and evaluation choices affect the result.

**Status** · Phase 1 technical checks passed and closure awaits explicit approval. Dataset preparation, training, evaluation, and the demo are planned work.

## Scope and learning outcomes

The planned system will audit and split the dataset, train a small CNN from scratch, compare controlled improvements, and evaluate one frozen model on held-out data. A local CLI and Gradio demo will accept PNG and JPEG (`.jpg` and `.jpeg`) and return a predicted gesture and model score. The project focuses on still-image classification and local inference. Webcam recognition, gesture localization, pretrained models, and cloud deployment are outside the initial scope.

Students will learn to detect leakage, reason about image tensors and convolutions, interpret learning curves and per-class errors, reproduce an experiment, and explain the model’s limits.

## Dataset and attribution

The sole dataset is **Rock–Paper–Scissors Images** by Julien de la Bruère-Terreault (DrGFreeman), released under **CC BY-SA 4.0**. Published materials describe 2,188 real hand-gesture photographs, RGB PNGs at 300 × 200 pixels. Published Rock and Paper counts disagree across the source descriptions. Phase 2 will verify the acquired archive and record observed counts. The green background and limited variation in lighting and environment constrain claims about unfamiliar scenes. [Kaggle dataset](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors) · [Creator’s dataset overview](https://github.com/DrGFreeman/rps-cv-data-science#dataset) · [Dataset Design and source notes](docs/03_DATASET_DESIGN.md) · [Dataset license](https://creativecommons.org/licenses/by-sa/4.0/)

The CC BY-SA notice applies to the dataset. A software license for this repository is pending approval.

## Stack and workflow

| Tool | Planned role |
| --- | --- |
| Python 3.13 and TensorFlow/Keras | Shared data pipeline, custom CNN, training and `.keras` model persistence |
| Local VS Code and Jupyter | Development, dataset validation and EDA, tests, final evaluation, and inference |
| Google Colab, GPU when available | Full epoch-based training, training-time validation, and controlled training experiments |
| Pytest, Git, and GitHub | Automated checks, versioned configuration and code, and review |
| CLI and Gradio | Local single-image prediction from the same evaluated bundle |

The proposed workflow is **validate and explore → freeze grouped splits → preprocess and train a baseline → compare by validation macro F1 → freeze the selected model → evaluate locally → demonstrate local inference**. Colab uses the same repository code and receives only the training and validation files and filtered manifests. The reserved test partition stays local. The CNN and its input resolution remain provisional until data and model design reviews.

## Proposed repository structure

| Path | Purpose |
| --- | --- |
| `docs/` | Planning documents and future reports |
| `notebooks/` | Guided EDA, Colab training entry point, and analysis |
| `src/rps/` | Shared data, model, training, evaluation, and inference modules |
| `configs/`, `tests/` | Versioned run settings and Pytest checks |
| `data/`, `artifacts/` | Local datasets, manifests, checkpoints, and run evidence. Raw data and bulk artifacts stay outside Git. |

## Documentation and roadmap

| Document | Purpose |
| --- | --- |
| [Product Requirements](docs/01_PRD.md) | Scope, objectives, requirements, and acceptance criteria |
| [Technical Requirements](docs/02_TRD.md) | Technical design and requirement traceability |
| [Dataset Design](docs/03_DATASET_DESIGN.md) | Provenance, curation, split, and preprocessing contracts |
| [Implementation Plan](docs/04_IMPLEMENTATION_PLAN.md) | Phase dependencies and review checkpoints |
| [Test Plan](docs/05_TEST_PLAN.md) | Verification cases, tolerances, and release gates |
| [Contributor guidance](AGENTS.md) | Repository development and review rules |

**Phase 0** closes after review of the planning documents. **Phases 1–4** cover learning and setup, dataset validation, EDA, and splitting. **Phases 5–8** cover CNN design, baseline training, validation analysis, and controlled improvement. **Phases 9–11** cover final evaluation, the local demo, and integration. Each phase requires human approval before work advances.

## Running the project

### Local setup in Windows PowerShell

Use an existing 64-bit Python 3.13 installation. From the project root, run the following to recreate the local environment and verify it.

```powershell
py -3.13 -m venv .venv313
.\.venv313\Scripts\python.exe -m pip install --upgrade pip
.\.venv313\Scripts\python.exe -m pip install --timeout 90 -r requirements.txt
.\.venv313\Scripts\python.exe -m pip check
.\.venv313\Scripts\python.exe -m ipykernel install --sys-prefix --name rps-cnn-313 --display-name 'Python 3.13 (RPS CNN)'
.\.venv313\Scripts\python.exe -m pytest -q -s
```

If the launcher does not find Python 3.13, invoke its `python.exe` directly for the first command. The local stack was verified on Python 3.13.3 with TensorFlow 2.21.0, Keras 3.13.2, and NumPy 2.4.6. VS Code defaults to `.venv313`. If an open workspace retained an earlier selection, use **Python: Select Interpreter** to choose `.venv313\Scripts\python.exe`. The original `.venv` remains available for rollback. [Phase 1 test evidence](docs/05_TEST_PLAN.md#phase-1-execution-evidence)

### Phase 1 notebooks

Open the [prerequisite notebook](notebooks/00_prerequisite_learning.ipynb) in VS Code, select **Python 3.13 (RPS CNN)**, and run all cells. If the kernel name is hidden, choose **Python Environments** and select `.venv313\Scripts\python.exe`. The notebook prints the active interpreter. The exercises and reflection questions are available for self-directed practice.

### Colab verification and Windows portability

1. In Colab, use **File → Upload notebook** to open the [verification notebook](notebooks/00_colab_verification.ipynb). Connect to the existing Python 3.13 GPU runtime and run all cells in order. The cells report device visibility and `pip check`. Record any version mismatch or dependency conflict before changing packages. The fixed model runs on CPU for the comparison.
2. Once the same-runtime check passes, download `phase1_smoke.keras` and `phase1_reference.json` from the printed directory using the Colab Files panel. Place the pair from that run in `artifacts/phase1/colab/` in the Windows workspace.
3. In PowerShell at the project root, run the portability test. It checks the file hash, framework versions, top class, and unrounded CPU scores against the Colab export. The maximum allowed absolute score difference is `1e-5`. The ordinary local test run skips this case until the artifact directory is provided.

```powershell
$env:RPS_COLAB_SMOKE_DIR = 'artifacts/phase1/colab'
.\.venv313\Scripts\python.exe -m pytest -q -s -k colab_export
```

Record the hosted notebook and Windows results in the [Test Plan](docs/05_TEST_PLAN.md#phase-1-execution-evidence).

### Later phases

| Section | Details will be added when verified |
| --- | --- |
| Training | Phase 6 will document the Colab entry point, filtered inputs, run configuration, and export. |
| Evaluation | Phase 9 will add the final evaluation procedure and link the recorded results. |
| Inference | Phase 10 will add CLI and Gradio launch commands and supported input limits. |

Reproducibility checks will record the code revision, archive and split checksums, versions, resolved configuration, class mapping, and seeds. Pytest will check deterministic contracts, artifact reload, and inference behavior. Human review will verify curation decisions, visual evidence, controlled comparisons, and the pre-test approval for model-quality targets, latency budgets, and repeatability tolerances. [Test Plan](docs/05_TEST_PLAN.md)
