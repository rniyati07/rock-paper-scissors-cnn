# Rock–Paper–Scissors Image Classification with a CNN

This learning project will train a custom convolutional neural network to classify a still image of one hand as **Rock**, **Paper**, or **Scissors**. It follows the full deep learning workflow so students can explain how data quality, CNN architecture, and evaluation choices affect the result.

**Status** · Phases 1–4 are closed. The frozen split contains 1,531 training, 329 validation, and 328 test images. The approved 96 × 144 RGB preprocessing contract passed local–Colab parity. Phase 5 awaits separate approval. [Phase 4 evidence](docs/05_TEST_PLAN.md#phase-4-execution-evidence)

## Scope and learning outcomes

The planned system will audit and split the dataset, train a small CNN from scratch, compare controlled improvements, and evaluate one frozen model on held-out data. A local CLI and Gradio demo will accept PNG and JPEG (`.jpg` and `.jpeg`) and return a predicted gesture and model score. The project focuses on still-image classification and local inference. Webcam recognition, gesture localization, pretrained models, and cloud deployment are outside the initial scope.

Students will learn to detect leakage, reason about image tensors and convolutions, interpret learning curves and per-class errors, reproduce an experiment, and explain the model’s limits.

## Dataset and attribution

The sole dataset is **Rock–Paper–Scissors Images** by Julien de la Bruère-Terreault (DrGFreeman), released under **CC BY-SA 4.0**. The acquired ZIP contained two byte-identical sets of 2,188 RGB PNGs at 300 × 200 pixels. Phase 2 retained 726 Rock, 712 Paper, and 750 Scissors images in the eligible inventory. Published class counts differ across source descriptions. The green background and limited variation in lighting and environment constrain claims about unfamiliar scenes. [Kaggle dataset](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors) · [Creator’s dataset overview](https://github.com/DrGFreeman/rps-cv-data-science#dataset) · [Observed counts and validation](docs/05_TEST_PLAN.md#phase-2-execution-evidence) · [Dataset license](https://creativecommons.org/licenses/by-sa/4.0/)

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

### Phase 2 dataset validation

From the project root, run the approved archive through the local Python 3.13 environment. The command preserves the download and writes the raw copy, inventory, and validation results to Git-ignored paths keyed by the archive’s SHA-256.

```powershell
.\.venv313\Scripts\python.exe -m src.rps.dataset "$env:USERPROFILE\Downloads\archive (4).zip"
.\.venv313\Scripts\python.exe -m pytest tests/test_dataset.py -q
```

Review the checks and observed counts in the [Test Plan](docs/05_TEST_PLAN.md#phase-2-execution-evidence).

### Phase 3 exploratory data analysis

Open [01_eda.ipynb](notebooks/01_eda.ipynb) in VS Code, select **Python 3.13 (RPS CNN)**, and run all cells. The notebook verifies the approved archive and inventory, then prints the local figure and review directories. From the project root you can also execute it with a fresh kernel from PowerShell.

```powershell
.\.venv313\Scripts\python.exe -c "import nbformat; from nbclient import NotebookClient; nb = nbformat.read('notebooks/01_eda.ipynb', as_version=4); NotebookClient(nb, timeout=300, kernel_name='rps-cnn-313').execute()"
.\.venv313\Scripts\python.exe -m pytest -q
```

The local `image-review.csv`, `pair-review.csv`, and `calibration-review.csv` contain the owner's recorded case decisions. The accompanying `review-policy.json` records the approved waiver for the remaining cases. Rerunning validates those records and regenerates the versioned inventory, group map, and review snapshot. The output `summary.json` lists their filenames and checksums. [Phase 3 evidence](docs/05_TEST_PLAN.md#phase-3-execution-evidence)

### Phase 4 owner execution

Use PowerShell from the repository root, the existing `.venv313`, and the local Phase 3 release. [configs/data.json](configs/data.json) pins the dataset, inventory and group hashes. Its 96 × 144 resolution and augmentation ranges are candidates until you review the training previews. Only aggregate counts, hashes, check results and training-preview observations are needed for reporting.

**1. Run the local verification suite.** The split, preprocessing and staging tests use temporary fixtures. The three test files cover TP-07 through TP-11. Expect a passing Pytest summary and exit code zero. Pytest writes temporary test outputs.

```powershell
$py = '.\.venv313\Scripts\python.exe'
& $py -m pytest tests/test_split.py tests/test_preprocessing.py tests/test_phase4_stage.py -q
```

**2. Generate and audit the candidate.** The [Phase 4 command module](src/rps/phase4.py) calls the group allocator in [dataset.py](src/rps/dataset.py). It reads the approved inventory and groups, preserves each whole group, and targets 70:15:15 by image count. Run each command after the previous one succeeds.

```powershell
& $py -m src.rps.phase4 split --config configs/data.json
& $py -m src.rps.phase4 audit --config configs/data.json --regenerate
```

Expect JSON with `split_version`, manifest `sha256`, per-class counts, ratio deviations, and overlap counts. Review the totals against 2,188 images (726 Rock, 712 Paper, 750 Scissors). Every partition must contain all three classes, regeneration must match byte-for-byte, and all four overlap counts must be zero. The local assignments are in `data/manifests/<dataset_id>/splits/<split_version>/split.csv`. The aggregate evidence is in `artifacts/data/phase4/<split_version>/audit.json`. Below, **the evidence directory** means this latter directory.

**3. Freeze the reviewed split.** If you accept the candidate, enter the full manifest `sha256` from the audit. Success prints the verified identity and writes `frozen.json` in the evidence directory. The full assignment CSV and all test images stay local.

```powershell
$splitHash = Read-Host 'Paste the reviewed manifest sha256'
& $py -m src.rps.phase4 freeze --config configs/data.json --expected-sha256 $splitHash
```

**4. Review the training previews.** Open [phase4_data_preprocessing.ipynb](notebooks/phase4_data_preprocessing.ipynb) in VS Code. Select **Python 3.13 (RPS CNN)**, restart the kernel and run all cells. [preview.py](src/rps/preview.py) selects seeded training rows and displays the original, candidate size, 128 × 128 alternative, and each augmentation separately. [preprocessing.py](src/rps/preprocessing.py) provides the shared decoder and the `tf.data` batch builder tested in step 1.

Expect two PNG grids, `preview-summary.json` in the evidence directory, and a candidate contract SHA-256. Check shape, float32 range, class mapping, finger visibility, clipping, padding and lighting. The baseline pipeline defaults to augmentation disabled. Horizontal flip is displayed as a trial. If you change the candidate dimensions or transform settings in `configs/data.json`, restart and rerun the notebook to review the new candidate.

**5. Approve and freeze the preprocessing contract.** Run this only when you accept the candidate and its training previews. Enter the contract SHA-256 printed by that notebook run.

```powershell
$contractHash = Read-Host 'Paste the reviewed preprocessing contract SHA-256'
& $py -m src.rps.phase4 approve-contract --config configs/data.json --expected-sha256 $contractHash
```

Success prints the contract hash and frozen record path. Under `contracts/<version>/` in the evidence directory, `frozen.json` binds the settings to the frozen split, preprocessing source hash and preview evidence. `approved-preview.json` preserves the reviewed training IDs and figure hashes.

**6. Build the Colab bundle locally.** This requires both freeze records and the original train and validation files. The command checks their hashes and packages the filtered assignments, corresponding PNGs, contract, source files, runtime requirements and a CPU reference tensor from one training image.

```powershell
& $py -m src.rps.phase4 stage --config configs/data.json
```

Expect the train and validation counts to match the split audit. The command prints the ZIP path and `archive_sha256`. Under `staging/<bundle-version>/` in the evidence directory it writes `phase4-stage.zip` for upload and `staging-audit.json` for the local record.

**7. Verify on a fresh hosted Colab runtime.** Use Python 3.13 with the matching versions of TensorFlow, Keras, NumPy and Pillow from `requirements.txt`. In the Colab Files panel, upload only `phase4-stage.zip` to `/content/`. Run this cell and paste the ZIP hash printed locally in step 6.

```python
import hashlib
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

archive = Path("/content/phase4-stage.zip")
expected = input("Local archive_sha256: ").strip()
with archive.open("rb") as stream:
    if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
        raise ValueError("Upload checksum mismatch")

directory = Path("/content/rps-phase4")
directory.mkdir(exist_ok=False)
with ZipFile(archive) as bundle:
    bundle.extractall(directory)
subprocess.run(
    [sys.executable, "-B", "-m", "src.rps.phase4", "verify-stage",
     "--root", str(directory), "--archive", str(archive), "--expected-sha256", expected],
    cwd=directory, check=True,
)
print((directory / "colab-verification.json").read_text())
```

The verifier checks compatibility, file membership, and hashes before tensor comparison. A successful report includes `"status": "passed"`, the split and contract hashes, the development sample ID, runtime, counts, tensor shape and dtype, and maximum absolute difference on the normalized scale (at most `1e-5`). Download `/content/rps-phase4/colab-verification.json` and save it next to the local staging audit. If execution stops, share the error and completed steps. For phase review, provide the Pytest summary, aggregate split audit, approved preview observations and contract hash, staging audit, and Colab result.

### Later phases

| Section | Details will be added when verified |
| --- | --- |
| Training | Phase 6 will document the Colab entry point, filtered inputs, run configuration, and export. |
| Evaluation | Phase 9 will add the final evaluation procedure and link the recorded results. |
| Inference | Phase 10 will add CLI and Gradio launch commands and supported input limits. |

Reproducibility checks will record the code revision, archive and split checksums, versions, resolved configuration, class mapping, and seeds. Pytest will check deterministic contracts, artifact reload, and inference behavior. Human review will verify curation decisions, visual evidence, controlled comparisons, and the pre-test approval for model-quality targets, latency budgets, and repeatability tolerances. [Test Plan](docs/05_TEST_PLAN.md)
