# Narval environment setup

**Cluster:** Narval (Digital Research Alliance of Canada, Montréal)  
**Arch:** x86-64-v3 (AVX512), A100 GPUs  
**This file is Narval-specific.** Trillium (SciNet) has different module names,
different scratch layout, and different GPU hardware. Do not mix them.

---

## SLURM accounts

```
def-senger_cpu   # CPU jobs
def-senger_gpu   # GPU jobs
```

## GPU resource name (verified 2026-09-09)

Full A100 nodes expose `gpu:a100:4` (four GPUs per node).  
Request a single GPU with `--gres=gpu:a100:1`.

---

## Step 1 / Step 3 venv (CPU, no MED3pa)

```bash
module load python/3.11 scipy-stack
virtualenv --no-download ~/wsi-survival-interpretability/venv
source ~/wsi-survival-interpretability/venv/bin/activate
pip install --no-index numpy pandas scikit-learn scipy openpyxl
```

Tested versions (2026-09-09):
- numpy 2.4.2+computecanada
- pandas 3.0.5+computecanada
- scikit-learn 1.8.0+computecanada
- scipy 1.17.1+computecanada
- openpyxl 3.1.5+computecanada

**numpy 2.x note:** `ndarray.ptp()` was removed in numpy 2.0.
The new analysis scripts use `np.ptp(arr)`, which works on both 1.x and 2.x.
If you touch any other script, watch for `.ptp()` calls.

Activate for subsequent use:
```bash
module load python/3.11 scipy-stack
source ~/wsi-survival-interpretability/venv/bin/activate
```

---

## Step 2 venv (MED3pa, login node required)

MED3pa is NOT in the Alliance wheelhouse. Install on the login node (which has
internet access) before submitting any Step 2 jobs.

```bash
module load python/3.11 scipy-stack
source ~/wsi-survival-interpretability/venv/bin/activate
# Install wheelhouse packages first, then MED3pa without dependency resolution
pip install --no-index numpy pandas scikit-learn scipy openpyxl joblib matplotlib
pip install med3pa==1.0.4 --no-deps
```

MED3pa's remaining dependencies (torch, ray, xgboost, tqdm, PyYAML, relib,
checkpointer) are NOT in the wheelhouse. Install from PyPI on the login node:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install xgboost ray tqdm pyyaml relib checkpointer
```

Exact versions are TBD until Step 2 begins; pin them in this file after install.

---

## Data paths on Narval (as of 2026-09-09)

### Present on Narval scratch
- Raw slides + embeddings: `/scratch/sorkwos/TCGA-BLCA*`, `/scratch/sorkwos/TCGA-BRCA*`, etc.
- CDR spreadsheet: `/home/sorkwos/TCGA-CDR-SupplementalTableS1.xlsx`

### NOT on Narval — must transfer from Trillium via Globus

Step 1 critical (tiny, transfer first):
```
/scratch/sorkwos/med3pa_bootstrap/final_med3pa_input.npz               # BLCA
/scratch/sorkwos/med3pa_bootstrap_intermediate_BRCA/final_med3pa_input.npz  # BRCA
```

Step 3 (per-round npz files, still small):
```
/scratch/sorkwos/med3pa_bootstrap_intermediate/job_*/round_*/epoch_*.npz    # BLCA 102-round
/scratch/sorkwos/med3pa_bootstrap_intermediate_BRCA/job_*/round_*/epoch_*.npz  # BRCA
```

Step 2 / 5 (large, needs GPU — do not start until Steps 1–3 are done):
```
/scratch/sorkwos/med3pa_bootstrap/                    # BLCA 25-round (Step 2 baseline)
/scratch/sorkwos/med3pa_bootstrap_intermediate/       # BLCA 102-round (primary)
/scratch/sorkwos/med3pa_bootstrap_intermediate_BRCA/  # BRCA
clinical and pathomic feature CSVs (exact paths in med3pa driver scripts)
model checkpoints (exact paths in SLURM submit scripts)
```

**Runbook path note:** the runbook references
`/home/sorkwos/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81/TCGA-CDR-SupplementalTableS1.xlsx`
but that UUID subdirectory does not exist on Narval. Use
`/home/sorkwos/TCGA-CDR-SupplementalTableS1.xlsx` instead.

---

## SLURM job templates for Narval

### CPU job (Step 1 / 3)
```bash
sbatch --account=def-senger_cpu \
       --time=2:00:00 \
       --cpus-per-task=4 \
       --mem=16G \
       slurm/narval/step1_patient_error.sh
```

### GPU job (Step 5)
```bash
sbatch --account=def-senger_gpu \
       --gres=gpu:a100:1 \
       --time=8:00:00 \
       --cpus-per-task=6 \
       --mem=48G \
       slurm/narval/step5_region_ablation.sh
```

Use `salloc` with the same flags for short interactive tests.
Never run compute on the login node.
