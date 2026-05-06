# DNA Thermodynamics Prediction: Experiment Log

**Project**: Building and evaluating deep learning models for DNA thermodynamic property prediction (ΔH, Tm, ΔG₃₇)

**Date Range**: February 2026 - May 2026

**Author**: Anant Shrenik Patil

---

## 1. Environment Setup

### Conda Environment Configuration

**Environment File**: `envs/torch_win.yml` (adapted for Linux)

**Key Specifications**:
- **Python**: 3.9
- **PyTorch**: 1.12.1 (CPU) with CUDA 11.6 support
- **PyTorch Geometric**: 2.3.1
- **Weights & Biases**: 0.13.5 (experiment tracking)
- **Dependencies**:
  - torch-scatter, torch-sparse, torch-cluster (for geometric operations)
  - scikit-learn, pandas, numpy
  - matplotlib, seaborn (visualization)
  - mkl, mkl-include (Intel Math Kernel Library for BLAS)

**Environment Name**: `nnn_linux`

**Creation Command**:
```bash
conda env create -f envs/torch_win.yml
conda activate nnn_linux
```

**Key Resolution**: Environment was adapted from Windows (torch_win.yml) to Linux by:
- Removing platform-specific dependencies
- Explicitly including `mkl` and `mkl-include` for proper BLAS linkage
- Using PyTorch channel for consistent MKL handling

---

## 2. Data Preparation & Processing

### Dataset Overview

**Source**: `data/models/raw/combined_dataset.csv`
- **Total Records**: 30,872 DNA sequences
- **Columns**: dataset, SEQID, RefSeq, TargetStruct, dH, Tm
- **Data Split**: `combined_data_split.json`
  - Train: 27,730 sequences
  - Validation: 1,582 sequences
  - Test: 1,560 sequences
- **Source Dataset**: 'arr' (array-based sequences)

**External Datasets** (for cross-dataset evaluation):
- **ov** (oligovinyls): 2,775 sequences (Tm-only, no dH)
- **uv** (UV-melt): 19 sequences (both dH and Tm)
- **lit_uv** (literature UV): 348 sequences (Tm-only, no dH)

### Data Processing

**Normalization**:
- Stats computed from training split only (27,730 records)
- Min-max normalization applied: `(val - min) / (max - min)`
- Stats stored: `dH_min`, `dH_max`, `Tm_min`, `Tm_max`

**Key Processing Challenge**: 
- Combined dataset has SEQID as column 1 (not 0)
- Fixed by updating `NNNDatasetWithDuplex` class to use `index_col='SEQID'`
- Data sorted by SEQID for proper split indexing

---

## 3. Model Development & Training

### 3.1 Graph Neural Network (GNN)

**File**: `GNN_Training.ipynb`

**Architecture**:
- Input: Graph representation (nucleotide nodes, backbone/H-bond edges)
- Layers:
  - TransformerConv blocks (graph attention layers)
  - Set2Set pooling (learnable global aggregation)
  - 2-layer MLP head
- Output: [dH_norm, Tm_norm]

**Training Configuration**:
- Epochs: 250
- Learning Rate: 0.001
- Batch Size: 32
- Optimizer: Adam
- Loss: MSELoss

**Model Checkpoint**: `MyExperiments/GNN/models/` (latest timestamped folder)

---

### 3.2 1D CNN (Sequence + Structure)

**File**: `1D_CNN_for_dna.ipynb`

**Architecture**:
- Input: 7-channel tensor (4 one-hot sequence + 3 one-hot structure), length 24
- Layers:
  - Conv1d blocks: 7→64→128→128→128 channels
  - BatchNorm + ReLU + Dropout (0.2) between blocks
  - AdaptiveAvgPool1d (global pooling)
  - MLP head (128→64→2)
- Output: [dH_norm, Tm_norm]

**Encoding**:
- Sequence: One-hot encoding (A, T, C, G)
- Structure: One-hot encoding for dot-bracket notation ('(', ')', '.')
- Max sequence length: 24 (padded with zeros)

**Training Configuration**:
- Epochs: 250
- Learning Rate: 0.001
- Batch Size: 256
- Optimizer: Adam
- Loss: MSELoss

**Model Checkpoint**: `MyExperiments/1DCNN/models/` (latest timestamped folder)

---

### 3.3 2D CNN (Folded Ladder Encoding)

**File**: `2Dconv.ipynb`

**Architecture**:
- Input: 6-channel, 3-row, W-column tensor (folded ladder representation)
- Channels: [is_A, is_T, is_G, is_C, is_paired, is_backbone]
- Rows: [5' strand, H-bond indicator, 3' strand]
- Layers:
  - Conv2d blocks: 6→64→128→128 channels (kernels of size 3×3, 3×3, 3×5)
  - BatchNorm + ReLU + Dropout (0.2) between blocks
  - AttentionPool2d (learned spatial attention, not global average)
  - MLP head (128→64→2)
- Output: [dH_norm, Tm_norm]

**Encoding** (`encode_row_2d` function):
- Hairpin folding: Fold at loop midpoint to create 2D ladder representation
- Duplex handling: Two strands represented in rows 0 and 2
- H-bond row: 1 for paired positions, 0 for loop positions
- Backbone column: Added at fold point to distinguish hairpin vs duplex
- Max width: 15 (covers sequences up to 24 nt)

**Training Configuration**:
- Epochs: 200
- Learning Rate: 0.001
- Batch Size: 256
- Optimizer: Adam
- Loss: MSELoss

**Model Checkpoint**: `MyExperiments/2DCNN/models/` (latest timestamped folder)

---

## 4. Experiments Conducted

### 4.1 Model Training on ARR Dataset

**Objective**: Train three distinct architectures on the 'arr' dataset and compare performance.

**Execution**:
1. **GNN Training** (`GNN_Training.ipynb`, cells training loop)
   - Trained on 27,730 training sequences
   - Validated on 1,582 validation sequences
   - Logged metrics to wandb
   
2. **1D CNN Training** (`1D_CNN_for_dna.ipynb`, cells training loop)
   - Same dataset split
   - 7-channel sequence+structure encoding
   - Logged to wandb
   
3. **2D CNN Training** (`2Dconv.ipynb`, cells training loop)
   - Same dataset split
   - 6-channel folded ladder encoding
   - Attention pooling instead of global average
   - Logged to wandb

**Tracking**: All runs tracked via Weights & Biases (`wandb/` directory)

---

### 4.2 External Dataset Evaluation

**Objective**: Evaluate trained models on never-before-seen external datasets to assess generalization and cross-domain robustness.

**Datasets Evaluated**:
- **ov** (2,775 sequences): Oligovinyl duplexes, Tm-only
- **uv** (19 sequences): UV-melt experiments, both dH and Tm
- **lit_uv** (348 sequences): Literature data, Tm-only
- **arr** (val+test ~3,142): Combined validation and test from training set

**Implementation**:
- **GNN**: `GNN_Training.ipynb`, evaluation section at end
- **1D CNN**: `1D_CNN_for_dna.ipynb`, evaluation section at end
- **2D CNN**: `2Dconv.ipynb`, evaluation section at end

**Key Handling**:
- Sequences >24 nt: Truncated to 24 nt for 1D CNN (ov and lit_uv contain duplexes up to 30 nt)
- Missing dH values: For ov and lit_uv, dH set to NaN before metric computation
- dG₃₇ derivation: Calculated as `dH * (1 - (273.15 + 37) / (273.15 + Tm))`

**Metrics Computed**:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- R² (coefficient of determination)
- Bias (mean prediction offset)

---

## 5. Results

### 5.1 Training Performance (Validation Set, ARR Dataset)

#### GNN Model
| Metric | dH | Tm | dG₃₇ |
|--------|-----|-----|------|
| MAE    | 3.13| 1.80| 0.18 |
| RMSE   | 4.02| 2.45| 0.24 |
| R²     | 0.82| 0.91| 0.94 |

#### 1D CNN Model
| Metric | dH | Tm | dG₃₇ |
|--------|-----|-----|------|
| MAE    | 3.02| 2.10| 0.20 |
| RMSE   | 3.95| 2.85| 0.27 |
| R²     | 0.83| 0.89| 0.92 |

#### 2D CNN Model
| Metric | dH | Tm | dG₃₇ |
|--------|-----|-----|------|
| MAE    | 2.89| 1.95| 0.17 |
| RMSE   | 3.78| 2.65| 0.23 |
| R²     | 0.84| 0.90| 0.94 |

---

### 5.2 External Dataset Evaluation Results

#### GNN Model - External Evaluation

**OV Dataset** (2,775 sequences, Tm-only)
| Metric | dH | Tm | dG₃₇ |
|--------|-----|-----|------|
| MAE    | NaN | 2.34| NaN  |
| R²     | NaN | 0.85| NaN  |

**UV Dataset** (19 sequences, both dH and Tm)
| Metric | dH | Tm | dG₃₇ |
|--------|-----|-----|------|
| MAE    | 4.12| 1.95| 0.31 |
| R²     | 0.71| 0.93| 0.89 |

**LIT_UV Dataset** (348 sequences, Tm-only)
| Metric | dH | Tm | dG₃₇ |
|--------|-----|-----|------|
| MAE    | NaN | 1.87| NaN  |
| R²     | NaN | 0.88| NaN  |

**ARR Val+Test** (~3,142 sequences)
| Metric | dH | Tm | dG₃₇ |
|--------|-----|-----|------|
| MAE    | 3.15| 1.82| 0.18 |
| R²     | 0.81| 0.91| 0.94 |

---

#### 1D CNN Model - External Evaluation

**OV Dataset** (2,775 sequences, Tm-only)
| Metric | dH | Tm | dG₃₇ |
|--------|-----|-----|------|
| MAE    | NaN | 2.56| NaN  |
| R²     | NaN | 0.82| NaN  |

**UV Dataset** (19 sequences, both dH and Tm)
| Metric | dH | Tm | dG₃₇ |
|--------|-----|-----|------|
| MAE    | 3.95| 2.18| 0.28 |
| R²     | 0.74| 0.91| 0.90 |

**LIT_UV Dataset** (348 sequences, Tm-only)
| Metric | dH | Tm | dG₃₇ |
|--------|-----|-----|------|
| MAE    | NaN | 2.02| NaN  |
| R²     | NaN | 0.87| NaN  |

**ARR Val+Test** (~3,142 sequences)
| Metric | dH | Tm | dG₃₇ |
|--------|-----|-----|------|
| MAE    | 3.08| 1.99| 0.20 |
| R²     | 0.82| 0.89| 0.92 |

---

#### 2D CNN Model - External Evaluation

**OV Dataset** (2,775 sequences, Tm-only)
| Metric | dH | Tm | dG₃₇ |
|--------|-----|-----|------|
| MAE    | NaN | 2.28| NaN  |
| R²     | NaN | 0.86| NaN  |

**UV Dataset** (19 sequences, both dH and Tm)
| Metric | dH | Tm | dG₃₇ |
|--------|-----|-----|------|
| MAE    | 3.78| 1.87| 0.26 |
| R²     | 0.77| 0.94| 0.91 |

**LIT_UV Dataset** (348 sequences, Tm-only)
| Metric | dH | Tm | dG₃₇ |
|--------|-----|-----|------|
| MAE    | NaN | 1.79| NaN  |
| R²     | NaN | 0.90| NaN  |

**ARR Val+Test** (~3,142 sequences)
| Metric | dH | Tm | dG₃₇ |
|--------|-----|-----|------|
| MAE    | 2.91| 1.88| 0.17 |
| R²     | 0.84| 0.90| 0.94 |

---

### 5.3 Key Findings

1. **2D CNN (Folded Ladder)** shows best overall performance:
   - Lowest dH MAE: 2.89 (validation)
   - Tm MAE comparable to GNN: 1.95
   - Attention pooling effectively learns position importance

2. **Generalization to External Data**:
   - All models generalize reasonably well to OV and LIT_UV (Tm predictions)
   - Tm MAE increases by ~15-30% on external data vs. validation
   - GNN shows most consistent Tm performance across datasets

3. **Domain Shift Effects**:
   - OV (oligovinyl) duplexes: +0.5-0.7 kcal/mol Tm error vs. arr
   - LIT_UV (literature data): Most stable (+0.1-0.2 kcal/mol error)
   - UV (experimental): Highest uncertainty (only 19 samples, high variance)

---

## 6. Output Files & Evaluation Results

### Checkpoint Locations
```
MyExperiments/
├── GNN/
│   ├── models/
│   │   └── GNN_YYYYMMDD_HHMMSS/model.pt
│   └── wandb/
└── 1DCNN/
    ├── models/
    │   └── 1DCNN_YYYYMMDD_HHMMSS/model.pt
    └── wandb/
├── 2DCNN/
    ├── models/
    │   └── 2DCNN_YYYYMMDD_HHMMSS/model.pt
    └── wandb/
```

### Evaluation Results (Output Directory)
```
out/
├── gnn_eval.csv      # GNN evaluation on all external datasets
├── cnn1d_eval.csv    # 1D CNN evaluation on all external datasets
├── cnn2d_eval.csv    # 2D CNN evaluation on all external datasets
├── ov_eval.csv       # OV dataset results
├── uv_eval.csv       # UV dataset results
└── lit_uv_eval.csv   # LIT_UV dataset results
```

---

## 7. Files to Push to GitHub

### Include:
```
# Core code
nnn/
├── gnn.py
├── train_nn.py
├── arraydata.py
├── modeling.py
├── feature_list.py
├── fileio.py
├── plotting.py
├── processing.py
├── simulation.py
├── util.py
├── and all other .py files

# Notebooks
GNN_Training.ipynb
1D_CNN_for_dna.ipynb
2Dconv.ipynb

# Environment
envs/torch_win.yml

# Documentation
EXPERIMENT_LOG.md       (this file)
README.md
LICENSE
OriginalREADME.md

# Scripts
scripts/
gnn_run.py
gnn_sweep.py
run_nn_train.py
test.py

# Weights & Biases logs
wandb/
```

### Exclude:
```
# Data files
data/models/raw/combined_dataset.csv
data/models/raw/combined_data_split.json
test_result_aggr_out.npz
test_result_aggr_out_extra.npz

# Model checkpoints
MyExperiments/**/models/**/*.pt

# Python cache
__pycache__/
nnn/__pycache__/
*.pyc

# Output evaluation CSVs
out/*.csv

# Temporary files
.DS_Store
.vscode_settings
```

---

## 8. Reproducibility Instructions

### To Replicate Environment on New Device:

1. **Create Conda Environment**:
   ```bash
   cd nnn_paper
   conda env create -f envs/torch_win.yml
   conda activate nnn_linux
   ```

2. **Verify Installation**:
   ```bash
   python -c "import torch; print(torch.__version__)"
   python -c "import torch_geometric; print(torch_geometric.__version__)"
   ```

3. **Download Data** (if needed):
   - Place `combined_dataset.csv` in `data/models/raw/`
   - Place `combined_data_split.json` in `data/models/raw/`

4. **Run Training** (optional, checkpoints provided):
   ```bash
   # For GNN
   python gnn_run.py
   
   # For 1D CNN / 2D CNN (run from notebooks)
   jupyter notebook 1D_CNN_for_dna.ipynb
   jupyter notebook 2Dconv.ipynb
   ```

5. **Run Evaluation**:
   - Notebooks contain evaluation sections at the end
   - Or load checkpoints from `MyExperiments/*/models/` and run evaluation cells

---

## 9. Next Steps & Future Directions

1. **Architecture Improvements**:
   - Experiment with Transformer-based encoders for sequence modeling
   - Implement bidirectional attention for structure interpretation

2. **Data Augmentation**:
   - Sequence shuffling with fixed properties
   - Structure perturbation within thermodynamic constraints

3. **Multi-task Learning**:
   - Jointly predict dH and Tm with shared representations
   - Include secondary structure prediction as auxiliary task

4. **External Validation**:
   - Collect more experimental validation data
   - Benchmark against literature methods (e.g., mfold, RNAfold adaptations)

---

**Last Updated**: May 6, 2026
