# GNN for DNA Thermodynamic Prediction

> **Author:** Anant Patil  
> For the original paper's README, see [OriginalREADME.md](OriginalREADME.md).

## Overview

This experiment trains a **Graph Neural Network (GNN)** to predict DNA thermodynamic properties — specifically **enthalpy (dH)** and **melting temperature (Tm)** — from short  DNA hairpin and duplex sequences.

The work builds on the paper *"High-Throughput DNA melt measurements enable improved models of DNA folding thermodynamics"* and uses the same dataset of ~30,000 sequences with experimentally measured thermodynamic values.

### How it works

Each DNA sequence is represented as a **graph**:
- **Nodes** — one per nucleotide, encoded as a 4-dimensional one-hot vector (A, T, C, G).
- **Edges** — three types: 5'→3' backbone, 3'→5' backbone, and hydrogen bonds (from dot-bracket structure).

The GNN architecture uses **TransformerConv** layers (4 layers, 125 hidden channels), **Set2Set** pooling, and an MLP head to predict normalized dH and Tm.

## File Structure

| File | Description |
|------|-------------|
| `GNN_for_dna.ipynb` | Main notebook — data loading, graph construction, model training, and evaluation |
| `gnn_run.py` | Script-based entry point for training/validation (from original paper) |
| `nnn/gnn.py` | Original GNN model definition and dataset classes |
| `data/models/raw/combined_dataset.csv` | Full dataset (30,872 sequences) |
| `data/models/raw/combined_data_split.json` | Train/val/test split indices |

## Environment Setup

### Prerequisites
- Conda 
- GPU (better if availabel)

### Installation

**Linux:**
Use File: `envs/torch_full.yml`

**Cross-platform (portable, no CUDA pinning):**
Use File: `envs/torch_portable.yml`

### Key dependencies
- Python 3.9
- PyTorch 1.12 + CUDA 11.6
- torch-geometric 2.3.1





