# IEEE Conference Paper: Bayesian Deep Knowledge Tracing

This repository contains a research paper and implementation for an advanced machine learning approach to student knowledge assessment using Bayesian Deep Knowledge Tracing (BDKT).

## 📄 Contents

- **root.tex** - Main LaTeX source file for the IEEE conference paper
- **root.pdf** - Compiled PDF version of the paper
- **ieeeconf.cls** - IEEE conference LaTeX class file
- **technique/** - Implementation and experimental code
  - `bdkt_model.py` - BDKT model implementation
  - `train.py` - Training script
  - `eval.py` - Evaluation script
  - `etl.py` - Data processing
  - `plots.py` - Visualization utilities
  - `requirements.txt` - Python dependencies

## 📊 Data

- **skills_metadata.csv** - Metadata about educational skills
- **synthetic_bdkt_dataset.csv** - Synthetic dataset for model training
- **synthetic_bdkt_stats.json** - Dataset statistics

## 🔍 Overview

This project implements a Bayesian Deep Knowledge Tracing model to predict student learning outcomes and mastery levels. The model combines deep learning with Bayesian inference to provide uncertainty estimates in knowledge predictions.

## 📋 Documentation

See `explication.md` for detailed technical explanation and methodology.

## 🚀 Getting Started

1. Install dependencies:
   ```bash
   pip install -r technique/requirements.txt
   ```

2. Run the model:
   ```bash
   cd technique
   python train.py
   python eval.py
   ```

## 📝 Citation

If you use this work, please cite the paper (see root.pdf for full details).

---

**Author**: Mandasoa Narovanjanahary  
**Language**: English
