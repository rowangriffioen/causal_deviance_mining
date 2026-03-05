# **Causal Deviance Mining (CDM) – Thesis Implementation**

This repository contains the full implementation developed for the master’s thesis:

> **Combining Multi-Perspective Business Process Deviance Mining with Causal Rule Mining: Toward Holistic and Actionable Process Improvement**  
> *R.J. Griffioen, Eindhoven University of Technology, 2025*

The project integrates **Causal Rule Mining (CRM)** into a **multi-perspective Business Process Deviance Mining (DM)** pipeline to discover interpretable, causally grounded explanations of process deviations.  
It builds upon and extends the deviance mining framework of Bergami et al. (2021), introducing causal reasoning (Eshuis & Genga, 2025), enhanced feature extraction via **IMPresseD**, and sampling stability through **Inverse Propensity Weighting (IPW)**.

📦 **Public Datasets**  
The event logs and feature datasets used in this research are openly available on Zenodo:  
👉 [https://zenodo.org/records/18864747](https://zenodo.org/records/18864747)

---

## 📘 **Overview**

The repository implements all components of the causal DM pipeline as described and evaluated in the thesis:

| Phase | Focus | Main Component |
|:------|:------|:----------------|
| **1** | Engineering causal DM (CRM integration) | `3.dm_random_k*.py`, `3.dm_dt_ripperk.py` |
| **2** | Improving feature extraction (IMPresseD integration) | `1.feature_extraction.ipynb` |
| **3** | Exploring improvements (IPW variant) | `3.dm_ipweights_k*.py` |

The code supports both **public event logs** (Traffic, BPI15A, Sepsis) and **real-world data** (DHL case study).

---

## ⚙️ **Pipeline Overview**

The full workflow consists of sequential stages that mirror the research phases:

### **1. Feature Extraction**
Run `1.feature_extraction.ipynb`  
- Input: labeled `.xes` logs in `0_raw_log`  
- Output folders:  
  - `1_unique_log/` – deduplicated event logs  
  - `2_labelled_logs/` – labeled logs in `.csv`  
  - `3_extracted_features/` – extracted multi-perspective features  
- Integrates the **IMPresseD** framework for multi-interest pattern discovery.

---

### **2. Feature Selection**
Run `2.feature_selection.ipynb`  
- Takes extracted features from `3_extracted_features/`  
- Applies a Fisher-score–based coverage threshold (τ = 10 / 20)  
- Outputs selected features in `3.1_selected_features/`

---

### **3. Feature Binning**
Run `2.1.binning.ipynb`  
- Inputs: `3.1_selected_features/`  
- Performs **label-aware binning** on continuous attributes  
- Outputs binned datasets in `3.2_binned_features/`

---

### **4. Causal Deviance Mining**
Choose one of the causal or baseline model scripts:

| Script | Description |
|:--------|:-------------|
| `3.dm_random_k1.py`, `3.dm_random_k2.py`, `3.dm_random_k3.py` | Causal DM (CRM) with random subsampling (different rule depths) |
| `3.dm_ipweights_k1.py`, `3.dm_ipweights_k2.py`, `3.dm_ipweights_k3.py` | Causal DM (CRM) using inverse propensity weighting (IPW) |
| `3.dm_dt_ripperk.py` | Baseline supervised DM (Decision Tree / RipperK) |

- Input: binned feature sets from `3.2_binned_features/`  
- Output: discovered rules and statistics in `4_output/`

---

### **5. Analysis & Visualization**
Use the notebooks in `analysis_utils/`  
- Performs redundancy analysis, rule similarity checks, runtime evaluation, and visualization  
- Outputs summary tables and figures in `5_analysis/`  
- Results can be directly used in LaTeX tables/figures (as in the thesis appendices)

---

## 🧩 **Repository Structure**

```
├── 0_raw_log/                # Input event logs (XES)
├── 1_unique_log/             # Deduplicated logs
├── 2_labelled_logs/          # CSV-converted labeled logs
├── 3_extracted_features/     # Extracted features (multi-perspective)
├── 3.1_selected_features/    # After feature selection
├── 3.2_binned_features/      # After binning
├── 4_output/                 # Output of DM experiments
├── 5_analysis/               # Analysis outputs and plots
│
├── analysis_utils/           # Notebooks for analysis
├── config/                   # Configuration files for each pipeline stage
├── helper_functions/         # Custom utility scripts (shared across pipeline)
├── hpc_utils/                # Slurm scripts for HPC execution
├── IMPresseD/                # IMPresseD feature extraction implementation
├── models/                   # CRM model variants
├── wittgenstein/             # Supervised RipperK model helper files
│
├── 1.feature_extraction.ipynb
├── 2.feature_selection.ipynb
├── 2.1.binning.ipynb
├── 3.dm_random_k*.py
├── 3.dm_ipweights_k*.py
├── 3.dm_dt_ripperk.py
│
└── README.md
```

---

## 💻 **Execution Environment**

- Python **3.8+** (tested on **3.11.0**)  
- Anaconda environment recommended  
- Core dependencies:
  - `pandas`, `numpy`, `pm4py`, `yaml`, `scikit-learn`
  - `IMPresseD`, `wittgenstein`, `matplotlib`, `seaborn`

---

## 🚀 **Running on HPC**
Scripts in `hpc_utils/` provide **SLURM** job templates for running the pipeline on TU/e’s Umbrella HPC cluster.  
Each stage (feature extraction, selection, DM) can be scheduled as a separate job.

---

## 📊 **Reproducing Thesis Results**

This repository directly supports the experiments and figures in the thesis.  
Below is a mapping between the main thesis chapters/appendices and their corresponding scripts or notebooks:

| Thesis Section | Pipeline Stage | Relevant Script / Notebook | Output Folder |
|:----------------|:---------------|:----------------------------|:----------------|
| Chapter 4 – Design & Development | CRM Integration | `3.dm_random_k*.py` | `4_output/random/` |
| Chapter 5 – Evaluation Phase 1 | Random CRM Results | `3.dm_random_k1.py` – `3.dm_random_k3.py` | `4_output/random/` |
| Chapter 5 – Evaluation Phase 2 | IMPresseD Features | `1.feature_extraction.ipynb` | `3_extracted_features/` |
| Chapter 5 – Evaluation Phase 3 | IPW Variant | `3.dm_ipweights_k*.py` | `4_output/ipweights/` |
| Appendix F–G | Feature Selection / Binning | `2.feature_selection.ipynb`, `2.1.binning.ipynb` | `3.1_selected_features/`, `3.2_binned_features/` |
| Appendix I–N | Analysis & Rule Tables | `analysis_utils/` notebooks | `5_analysis/` |
| DHL Case Study | Full Pipeline (End-to-End) | all notebooks + IPW CRM | `4_output/ipweights/DHL/` |

---

## 🧠 **Reference**

If you use this code or its results, please cite:

> Griffioen, R. J. (2025). *Combining Multi-Perspective Business Process Deviance Mining with Causal Rule Mining: Toward Holistic and Actionable Process Improvement*.  
> Master Thesis, Eindhoven University of Technology.

---

## 🏗️ **Acknowledgments**

Developed in collaboration with:
- **Eindhoven University of Technology – Information Systems Group**  
- **Pipple B.V.** – Data Science Consultancy  
- **DHL Supply Chain** – Case Study Partner  

Supervisors: dr.ir. H. Eshuis, dr. L. Genga, J. Scheepers (Pipple), S. van den Bogaart (Pipple)
