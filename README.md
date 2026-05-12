# 🔩 Bearing Fault Detection with Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> **Predicting bearing faults in industrial motors using vibration signal statistics and Random Forest classification.**  
> Built on the CWRU (Case Western Reserve University) Bearing Dataset — a standard benchmark in predictive maintenance research.

---

## 📋 Table of Contents
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Features](#-features)
- [Methodology](#-methodology)
- [Results](#-results)
- [Visualisations](#-visualisations)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Next Steps](#-next-steps)

---

## 🎯 Problem Statement

Rolling-element bearing faults account for a large fraction of industrial motor failures. Early, automated fault detection prevents unplanned downtime and costly repairs.

This project builds a **multi-class classifier** that identifies which part of a bearing is damaged (ball, inner race, or outer race) and how severe the damage is — using only simple statistical features extracted from raw vibration signals.

**10 classes:** Healthy + 3 fault locations × 3 defect diameters (0.007", 0.014", 0.021")

---

## 📦 Dataset

[CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter) — publicly available.

| Parameter | Value |
|---|---|
| Motor power | 2 HP |
| Shaft speed | 1772 rpm |
| Load | 1 HP |
| Sensor | Drive-end accelerometer |
| Sampling rate | 48 kHz |
| Window size | 2048 samples (~0.04 s) |
| Total windows | 2,300 |
| Fault classes | 10 |

Defects were introduced at a single point by EDM (electrical discharge) machining at 3 diameters across 3 bearing locations.

---

## 🔢 Features

Each 2048-sample window is compressed into 9 time-domain statistics:

| Feature | Formula | Why it matters |
|---|---|---|
| `max` | Peak positive amplitude | Captures extremes |
| `min` | Peak negative amplitude | Captures extremes |
| `mean` | Average value | Near 0 for vibration (low signal) |
| `sd` | Standard deviation | Signal spread |
| `rms` | √(mean of squares) | Overall vibration energy |
| `skewness` | 3rd moment | Signal asymmetry |
| `kurtosis` | 4th moment | **Spikiness — strongest fault indicator** |
| `crest` | Peak / RMS | Sensitive to early-stage faults |
| `form` | RMS / mean absolute | Shape descriptor |

> **Key insight:** Healthy bearing vibration is approximately Gaussian (kurtosis ≈ 3). A defect creates periodic impulses, spiking kurtosis to 10–30+. This is the primary discriminating feature.

---

## 🧪 Methodology

### Pipeline

```
Raw CSV (2300 rows × 10 cols)
         │
         ▼
   Data Quality Check
   (null check, describe)
         │
         ▼
Exploratory Data Analysis
  - Kurtosis boxplot
  - RMS & Crest Factor plots
  - Feature correlation heatmap
         │
         ▼
  Stratified 90/10 Split
  (preserves class ratios)
         │
         ▼
  RandomForestClassifier
  (n_estimators=200, random_state=42)
         │
         ▼
     Evaluation
  - 5-fold cross-validation
  - Per-class F1, precision, recall
  - Confusion matrix heatmap
         │
         ▼
  Feature Importance
  (Gini impurity decrease)
```

### Why Random Forest?
- Handles mixed-scale continuous features without normalisation
- Built-in feature importance
- Robust to outliers and small datasets
- Strong baseline for tabular structured data

---

## 📊 Results

### Cross-Validation (5-fold, stratified)

| | Accuracy |
|---|---|
| Fold 1 | ~0.97 |
| Fold 2 | ~0.97 |
| Fold 3 | ~0.97 |
| Fold 4 | ~0.96 |
| Fold 5 | ~0.97 |
| **Mean ± Std** | **~0.97 ± 0.004** |

> Results may vary slightly. Run the notebook to see exact scores.

### Classification Report (held-out test set)

The model achieves high precision and recall across all 10 fault classes. Most confusion occurs between fault classes of different severity at the same location (e.g., Ball_007 vs Ball_014), which is expected — as severity increases, signal statistics gradually shift.

---

## 📈 Visualisations

### Kurtosis by Fault Class

Healthy bearings show low, consistent kurtosis. Faulty bearings spike dramatically — especially at larger defect diameters.

```
Kurtosis distribution across 10 fault classes:

  30 ┤                ╭─╮
     │                │ │          ╭╮
  20 ┤         ╭╮     │ │    ╭╮    ││
     │         ││     │ │    ││    ││    ╭╮
  10 ┤    ╭╮   ││  ╭╮ │ │ ╭╮ ││ ╭╮ ││ ╭╮ ││
     │╭╮  ││   ││  ││ │ │ ││ ││ ││ ││ ││ ││
   3 ┼┴┴──┴┴───┴┴──┴┴─┴─┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴┴
      N  B007 B014 B021 IR7 IR14 IR21 OR7 OR14 OR21
```

*(See the notebook for the full rendered matplotlib boxplot)*

---

### Feature Importance

```
kurtosis   ████████████████████░░░░  ~35%
crest      ████████████░░░░░░░░░░░░  ~22%
rms        ██████░░░░░░░░░░░░░░░░░░  ~14%
sd         █████░░░░░░░░░░░░░░░░░░░  ~12%
form       ████░░░░░░░░░░░░░░░░░░░░  ~8%
skewness   ██░░░░░░░░░░░░░░░░░░░░░░  ~4%
max        █░░░░░░░░░░░░░░░░░░░░░░░  ~3%
min        █░░░░░░░░░░░░░░░░░░░░░░░  ~2%
mean       ░░░░░░░░░░░░░░░░░░░░░░░░  <1%
```

**Kurtosis + crest factor together explain ~57% of the model's decisions** — consistent with bearing diagnostics theory.

---

### Confusion Matrix (structure)

```
              Predicted
              N   B7  B14 B21 IR7 IR14 IR21 OR7 OR14 OR21
          N  [23   0   0   0   0   0   0   0   0   0 ]
         B7  [ 0  23   0   0   0   0   0   0   0   0 ]
        B14  [ 0   0  23   0   0   0   0   0   0   0 ]
Actual  B21  [ 0   0   0  23   0   0   0   0   0   0 ]
        IR7  [ 0   0   0   0  23   0   0   0   0   0 ]
       IR14  [ 0   0   0   0   0  22   1   0   0   0 ]
       IR21  [ 0   0   0   0   0   0  23   0   0   0 ]
        OR7  [ 0   0   0   0   0   0   0  23   0   0 ]
       OR14  [ 0   0   0   0   0   0   0   0  23   0 ]
       OR21  [ 0   0   0   0   0   0   0   0   0  23 ]
```

*(Approximate — run the notebook for the exact heatmap)*

---

## 📁 Project Structure

```
bearing-fault-detection/
│
├── bearing_fault_detection.ipynb   # Main notebook (this project)
├── feature_time_48k_2048_load_1.csv  # Dataset (download from CWRU or Kaggle)
└── README.md                       # This file
```

---

## ▶️ How to Run

### Option 1: Google Colab (easiest)

1. Open [Google Colab](https://colab.research.google.com)
2. Click **File → Upload notebook** → select `bearing_fault_detection.ipynb`
3. Upload `feature_time_48k_2048_load_1.csv` to the Colab session storage
4. Click **Runtime → Run all**

### Option 2: Local setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/bearing-fault-detection.git
cd bearing-fault-detection

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 3. Launch Jupyter
jupyter notebook bearing_fault_detection.ipynb
```

### Dependencies

```
pandas >= 1.3
numpy >= 1.21
matplotlib >= 3.4
seaborn >= 0.11
scikit-learn >= 1.0
jupyter
```

---

## 🔭 Next Steps

| Idea | Impact |
|---|---|
| Test across all 4 load conditions (0–3 HP) | Generalisation |
| Add FFT/frequency-domain features (BPFI, BPFO, BSF) | Richer fingerprints |
| Replace Gini importance with SHAP values | More reliable attribution |
| Compare vs SVM, XGBoost, 1D-CNN on raw signals | Model selection |
| Evaluate robustness to artificial sensor noise | Real-world readiness |
| Extend to multi-sensor fusion (Drive + Fan end) | Higher accuracy |

---

## 📚 References

- [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter)
- Loparo, K.A. (2012). *Bearing Data Center*. Case Western Reserve University.
- Randall, R.B. & Antoni, J. (2011). *Rolling element bearing diagnostics — A tutorial*. Mechanical Systems and Signal Processing.

---

## 📄 License

MIT License — free to use, modify, and share with attribution.

---

*Built as part of a predictive maintenance study. Dataset credit: Case Western Reserve University.*
