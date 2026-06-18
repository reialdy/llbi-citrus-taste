# Design and Implementation of Laser-Light Backscattering Imaging System as a Non-Destructive Technique for Citrus Taste Evaluation

This repository contains the source code, hardware configuration details, and deep learning pipeline for an automated, non-destructive citrus taste evaluation system using **Laser-Light Backscattering Imaging (LLBI)** and deep learning. 

---

## 📌 Project Overview

Conventional citrus taste assessments (sensory panels and chemical analysis) are time-consuming and destructive. This project introduces a rapid, objective, and non-destructive alternative. 

By capturing the backscattering profiles of Siamese citrus fruits illuminated by specific laser wavelengths, we extract subsurface structural data correlated with fruit quality. A **ResNet50-based deep learning classifier** is then deployed to categorize the fruit into **Sweet** or **Sour** profiles based on sensory baseline data established by trained panelists.

### Key Features
* **Multi-Wavelength LLBI System:** Image acquisition using three distinct laser diodes: **450 nm**, **532 nm**, and **648 nm**.
* **Deep Learning Pipeline:** End-to-end ResNet50 architecture implemented for binary taste classification.
* **High Accuracy Deployment:** Optimized for the 648 nm wavelength, which achieved peak experimental performance.

---

## 📊 Experimental Results

Our study evaluated 150 Siamese citrus samples from Garut, West Java Indonesia Orchards. While all wavelengths were tested, the **648 nm (Red) wavelength** demonstrated exceptional predictive capability:

| Phase | Accuracy | AUC |
| :--- | :---: | :---: |
| **Training** | 98.968% | 0.9996 |
| **Validation** | 96.898% | 0.9967 |
| **Testing** | 96.759% | 0.9961 |

---

## 📂 Repository Structure

Once the codebase is fully uploaded, the repository will be organized as follows:

```text
├── hardware/               # Schematic diagrams, laser control scripts, and setup specs
├── src/
│   ├── preprocessing/      # Image cropping, normalization, and ROI extraction
│   ├── model/              # ResNet50 architecture setup and custom training loops
│   └── utils/              # Helper functions (logging, data loading, metrics evaluation)
├── notebooks/              # Jupyter notebooks for exploratory data analysis (EDA) and plotting ROC curves
├── requirements.txt        # Python dependencies (TensorFlow/PyTorch, OpenCV, NumPy, etc.)
└── README.md
