# ECG5000 Time-Series Anomaly Detection

A reproducible study of reconstruction-based anomaly detection on physiological time-series data. This project evaluates deep learning architectures (Conv1D and LSTM Autoencoders) against classical machine learning baselines to accurately identify abnormal heartbeats in the UCR ECG5000 dataset.

## Overview
This project frames anomaly detection as a semi-supervised learning problem. By training autoencoders exclusively on normal heartbeats, the models learn to accurately reconstruct typical physiological patterns. Anomalies are then identified using the Mean Squared Error (MSE) between the original and reconstructed signals. The study systematically compares threshold-selection strategies, contrasts convolutional and recurrent architectures, and explores the "reconstruction paradox" where excess model capacity degrades detection performance.

## Project Structure
- `ECG5000 Anomaly Detection.ipynb`: Jupyter notebook containing the complete data processing, autoencoder construction, threshold optimization, and evaluation pipeline.
- `Report - Reconstruction-Based Anomaly Detection on ECG5000.pdf`: The comprehensive academic report detailing the methodology, capacity ablation study, and failure-case analysis.

## Key Phases
1. **Data Preprocessing (Leakage-Free):** Sourced the official 500/4500 train/test split. Applied a global z-score standardization fitted *strictly* on the 234 normal training rows to prevent data leakage into the validation and test sets.
2. **Model Implementation:** Engineered a Conv1D Autoencoder (symmetric max-pool downsampling and linear upsampling) and an LSTM Autoencoder operating on the temporal dimension. 
3. **Threshold Selection:** Systematically evaluated four thresholding strategies on a dedicated validation set (Percentile, Mean+k·std, Median+k·MAD, and Validation-F1 argmax). The Validation-F1 argmax strategy was frozen and applied to the test set.
4. **Capacity Ablation & Error Analysis:** Conducted a latent-dimension capacity ablation (ranging from 4 to 64) demonstrating that intermediate capacities isolate anomalies best. Error analysis revealed that False Negatives were heavily concentrated in Class 4 (ectopic beats) due to their morphological similarity to normal heartbeats.

## Key Technologies
- **Python:** Primary analytical programming language.
- **PyTorch:** Framework used to design, train, and optimize the Conv1D and LSTM autoencoders.
- **Scikit-Learn:** Implementation of classical baselines (One-Class SVM, Isolation Forest) and performance metrics (ROC-AUC, PR-AUC).
- **Matplotlib/Seaborn:** Visualization of signal reconstructions, error distributions, and threshold sensitivity curves.

## Metrics & Findings
| Model / Metric | F1-Score | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- |
| **Conv1D Autoencoder** | 0.943 | 0.988 | 0.967 |
| **LSTM Autoencoder** | 0.912 | 0.965 | 0.934 |
| **One-Class SVM** | 0.961 | 0.988 | 0.972 |
| **Isolation Forest** | 0.538 | 0.696 | 0.531 |

## Author
**Sepehr Barekati**
