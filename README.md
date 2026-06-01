<h1 align="center">Customer Purchase Prediction — Ensemble</h1>

<p align="center">
  <code>Python</code> &nbsp;&nbsp; <code>NumPy</code> &nbsp;&nbsp; <code>Pandas</code> &nbsp;&nbsp; <code>Matplotlib</code> &nbsp;&nbsp; <code>Machine Learning</code>
</p>

<p align="center">
  Academic Project &nbsp;·&nbsp; MPSTME, NMIMS &nbsp;·&nbsp; 2025–2026
</p>

---

### Overview

A from-scratch implementation and comparative study of three supervised machine learning classifiers — Decision Tree, Naive Bayes, and a Hybrid AdaBoost-NB ensemble — built entirely in Python without any ML libraries.

**Objective:** Predict whether a customer will make a purchase based on 8 behavioural and demographic features from a dataset of 1,500 records (`PurchaseStatus` — binary classification).

The project was developed as part of the Principles of Artificial Intelligence course and demonstrates the implementation of core ML algorithms from first principles, including entropy-based splitting, probabilistic inference with Laplace smoothing, and boosting ensembles.

---

### Pipeline

<table style="border: none; border-collapse: collapse;">
  <tr>
    <td style="border: none; padding: 6px 16px 6px 0;"><strong>1</strong></td>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Data Loading & Exploration</strong></td>
    <td style="border: none; padding: 6px 0;">Load dataset, inspect class distribution and feature types</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 16px 6px 0;"><strong>2</strong></td>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Preprocessing & Feature Engineering</strong></td>
    <td style="border: none; padding: 6px 0;">Min-Max normalisation; categorical binning for NB and AdaBoost</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 16px 6px 0;"><strong>3</strong></td>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Train-Test Split</strong></td>
    <td style="border: none; padding: 6px 0;">80/20 split (1,200 train / 300 test) with <code>seed=42</code>; identical partitions across all models</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 16px 6px 0;"><strong>4</strong></td>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Model Implementation & Training</strong></td>
    <td style="border: none; padding: 6px 0;">Decision Tree, Naive Bayes, and Hybrid AdaBoost-NB built from scratch</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 16px 6px 0;"><strong>5</strong></td>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Feature Importance Analysis</strong></td>
    <td style="border: none; padding: 6px 0;">Cumulative Information Gain tracked per feature across tree splits</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 16px 6px 0;"><strong>6</strong></td>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Evaluation & Results</strong></td>
    <td style="border: none; padding: 6px 0;">Accuracy, Precision, Recall, F1-Score computed manually; confusion matrices plotted</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 16px 6px 0;"><strong>7</strong></td>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>K-Fold Cross-Validation</strong></td>
    <td style="border: none; padding: 6px 0;">5-Fold CV for robust, partition-independent performance estimates</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 16px 6px 0;"><strong>8</strong></td>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Learning Curves</strong></td>
    <td style="border: none; padding: 6px 0;">Training vs. validation F1 plotted across increasing dataset sizes</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 16px 6px 0;"><strong>9</strong></td>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Final Comparison</strong></td>
    <td style="border: none; padding: 6px 0;">Side-by-side metric summary with task-based model recommendations</td>
  </tr>
</table>

---

### Models

<table style="border: none; border-collapse: collapse;">
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Decision Tree</strong></td>
    <td style="border: none; padding: 6px 0;">Recursive binary splitting using Information Gain (entropy). Stopping conditions: pure node · depth ≥ 8 · fewer than 5 samples</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Naive Bayes</strong></td>
    <td style="border: none; padding: 6px 0;">Frequency-based probabilistic classifier with Laplace smoothing (α=1). Features discretised into labelled bins before training</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Hybrid AdaBoost-NB</strong></td>
    <td style="border: none; padding: 6px 0;">Boosting ensemble using Naive Bayes as the weak learner across T=20 rounds. Architecturally independent of the standalone NB model</td>
  </tr>
</table>

---

### Recommendations

<table style="border: none; border-collapse: collapse;">
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Revenue Maximisation</strong></td>
    <td style="border: none; padding: 6px 0;">Decision Tree — highest F1 and Recall, lowest false negatives</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Cost-Sensitive Campaigns</strong></td>
    <td style="border: none; padding: 6px 0;">Naive Bayes — highest Precision, lowest false positives</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Balanced Robustness</strong></td>
    <td style="border: none; padding: 6px 0;">Hybrid AdaBoost-NB — best overall balance across all metrics</td>
  </tr>
</table>

---

### Technologies

<table style="border: none; border-collapse: collapse;">
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Language</strong></td>
    <td style="border: none; padding: 6px 0;">Python 3</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Environment</strong></td>
    <td style="border: none; padding: 6px 0;">Google Colab (Jupyter Notebook)</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Libraries</strong></td>
    <td style="border: none; padding: 6px 0;">NumPy · Pandas · Matplotlib · math · random · collections</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>ML Libraries Used</strong></td>
    <td style="border: none; padding: 6px 0;">None — all algorithms implemented from scratch</td>
  </tr>
</table>

---

### Dataset

<table style="border: none; border-collapse: collapse;">
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>File</strong></td>
    <td style="border: none; padding: 6px 0;"><code>customer_purchase_data.csv</code></td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Records</strong></td>
    <td style="border: none; padding: 6px 0;">1,500 customer records</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Features</strong></td>
    <td style="border: none; padding: 6px 0;">8 behavioural and demographic inputs (Age, AnnualIncome, TimeSpentOnWebsite, NumberOfPurchases, and others)</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Target</strong></td>
    <td style="border: none; padding: 6px 0;"><code>PurchaseStatus</code> — binary (0: No Purchase, 1: Purchase)</td>
  </tr>
</table>

---

### How to Run

1. Clone the repository
2. Upload `customer_purchase_data.csv` when prompted by the Colab file picker
3. Run all cells in order in Google Colab or a local Jupyter environment
4. Results, plots, and the final comparison table will be generated inline

---

*Academic project submitted in partial fulfilment of B.Tech in Information Technology, MPSTME NMIMS, 2025–2026.*
