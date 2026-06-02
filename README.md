<h1 align="center">Customer Purchase Prediction</h1>

<p align="center">
  <code>Python</code> &nbsp;&nbsp; <code>Machine Learning from Scratch</code> &nbsp;&nbsp; <code>Classification</code>
</p>

<p align="center">
  Academic Project &nbsp;·&nbsp; MPSTME, NMIMS &nbsp;·&nbsp; 2025–2026
</p>

---

### Overview

A comparative study of three supervised machine learning classifiers — Decision Tree, Naive Bayes, and a Hybrid AdaBoost-NB ensemble — built entirely from scratch in Python without any ML libraries. Applied to a 1,500-record customer behavioural dataset to predict whether a customer will make a purchase.

A distinguishing feature of this project is the Hybrid AdaBoost-NB ensemble, which uses weighted frequency tables to make Naive Bayes genuinely sensitive to AdaBoost's iterative sample reweighting — producing a model that demonstrably differs from standalone Naive Bayes and illuminates the trade-offs of ensemble boosting on probabilistic base learners.

All algorithms implemented using only: `pandas`, `numpy`, `math`, `random`, `matplotlib`, and `collections`.

---

### Dataset

<table style="border: none; border-collapse: collapse;">
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Source</strong></td>
    <td style="border: none; padding: 6px 0;">customer_purchase_data.csv</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Records</strong></td>
    <td style="border: none; padding: 6px 0;">1,500 customer records</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Features</strong></td>
    <td style="border: none; padding: 6px 0;">8 behavioural and demographic features (Age, Annual Income, Time Spent on Website, Number of Purchases, Product Category, Loyalty Program, Discounts Availed, Gender)</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Target</strong></td>
    <td style="border: none; padding: 6px 0;">Purchase status — No Purchase (0) vs Purchase (1)</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Class Distribution</strong></td>
    <td style="border: none; padding: 6px 0;">57% No Purchase · 43% Purchase (moderate imbalance)</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Missing Values</strong></td>
    <td style="border: none; padding: 6px 0;">None — all 1,500 records complete</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Train / Test Split</strong></td>
    <td style="border: none; padding: 6px 0;">80/20 (1,200 train · 300 test · seed=42)</td>
  </tr>
</table>

---

### Classifiers Implemented

<table style="border: none; border-collapse: collapse;">
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Decision Tree</strong></td>
    <td style="border: none; padding: 6px 0;">Recursive entropy-based splitting with Information Gain; max_depth=8, min_samples=5; feature importance tracked via cumulative information gain across all splits</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Naive Bayes</strong></td>
    <td style="border: none; padding: 6px 0;">Frequency-based classifier with weighted Laplace smoothing; continuous features discretised into labelled bins (Age, AnnualIncome, TimeSpentOnWebsite, NumberOfPurchases); supports sample_weight parameter for use as AdaBoost base learner</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Hybrid AdaBoost-NB</strong></td>
    <td style="border: none; padding: 6px 0;">AdaBoost ensemble with weighted Naive Bayes as the base learner; T=20 boosting rounds; sample weights propagated via weighted frequency tables so each weak learner genuinely responds to the reweighted distribution; error guard (error ≥ 0.5) prevents sign reversal</td>
  </tr>
</table>

---

### Analyses Included

<table style="border: none; border-collapse: collapse;">
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Feature Importance</strong></td>
    <td style="border: none; padding: 6px 0;">Cumulative normalised information gain per feature — identifies which behavioural and demographic signals most strongly predict purchase intent</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>5-Fold Cross-Validation</strong></td>
    <td style="border: none; padding: 6px 0;">Mean F1-Score and standard deviation across 5 folds for all three models — provides statistically robust performance estimates beyond a single split</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Learning Curves</strong></td>
    <td style="border: none; padding: 6px 0;">Training vs validation F1-Score across increasing training set sizes — diagnoses overfitting, underfitting, and convergence behaviour per model</td>
  </tr>
</table>

---

### Results

<table style="border: none; border-collapse: collapse;">
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Model</strong></td>
    <td style="border: none; padding: 6px 20px 6px 0;"><strong>Accuracy</strong></td>
    <td style="border: none; padding: 6px 20px 6px 0;"><strong>Precision</strong></td>
    <td style="border: none; padding: 6px 20px 6px 0;"><strong>Recall</strong></td>
    <td style="border: none; padding: 6px 0;"><strong>F1-Score</strong></td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;">Decision Tree</td>
    <td style="border: none; padding: 6px 20px 6px 0;">0.9033</td>
    <td style="border: none; padding: 6px 20px 6px 0;">0.8672</td>
    <td style="border: none; padding: 6px 20px 6px 0;">0.9024</td>
    <td style="border: none; padding: 6px 0;">0.8845</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;">Hybrid AdaBoost-NB</td>
    <td style="border: none; padding: 6px 20px 6px 0;">0.8633</td>
    <td style="border: none; padding: 6px 20px 6px 0;">0.8475</td>
    <td style="border: none; padding: 6px 20px 6px 0;">0.8130</td>
    <td style="border: none; padding: 6px 0;">0.8299</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;">Naive Bayes</td>
    <td style="border: none; padding: 6px 20px 6px 0;">0.7367</td>
    <td style="border: none; padding: 6px 20px 6px 0;">0.9231</td>
    <td style="border: none; padding: 6px 20px 6px 0;">0.3902</td>
    <td style="border: none; padding: 6px 0;">0.5486</td>
  </tr>
</table>

F1-Score is the primary evaluation metric, selected for its ability to penalise classifiers that sacrifice Precision for Recall on this moderately imbalanced dataset (57/43).

**Recommended model: Decision Tree** — leads on F1-Score (0.8845) and Recall (0.9024), making it the primary recommendation when minimising missed buyers is the business priority. **Hybrid AdaBoost-NB is the recommended ensemble** — it substantially recovers the Recall lost by weighted Naive Bayes (0.8130 vs 0.3902) and demonstrates that boosting can meaningfully compensate for a destabilised probabilistic base learner. Naive Bayes (weighted) leads on Precision (0.9231) and is preferred only in highly cost-sensitive campaigns where false positives are extremely costly and low Recall is acceptable.

---

### Key Findings

- **Decision Tree is the recommended model** under F1-Score (primary metric) and Recall — it misses the fewest actual buyers and is the strongest choice when revenue maximisation is the business objective
- **Hybrid AdaBoost-NB demonstrates the core value of boosting** — weighted Naive Bayes alone collapses to low Recall (0.3902) under AdaBoost's reweighting pressure, but the ensemble recovers to F1 0.8299 by aggregating 20 weighted weak learners, each responding to a different reweighted distribution
- **Weighted Naive Bayes achieves the highest Precision (0.9231)** at the cost of very low Recall — a meaningful trade-off for campaigns where each customer contact is expensive and false positives must be minimised
- **Propagating sample weights through frequency tables** is essential for AdaBoost-NB to function correctly — without weighted counts, every weak learner is identical and the ensemble reduces to plain Naive Bayes
- **AnnualIncome, TimeSpentOnWebsite, and NumberOfPurchases** are the strongest purchase predictors by information gain — consistent with the intuition that engaged, high-income repeat visitors are the most likely buyers

---

### Technologies

<table style="border: none; border-collapse: collapse;">
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Language</strong></td>
    <td style="border: none; padding: 6px 0;">Python</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Environment</strong></td>
    <td style="border: none; padding: 6px 0;">Google Colab</td>
  </tr>
  <tr>
    <td style="border: none; padding: 6px 24px 6px 0;"><strong>Libraries</strong></td>
    <td style="border: none; padding: 6px 0;">pandas · numpy · math · random · matplotlib · collections</td>
  </tr>
</table>

---

*Academic project submitted in partial fulfilment of B.Tech in Information Technology, MPSTME NMIMS, 2025–2026.*
