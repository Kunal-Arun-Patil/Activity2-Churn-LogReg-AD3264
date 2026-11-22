# Activity 2: Binary Classification – Telecom Customer Churn (Logistic Regression)

**Course:** Regression and Classification Algorithms | **CIE 2 – Activity 2** | Unit II – Classification & Logistic Regression
**Student:** Kunal Arun Patil | **Roll No.:** AD3264
**Submission GitHub ID:** [https://github.com/complexiota](https://github.com/complexiota)

## Activity description
Build a Logistic Regression model for a two-class problem, convert predicted probabilities into class labels, evaluate it with classification metrics, compare it with a majority-class baseline, study the decision threshold and justify a recommended threshold.

**Problem:** A telecom retention team wants to know which customers are likely to leave (churn) so they can contact them first. A missed churner (false negative) costs more than a wasted offer (false positive), so recall on churners matters most.

## Dataset
- **Name:** IBM Telco Customer Churn
- **Source:** https://github.com/IBM/telco-customer-churn-on-icp4d (also on Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **File:** `telco.csv` – 7,043 rows × 21 columns; one row = one customer; public sample data (no real personal data)
- **Target:** `Churn` – **1 = Churn (positive class)**, **0 = Stay**. Churn rate = 26.5% (imbalanced)
- **Features:** tenure, contract, internet service, payment method, services used, demographics, charges (`customerID` is not used)

## Work completed
1. Problem framing, data dictionary and data-quality checks (missing `TotalCharges` filled with 0 for tenure-0 customers; duplicates reviewed; ID dropped)
2. Class-distribution bar chart and exploratory plots, each with an observation
3. Feature preparation: one-hot encoding, merging "No internet/phone service" into "No", VIF check (dropped `MonthlyCharges` and `TotalCharges`: max VIF 800+ → 2.8), standard scaling fitted on **training data only**
4. **Stratified 80:20 split** (5,634 train / 1,409 test; churn rate 0.265 in both)
5. Sigmoid explanation and plot; why Linear Regression is unsuitable
6. Logistic Regression: intercept, coefficients, odds ratios, p-values and interpretation
7. Predicted probabilities and labels (sample table), effect of thresholds 0.3 / 0.5 / 0.7
8. Labelled confusion matrix (TP, FP, FN, TN explained), accuracy, precision, recall, specificity, F1 (with formulas), full classification report
9. Majority-class baseline comparison, ROC curve with ROC-AUC, precision–recall curve
10. Threshold comparison table and a **cost-based recommended threshold chosen with cross-validation on training data only** (no test-set tuning)
11. Train vs test comparison (overfitting check), decision-behaviour plot (churn probability vs tenure by contract type)
12. One new-customer prediction, conclusion, limitations, ethics note and decision log

## Key results (test set)

| Metric (threshold 0.5) | Result |
|---|---|
| Accuracy | 0.796 |
| Precision | 0.634 |
| Recall (sensitivity) | 0.545 |
| Specificity | 0.886 |
| F1-score | 0.586 |
| ROC-AUC | 0.839 |
| Baseline "always Stay" | accuracy 0.735, recall 0.0, AUC 0.5 |

| Threshold | Predicted positive | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 0.3 | 530 | 0.752 | 0.523 | 0.741 | 0.613 |
| 0.5 | 322 | 0.796 | 0.634 | 0.545 | 0.586 |
| 0.7 | 94 | 0.764 | 0.723 | 0.182 | 0.291 |

**Recommended threshold ≈ 0.16** (chosen on training data using an illustrative cost of $50 per wasted offer vs $200 per missed churner): on the test set it catches about 90% of churners (precision about 45%) and lowers total cost versus 0.5. The trade-off accepted is more false alarms.
Train and test scores are close (ROC-AUC 0.848 vs 0.839), so there is no sign of overfitting.

## Limitations
- Cost figures are illustrative; real company values would change the best threshold.
- Single snapshot, no time dimension or reasons for leaving; results show association, not causation.
- The model was not checked for fairness across groups.
- A prediction is a probability-based estimate, not a guaranteed outcome. The model should only help a human team prioritise retention offers, not decide on its own to refuse service or change prices.

## Repository files
```
.
├── telco_churn_logistic_regression_prediction.ipynb   # full executed notebook with outputs
├── telco.csv                                          # dataset
├── requirements.txt
└── README.md
```

## How to run
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
jupyter notebook telco_churn_logistic_regression_prediction.ipynb
```
Run all cells top to bottom (random seed fixed at 42). `telco.csv` must be in the same folder as the notebook.
