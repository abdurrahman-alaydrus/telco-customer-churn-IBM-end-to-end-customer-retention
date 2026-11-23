# Telco Customer Churn Prediction (IBM Dataset)

End-to-end churn analysis and classification on the IBM Telco Customer Churn dataset: exploratory analysis, a leakage-aware preprocessing pipeline, a comparison of four classifiers, model interpretation, and a simple business-impact simulation of a retention campaign.

**Notebook:** `telco-churn-customer-ibm.ipynb`
**Data source:** [Telco Customer Churn (IBM) on Kaggle](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset/data)

---

## Dataset

- 7,043 customers, 33 columns (all in California, US)
- Target: `Churn Label` (Yes/No), with a class balance of **73.5% stay / 26.5% churn**
- 19 input features used: demographics (gender, senior citizen, partner, dependents), account info (tenure, contract, payment method, paperless billing, monthly/total charges), and services (phone, multiple lines, internet type, online security/backup, device protection, tech support, streaming TV/movies)

**Columns deliberately excluded from the model:** `Churn Score`, `Churn Value`, `CLTV`, `Churn Reason`, and geographic fields. The first four are either derived from or only known after the churn outcome, so including them would leak the target. `Churn Reason` and `CLTV` are used only for analysis and the business simulation.

## Workflow

1. **Data cleaning**
   - `Total Charges` coerced to numeric (11 blank values become NaN and are median-imputed in the pipeline)
   - `"No phone service"` / `"No internet service"` collapsed to `"No"` in the add-on service columns
2. **EDA**
   - Churn distribution, churn rate by contract type, churn reasons, categorical features split by churn, and box plots for tenure, monthly charges, total charges, and CLTV
3. **Train/test split**
   - 80/20 stratified split (5,634 / 1,409), `random_state=42`
   - 26 duplicate rows dropped from the training set only
4. **Preprocessing (`ColumnTransformer`)**
   - Numeric (tenure, monthly charges, total charges): median imputation, then standard scaling
   - Low-cardinality categoricals (internet service, contract, payment method): `TargetEncoder`
   - Binary features: Yes/No and Female/Male mapped to 1/0, then passed through
5. **Modeling**
   - Logistic Regression, KNN, Decision Tree, and SVM (RBF), each wrapped in the same pipeline
6. **Evaluation**
   - Train vs. test confusion matrix, accuracy, precision, recall, F1, ROC-AUC, and PR-AUC
7. **Interpretation**
   - Logistic regression coefficients and permutation importance (F1, test set)
8. **Business simulation**
   - Estimated net savings and ROI of a retention campaign driven by the model's predictions

## Results

### Churn drivers from EDA

| Contract type  | Churn rate |
|----------------|-----------:|
| Month-to-month | 42.7%      |
| One year       | 11.3%      |
| Two year       | 2.8%       |

### Model comparison (test set)

| Model               | Recall | Precision | F1    | ROC-AUC | PR-AUC |
|---------------------|-------:|----------:|------:|--------:|-------:|
| **Logistic Regression** | 0.575  | 0.653     | **0.612** | **0.849** | 0.647 |
| SVM (RBF)           | 0.487  | 0.679     | 0.567 | 0.817   | 0.647  |
| KNN                 | 0.540  | 0.572     | 0.556 | 0.789   | 0.543  |
| Decision Tree       | 0.553  | 0.528     | 0.540 | 0.687   | 0.411  |

**Selected model: Logistic Regression.** Test accuracy 0.806, with a small train/test gap (F1 0.622 vs. 0.612). The Decision Tree overfits heavily (train F1 0.871 vs. test F1 0.540), and KNN also shows a notable gap.

Confusion matrix (test, threshold 0.5): TP = 215, FP = 114, FN = 159, TN = 921.

### What drives churn (best model)

- **Increases churn risk:** contract type (as target-encoded), payment method, higher monthly charges, fiber-optic internet
- **Decreases churn risk:** longer tenure, having dependents, having phone service, online security, tech support
- Permutation importance ranks **tenure**, **monthly charges**, **dependents**, and **contract** as the most influential features. Gender, streaming services, and senior-citizen status contribute almost nothing.

### Business simulation

Assumptions (set in the notebook, **not** derived from data except CLTV): average CLTV $4,400, retention offer cost $150 per targeted customer, 60% offer acceptance rate.

| Metric                                   | Value        |
|------------------------------------------|-------------:|
| Churners in test set (TP + FN)           | 374          |
| Loss if no action                        | $1,645,600   |
| Customers targeted (TP + FP)             | 329          |
| Campaign cost                            | $49,350      |
| Spend on false positives                 | $17,100      |
| Customers rescued (TP x 60%)             | 129          |
| Revenue saved                            | $567,600     |
| **Net savings**                          | **$518,250** |
| ROI                                      | ~1,050%      |

> **Read this with care.** The ROI is driven almost entirely by the assumed acceptance rate, offer cost, and a single average CLTV applied to every customer. It is a scenario illustration, not a measured result. See limitations below.

## Limitations and next steps

- **No hyperparameter tuning or cross-validation.** All models use mostly default settings, and the best model was chosen by test-set F1, so the reported test score is slightly optimistic. Use cross-validation for selection and keep the test set for a final check.
- **Default 0.5 threshold.** Recall on churners is only ~0.58, so 159 churners are missed. Since a missed churner is far costlier than a wasted offer under the notebook's own assumptions, threshold tuning (e.g., on the precision-recall curve) would likely improve the business outcome.
- **No class-imbalance handling.** The pipeline is an `imblearn` pipeline but applies no resampling or class weights. Trying `class_weight='balanced'` or SMOTE is a natural next step.
- **Business simulation assumptions are unvalidated.** Sensitivity analysis over acceptance rate and offer cost would show how fragile the ROI is. Using per-customer CLTV instead of the average would also be more realistic.
- **Target encoding is fit inside the pipeline**, which avoids leakage across folds, but with only 3 to 4 categories per column a one-hot encoding is a simple alternative worth comparing.
- **Single-state, single-snapshot data.** Results may not generalize to other regions or time periods.
- Candidate models to add: Random Forest, XGBoost/LightGBM, and calibrated probabilities.

## Getting started

### Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn>=1.3   # TargetEncoder requires 1.3+
imbalanced-learn
gdown               # only needed for the Colab download cell
```

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn gdown
```

### Run

1. Download the dataset from the Kaggle link above (or let the first notebook cell fetch it with `gdown`) and save it as `data.xlsx`.
2. If you are not running in Colab, update the path in `pd.read_excel('/content/data.xlsx')` to point to your local file.
3. Open `telco-churn-customer-ibm.ipynb` and run all cells.

## Project structure

```
.
├── telco-churn-customer-ibm.ipynb   # full analysis and modeling
├── data.xlsx                        # dataset (not included; download separately)
└── README.md
```
