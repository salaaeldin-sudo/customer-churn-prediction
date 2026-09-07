
# Customer Churn Prediction

A small end-to-end machine learning project: clean a real customer dataset, train a classification model to predict churn, evaluate it properly, and explain what the results mean for the business.

Built as part of the **Arabian Academy AI Engineering Track — PROVE IT, Day 03 ("Build It")**.

## The problem

Telecom companies lose revenue every time a customer leaves ("churns"). If we can flag customers who are likely to churn *before* they leave, the business can offer them a retention deal instead of losing them for good. This project builds a first, simple model for that.

## Dataset

[Telecom Customer Churn Insights for Analysis](https://www.kaggle.com/datasets/abdullah0a/telecom-customer-churn-insights-for-analysis) — 1,000 telecom customers, with age, gender, tenure, contract type, internet service, charges, tech support, and whether each one churned.

> Note: 88% of customers in this dataset churned, the opposite of a typical real telecom dataset (usually 15–30%). See "Results" below for what that means for reading the metrics.

## What this project does

1. **Clean the data**
   - **Missing values:** `InternetService` was blank for 297 customers. Checked *who* they were first rather than guessing — every one of them also has `TechSupport == "No"`, meaning they're customers with no internet subscription at all, not a data entry error. Filled with the label `"No Internet"`.
   - **Duplicates:** checked for duplicate rows and duplicate customer IDs — found none.
   - **Outliers:** IQR rule on `Age`, `Tenure`, `MonthlyCharges`, `TotalCharges` — found real outliers in Age (8), Tenure (61), and TotalCharges (57), capped rather than dropped so no customers were lost.
2. **Encode & split** the features, then split into train/test sets (80/20, stratified on churn).
3. **Train** a Logistic Regression classifier (a simple, interpretable baseline).
4. **Evaluate** with Accuracy, Precision, Recall, F1, and a Confusion Matrix — not accuracy alone, since 88% of customers churned.
5. **Interpret** the model's coefficients to explain, in plain business terms, what drives churn up or down — and sanity-check the result instead of just reporting it.

## Results

| Metric | Score |
|---|---|
| Accuracy | 0.965 |
| Precision | 0.962 |
| Recall | 1.000 |
| F1-score | 0.981 |

**Being skeptical of the score:** 100% recall on churners is unusually clean for a churn model — real customer behavior is noisier than this. It suggests `TechSupport` and `ContractType` are close to a deterministic rule for churn in this particular dataset, rather than the messier signal you'd see in production. The one real weak spot: recall on the minority *No Churn* class was only 70% (7 loyal customers flagged as risks).

**What drives churn:**
- ⬇️ Having **Tech Support**, a **one/two-year contract**, and **longer tenure** are the strongest protection against churn.
- ⬆️ **No internet service** and **higher monthly/total charges** raise churn risk the most.

See [`notebook/churn_prediction.ipynb`](notebook/churn_prediction.ipynb) for the full walkthrough with charts, or [`src/train.py`](src/train.py) for the same pipeline as a plain script.

## Project structure

```
customer-churn-prediction/
├── data/
│   └── customer_churn.csv        # raw dataset
├── notebook/
│   └── churn_prediction.ipynb    # full walkthrough (EDA → cleaning → model → evaluation)
├── src/
│   └── train.py                  # same pipeline as a runnable script
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
python3 src/train.py
```

or open `notebook/churn_prediction.ipynb` in Jupyter to see it step by step with charts.

## Tech stack

Python · pandas · scikit-learn · matplotlib · seaborn

## Next steps

Validate the pipeline against a second, independently-collected churn dataset before trusting the coefficients — the near-perfect separability here is a property of this particular data, not a guarantee the same model would perform this well in production.
