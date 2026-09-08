# Customer Churn Prediction

A small end-to-end machine learning project: clean a real customer dataset, train a classification model to predict churn, evaluate it properly, and explain what the results mean for the business.

Built as part of the **Arabian Academy AI Engineering Track — PROVE IT, Day 03 ("Build It")**.

## The problem

Telecom companies lose revenue every time a customer leaves ("churns"). If we can flag customers who are likely to churn *before* they leave, the business can offer them a retention deal instead of losing them for good. This project builds a first, simple model for that.

## Dataset

[Telecom Customer Churn Insights for Analysis](https://www.kaggle.com/datasets/abdullah0a/telecom-customer-churn-insights-for-analysis) — 1,000 telecom customers, with age, gender, tenure, contract type, internet service, charges, tech support, and whether each one churned.

> Note: 88% of customers in this dataset churned, the opposite of a typical real telecom dataset (usually 15–30%). See "Results" below for what that means for reading the metrics.

## Everything is in one notebook

**[`churn_prediction.ipynb`](churn_prediction.ipynb)** — the whole pipeline, in order, already executed with its charts and output visible on GitHub without running anything:

1. **Load & explore (EDA)** — check the churn balance first.
2. **Clean** — missing values in `InternetService` (a real "no internet" category, confirmed by checking, not assumed) and a duplicate check.
3. **Encode & split** — one-hot encode the text columns, then split into train/test *before* touching outliers.
4. **Outliers** — IQR rule on the numeric columns, with bounds learned from the training set only and applied to both sets (avoids leaking test-set information into cleaning).
5. **Train** a Logistic Regression classifier.
6. **Evaluate** with Accuracy, Precision, Recall, F1, and a Confusion Matrix — not accuracy alone, since 88% of customers churned.
7. **Interpret** the model's coefficients in plain business terms, and sanity-check the result instead of just reporting it.

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

## Project structure

```
customer-churn-prediction/
├── data/
│   └── customer_churn.csv     # raw dataset
├── churn_prediction.ipynb     # the whole pipeline — load, clean, encode, train, evaluate, interpret
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
jupyter notebook churn_prediction.ipynb
```

Run all cells top to bottom — each cleaning/encoding step prints or displays the data right after it changes, so you can see the effect of every step as you go.

## Tech stack

Python · pandas · scikit-learn · matplotlib · seaborn · Jupyter

## Next steps

Validate the pipeline against a second, independently-collected churn dataset before trusting the coefficients — the near-perfect separability here is a property of this particular data, not a guarantee the same model would perform this well in production.
