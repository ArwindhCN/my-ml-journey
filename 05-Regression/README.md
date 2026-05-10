# Chapter 5 — Regression Algorithms

## What this chapter covers

Supervised learning for **continuous output prediction** — predicting a number, not a category.

Every algorithm here answers the same question: *given input features, predict a numerical value.* The difference is in how each algorithm models the relationship between inputs and output.

---

## The Big Idea

In Chapter 4 we predicted categories — spam/not-spam, survived/not-survived. The output was a label.

In Chapter 5 the output is a **number** — house price, temperature, salary, sales volume. The goal is to get as close to the true number as possible.

The real-world thread running through this chapter is **Mumbai house price prediction** — predicting ₹ price from features like area, location, bedrooms, and age.

---

## Algorithms Covered

| Notebook | Algorithm | Core idea |
|---|---|---|
| 5.1 | Linear Regression | Fit a straight line through the data — minimise sum of squared errors |
| 5.2 | Ridge Regression | Linear Regression + L2 penalty — shrinks coefficients, handles multicollinearity |
| 5.3 | Lasso Regression | Linear Regression + L1 penalty — shrinks some coefficients to exactly zero (feature selection) |
| 5.4 | Decision Tree Regressor | Split data into rectangles — predict the mean of each rectangle |
| 5.5 | Random Forest Regressor | Ensemble of many decision trees — average their predictions |
| 5.6 | Gradient Boosting / XGBoost | Ensemble where each tree corrects the previous tree's errors |

---

## Evaluation Metrics (Chapter 3 recap)

All regression algorithms are evaluated using the same metrics:

| Metric | Formula | Interpretation |
|---|---|---|
| MAE | mean(\|y - ŷ\|) | Average absolute error — same unit as target |
| MSE | mean((y - ŷ)²) | Penalises large errors more — sensitive to outliers |
| RMSE | √MSE | Same unit as target — most interpretable |
| R² | 1 - SS_res/SS_tot | Proportion of variance explained — 1.0 = perfect, 0 = predicts mean only |

---

## Key Concepts Introduced

**Regularisation** — adding a penalty to the loss function to prevent overfitting.
- L2 (Ridge): penalises sum of squared coefficients — shrinks all, removes none
- L1 (Lasso): penalises sum of absolute coefficients — shrinks some to exactly zero

**Bias-Variance Tradeoff** — underfitting (high bias) vs overfitting (high variance). Regularisation controls this.

**Ensemble methods** — combining many weak models into one strong model.
- Bagging (Random Forest): parallel trees on random subsets — reduce variance
- Boosting (Gradient Boosting): sequential trees each correcting previous errors — reduce bias

**Hyperparameter tuning** — `max_depth`, `n_estimators`, `learning_rate`, `alpha` — how to choose them and what they control.

---

## Algorithm Selection Guide

| Situation | Recommended algorithm |
|---|---|
| Need interpretable coefficients | Linear / Ridge / Lasso |
| Many correlated features | Ridge Regression |
| Want automatic feature selection | Lasso Regression |
| Non-linear relationships in data | Decision Tree / Random Forest |
| Best possible accuracy | Gradient Boosting / XGBoost |
| Large dataset, fast training needed | Linear Regression / Random Forest |
| Small dataset, overfitting risk | Ridge or Lasso |

---

## Chapter Project

**Mumbai House Price Prediction** — predict ₹ price from area (sq ft), location tier, number of bedrooms, building age, and distance to metro.

Builds progressively: Linear Regression baseline → Ridge/Lasso for regularisation → Random Forest for non-linearity → XGBoost for best performance. Each step benchmarked against the previous.

---

## Prerequisites

- Chapter 3 (Evaluation Metrics — MAE, RMSE, R², cross-validation)
- Chapter 4 (Classification algorithms — Decision Trees and ensemble intuition carries over)

## Environment

```
Python 3.11.9 | ml-env virtual environment
sklearn, numpy, pandas, matplotlib, xgboost
```
