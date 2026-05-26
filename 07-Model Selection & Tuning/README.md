# Chapter 7 — Model Selection & Tuning

## What This Chapter is About

For 6 chapters, every hyperparameter we set (`max_depth=6`, `n_estimators=200`, `learning_rate=0.1`) was a guess. This chapter teaches you to find the best hyperparameters **systematically** instead of guessing.

---

## Notebooks

| Notebook | Topics Covered |
|---|---|
| `7.1-7.3-model-selection-tuning.ipynb` | GridSearchCV, RandomizedSearchCV, Pipelines |

---

## Concepts Covered

### GridSearchCV
- Tries every combination in a grid of hyperparameters
- Evaluates each combination with 5-fold cross-validation (not a single split — to avoid overfitting to the validation set)
- After finding the best combination, refits on 100% of training data
- Use when search space is small (< few hundred combinations)

### RandomizedSearchCV
- Randomly samples `n_iter` combinations instead of trying all of them
- Accepts continuous distributions (e.g. `uniform(0.01, 0.3)`) — can find values a fixed list would miss
- Use when search space is large (many hyperparameters or many values)

### Pipeline
- Chains preprocessing (StandardScaler) and model into one object
- Prevents **data leakage during CV** — scaler fits only on training folds, never sees the validation fold
- With GridSearch: use `stepname__param` syntax (e.g. `model__max_depth`)

---

## Key Rules

| Concept | Rule |
|---|---|
| `neg_mean_squared_error` | GridSearch always maximises — flip MSE sign so lower MSE = higher score |
| `n_jobs=-1` | Use all CPU cores — runs combinations in parallel |
| `best_estimator_` | The refitted model on all training data — always use this for predictions |
| Pipeline + GridSearch | Pass raw (unscaled) X_train — Pipeline handles scaling per fold internally |
| `model__max_depth` | Double underscore tells GridSearch which Pipeline step the param belongs to |

---

## Problem Used
**Tamil Nadu Crop Yield Prediction** — predict yield (kg/hectare) from rainfall, temperature, fertilizer, soil quality, irrigation, and crop type.

---

## Prerequisites
Chapters 1–6 complete. Especially:
- Cross-validation (Ch 3)
- RandomForestRegressor (Ch 5.6)
- StandardScaler (Ch 2, 5)
