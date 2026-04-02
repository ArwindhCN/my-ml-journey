# Chapter 1 — Foundations

> **Goal:** Understand the maths and tools that every ML algorithm is built on. Not to memorise formulas — but to understand *why* they exist and *where* you'll see them again.

---

## What's in this chapter

| Notebook | Topic | Core idea |
|---|---|---|
| [1.1-linear-algebra.ipynb](./1.1-linear-algebra.ipynb) | Vectors, Matrices, Dot Products | All data is numbers in grids. ML prediction = dot product. |
| [1.2-calculus-gradient-descent.ipynb](./1.2-calculus-gradient-descent.ipynb) | Derivatives, Gradient Descent | Models learn by walking downhill on the error surface. |
| [1.3-statistics-probability.ipynb](./1.3-statistics-probability.ipynb) | Distributions, Bayes | Understand your data's shape before building any model. |
| [1.4-numpy.ipynb](./1.4-numpy.ipynb) | Arrays, Operations | NumPy is the fast number engine under all ML libraries. |
| [1.5-pandas.ipynb](./1.5-pandas.ipynb) | DataFrames, Cleaning, EDA | Your data workbench — load, inspect, clean, prepare. |

---

## The big picture — how it all connects

```
Raw data (CSV, database, sensor)
        ↓
  Pandas — load, clean, shape the data
        ↓
  NumPy — fast numerical operations
        ↓
  Linear Algebra — data is matrices, predictions are dot products
        ↓
  Statistics — understand distributions, handle outliers, correlations
        ↓
  Calculus — model learns weights by gradient descent
        ↓
  sklearn model (Chapter 4 onwards)
```

---

## Key things to never forget

### From 1.1 — Linear Algebra
- Every ML prediction is a **dot product**: `prediction = data_vector · weight_vector`
- For all rows at once: `predictions = X @ W`
- Matrix shape = (rows=samples, cols=features) — always check `.shape`

### From 1.2 — Calculus
- The model doesn't know the right weights — it **starts random and improves**
- **Loss function** = measures how wrong the model is (goal: minimise it)
- **Gradient descent** = walk downhill on the loss surface step by step
- Update rule: `W = W - learning_rate × gradient`
- Learning rate too large = overshoot. Too small = too slow.

### From 1.3 — Statistics
- **Mean** = use for normal data. **Median** = use for skewed data (salaries, prices)
- **Std deviation** = how spread out the data is (StandardScaler divides by this)
- **Normal distribution** → Z-score outlier detection
- **Skewed distribution** → IQR outlier detection, log transform
- **Bayes theorem** = update belief with evidence → foundation of Naive Bayes (Ch 4.6)
- **Correlation** ranges -1 to +1. Use in EDA to find useful features.

### From 1.4 — NumPy
- Always check: `.shape`, `.dtype`, `.ndim`
- `arr.reshape(-1, 1)` → column vector (sklearn often needs this)
- `arr[:, 0]` → select column. `arr[arr > 0]` → filter by condition
- Broadcasting = apply operation to whole array at once (like Excel formula on column)

### From 1.5 — Pandas
- First 5 commands on ANY new dataset: `shape`, `head()`, `info()`, `describe()`, `isnull().sum()`
- `df['col']` → Series (1D). `df[['col']]` → DataFrame (2D). Double brackets = 2D.
- sklearn expects `X` as 2D DataFrame, `y` as 1D Series
- Always check `df.dtypes` — no `object` columns should remain before ML
- `X = df.drop(columns=['target'])` / `y = df['target']` — the split pattern

---

## Where these concepts appear in later chapters

| Concept | Where it reappears |
|---|---|
| Dot product (X · W) | Every algorithm in Ch 4 & 5 internally |
| Gradient descent | Logistic Regression, SVM, Neural Nets |
| Mean/Median for missing values | Chapter 2 (Data Skills) |
| Skew & IQR | Chapter 2 (Outlier handling) |
| Bayes theorem | Chapter 4.6 (Naive Bayes) |
| Correlation | EDA in every project |
| `.reshape(-1, 1)` | Feature scaling in sklearn |
| X / y split | Every model from Chapter 4 onwards |

---

## Status
✓ Complete
