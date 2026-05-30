# My ML Journey

Learning machine learning from the ground up — starting with the maths, then applying algorithms to real-world problems to understand what works best and when.

The goal: given any dataset, identify the right algorithm, clean and prepare the data, build a pipeline, tune the hyperparameters, and evaluate the model properly.

## Progress

| Chapter | Topic | Status |
|---------|-------|--------|
| 01 | Foundations — maths & tools | ✅ done |
| 02 | Data skills — cleaning & feature engineering | ✅ done |
| 03 | Evaluation & metrics | ✅ done |
| 04 | Classification algorithms | ✅ done |
| 05 | Regression algorithms | ✅ done |
| 06 | Unsupervised learning | ✅ done |
| 07 | Model selection & tuning | ✅ done |
| 08 | End-to-end projects | 🔄 in progress |

## Structure

```
my-ml-journey/
├── 01-Foundations/              # Linear algebra, calculus, stats, NumPy, Pandas
├── 02-Data-Skills/              # Missing values, encoding, scaling, EDA
│   └── titanic-survival.ipynb  # Chapter 2 project — 81% accuracy
├── 03-Evaluation/               # Confusion matrix, precision, recall, F1, ROC-AUC, CV
├── 04-Classification/           # LR, KNN, DT, RF, SVM, NB
│   ├── 4.1 through 4.7 notebooks
│   └── 4.8-spam-detector.ipynb # Chapter 4 project — spam classifier
├── 05-Regression/               # Linear, Ridge, Lasso, Polynomial, DT, RF, XGBoost
│   ├── 5.1-linear-regression.ipynb
│   ├── 5.2-ridge-regression.ipynb
│   ├── 5.3-lasso-regression.ipynb
│   ├── 5.4-polynomial-regression.ipynb
│   ├── 5.5-decision-tree-regression.ipynb
│   ├── 5.6-random-forest-regression.ipynb
│   ├── 5.7-xgboost-regression.ipynb       # Tamil Nadu crop yield
│   └── 5.8-mumbai-house-price-project.ipynb # Chapter 5 project — all 7 algorithms
├── 06-Unsupervised-Learning/    # K-Means, Hierarchical, DBSCAN, PCA, t-SNE
│   ├── 6.1-kmeans-clustering.ipynb
│   ├── 6.2-hierarchical-clustering.ipynb
│   ├── 6.3-dbscan.ipynb
│   └── 6.4-pca-tsne.ipynb
├── 07-Model-Selection-&-Tuning/ # GridSearchCV, RandomizedSearchCV, Pipelines
│   ├── 7.1-7.3-model-tuning.ipynb
│   └── 7.4-project.ipynb        # Chapter 7 project
└── README.md
```

## Chapter Notes

### 01 — Foundations
The maths that makes ML work — linear algebra, calculus, statistics — and getting comfortable with NumPy and Pandas.

### 02 — Data Skills
Cleaning and preparing data before modelling. Covers missing values, outliers, encoding, scaling, feature engineering and EDA.

**Project — Titanic Survival Prediction**
- Baseline logistic regression accuracy: **81%**
- Key finding: sex and pclass were the strongest predictors

### 03 — Evaluation & Metrics
Beyond accuracy — confusion matrix, precision, recall, F1, ROC-AUC, cross-validation, and regression metrics (MAE, MSE, RMSE, R²).

### 04 — Classification
Six algorithms compared — Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Naive Bayes. Built understanding of the math behind each.

**Project — Spam Detector**
- Real-world text classification using NLP preprocessing + Naive Bayes

### 05 — Regression
Seven regression algorithms from scratch — Linear through XGBoost. Core insight: when to regularise (Ridge/Lasso), when to go non-linear (Polynomial/Tree-based), when to boost (XGBoost).

**Project — Mumbai House Price Prediction**
- All 7 algorithms compared on the same dataset
- XGBoost delivered best performance

**Bonus — Tamil Nadu Crop Yield (5.7)**
- XGBoost regression on district-level agricultural data

### 06 — Unsupervised Learning
Clustering (K-Means, Hierarchical, DBSCAN) and dimensionality reduction (PCA, t-SNE). Key insight: choosing the right algorithm depends on cluster shape, dataset size, and whether K is known upfront.

### 07 — Model Selection & Tuning
Systematic hyperparameter search instead of guessing. Pipelines to prevent data leakage during cross-validation.

- **GridSearchCV** — exhaustive search over all parameter combinations
- **RandomizedSearchCV** — faster search by sampling the parameter space
- **Pipelines** — chains preprocessing + model so the scaler never sees validation fold data

---

*Ongoing. Each project is an experiment in understanding which algorithm fits which kind of problem — and why.*
