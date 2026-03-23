# My ML Journey

Learning machine learning from the ground up — starting with the maths, then applying algorithms to real-world problems to understand what works best and when.

The goal: given any dataset, identify the right algorithm, clean and prepare the data, and evaluate the model properly.

## Progress

| Chapter | Topic | Status |
|---------|-------|--------|
| 01 | Foundations — maths & tools | ✓ done |
| 02 | Data skills — cleaning & engineering | ✓ done |
| 03 | Evaluation & metrics | in progress |
| 04 | Classification algorithms | upcoming |
| 05 | Regression algorithms | upcoming |
| 06 | Unsupervised learning | upcoming |
| 07 | Tuning & generalisation | upcoming |
| 08 | Algorithm selection mastery | upcoming |

## Structure

```
my-ml-journey/
├── 01-Foundations/              # Core maths behind ML
├── 02-Real-World/
│   └── phishing-detector/       # Binary classification project
├── 03-Data-Skills/              # Cleaning, EDA, feature engineering
│   ├── missing-values.ipynb
│   ├── outliers.ipynb
│   ├── encoding-categorical-data.ipynb
│   ├── feature-scaling.ipynb
│   ├── feature-engineering.ipynb
│   ├── eda.ipynb
│   └── titanic-survival.ipynb   # Chapter 2 project
└── README.md
```

## Chapter Notes

### 01 — Foundations
The maths that makes ML work — linear algebra, calculus, statistics, and getting comfortable with NumPy and Pandas.

### 02 - Real-World
Applying algorithms to real problems and comparing their performance.

**Phishing Detector** — binary classification to detect phishing URLs/emails using Logistic Regression, Decision Tree, Random Forest, and Naive Bayes.

### 03 — Data Skills
Cleaning and preparing data before modelling. Covers missing values, outliers, encoding, scaling, feature engineering and EDA.

**Project — Titanic Survival Prediction**
- Baseline logistic regression accuracy: **81%**
- Key finding: sex and pclass were the strongest predictors
- Tools: pandas, sklearn, seaborn, matplotlib



---

*Ongoing. Each project is an experiment in figuring out which algorithm fits which kind of data.*
