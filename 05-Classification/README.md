# 05-Classification

Chapter 4 of the ML Journey — Classification Algorithms.

Learning to classify data points into categories using 6 different algorithms,
evaluated on real-world datasets using the metrics from Chapter 3.

---

## Notebooks

| File | Topic | Dataset | Status |
|------|-------|---------|--------|
| knn.ipynb | 4.1 Logistic Regression + 4.2 KNN | Pima Diabetes (768 patients) | ✓ done |
| decision-tree.ipynb | 4.3 Decision Tree | TBD | upcoming |
| random-forest.ipynb | 4.4 Random Forest | TBD | upcoming |
| svm.ipynb | 4.5 Support Vector Machine | TBD | upcoming |
| naive-bayes.ipynb | 4.6 Naive Bayes | TBD | upcoming |
| spam-classifier.ipynb | Chapter 4 Project — compare all 6 | Email spam/ham | upcoming |

---

## Key Results (knn.ipynb)

| Model | Accuracy | AUC |
|-------|----------|-----|
| KNN (best K) | ~75–78% | ~0.82 |
| Logistic Regression | ~77–80% | ~0.84 |

*Exact numbers depend on best K found during K-tuning.*

---

## What This Chapter Covers

**4.1 Logistic Regression**
- Sigmoid function: σ(z) = 1/(1+e^-z)
- Weighted sum z = w1×f1 + w2×f2 + bias
- Weights learned automatically via gradient descent
- Threshold tuning for imbalanced problems

**4.2 KNN — K-Nearest Neighbors**
- Lazy learner — no training, memorises all data
- Euclidean and Manhattan distance from scratch
- Effect of K — overfitting vs underfitting
- K-tuning: plot accuracy for K=1..20
- Why feature scaling is mandatory

---

## Algorithms Needing Scaling

| Algorithm | Needs Scaling? |
|-----------|---------------|
| Logistic Regression | YES |
| KNN | YES |
| SVM | YES |
| Decision Tree | No |
| Random Forest | No |
| Naive Bayes | No |

---

## Dataset — Pima Indians Diabetes

- Source: diabetes.csv (local copy in this folder)
- 768 patients, 8 features, 1 target (Outcome)
- Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- Target: 0 = Healthy (65%), 1 = Diabetic (35%)
- Note: Zeros in Glucose/BP/Insulin/BMI/SkinThickness are hidden missing values — replaced with median
