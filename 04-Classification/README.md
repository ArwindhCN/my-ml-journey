# 05-Classification

Chapter 4 of the ML Journey — Classification Algorithms.

Learning to classify data points into categories using 6 different algorithms,
evaluated on real-world datasets using the metrics from Chapter 3.

---

## Notebooks

| File | Topic | Dataset | Status |
|------|-------|---------|--------|
| 4.2-knn.ipynb | 4.1 Logistic Regression + 4.2 KNN | Pima Diabetes (768 patients) | ✅ done |
| 4.2-knn-concepts-deep.ipynb | KNN — every concept from scratch | Pima Diabetes | ✅ done |
| 4.2-knn-credit-risk-solved.ipynb | KNN real world problem | Chennai Bank — loan default | ✅ done |
| 4.3-decision-tree.ipynb | Decision Tree | Employee attrition — Bangalore IT | ✅ done |
| 4.3-decision-tree-concepts-deep.ipynb | Decision Tree — every concept from scratch | Employee attrition | ✅ done |
| 4.4-random-forest.ipynb | Random Forest | Crop disease — Wayanad, Kerala | ✅ done |
| 4.4-random-forest-concepts-deep.ipynb | Random Forest — every concept from scratch | Crop disease | ✅ done |
| 4.5-svm.ipynb | Support Vector Machine | Cancer detection — Kerala hospital | ✅ done |
| 4.6-naive-bayes.ipynb | Naive Bayes | SMS Spam (5,572 messages) | ✅ done |
| 4.7-chapter4-project-spam.ipynb | Chapter 4 Project — all 6 algorithms | SMS Spam (5,572 messages) | ✅ done |
| 4.8-chapter4-qa-revision.ipynb | Q&A Revision Notes — quiz session | — | ✅ done |

---

## Chapter 4 Project — Final Results

All 6 algorithms compared on the SMS Spam Collection dataset (5,572 messages).
Evaluated using 5-fold cross validation F1 score.

| Rank | Algorithm | F1 Score | Notes |
|------|-----------|----------|-------|
| 🥇 1st | SVM | 0.924 | Best for high-dimensional text |
| 🥈 2nd | Random Forest | 0.908 | Robust, stable across folds |
| 🥉 3rd | Naive Bayes | 0.896 | Speed champion — trains in milliseconds |
| 4th | Logistic Regression | 0.783 | Would improve significantly with C tuning |
| 5th | Decision Tree | 0.775 | Overfits on text data |
| 6th | KNN | 0.394 | Curse of dimensionality — 5000 TF-IDF dimensions |

**Key insight:** KNN was predicted to be worst before running — and it was. The curse of dimensionality is real.

---

## What This Chapter Covers

**4.1 Logistic Regression**
- Sigmoid function: σ(z) = 1/(1+e^-z) — squashes any number to 0-1
- Weighted sum z = w1×f1 + w2×f2 + bias
- Weights learned via gradient descent
- Output is a probability, threshold converts to class label

**4.2 KNN — K-Nearest Neighbors**
- Lazy learner — no training, memorises all data
- All work happens at prediction time
- Euclidean and Manhattan distance from scratch
- K controls bias-variance tradeoff
- Feature scaling is mandatory — distance is meaningless without it
- Curse of dimensionality — fails in high dimensions

**4.3 Decision Tree**
- Gini impurity: 1 - Σpᵢ² — Pure=0, Messy=0.5
- Information Gain — chooses the split that reduces impurity most
- Tries every feature × every threshold (exhaustive search)
- Overfits without pruning (max_depth, min_samples_leaf)
- Unstable — small data change = completely different tree

**4.4 Random Forest**
- Bagging — each tree sees ~63% of rows (bootstrap sample)
- Feature randomness — each split considers only √n_features
- Diversity makes errors cancel out across 100+ trees
- OOB score — free validation without a separate validation set
- MDI vs Permutation importance

**4.5 SVM — Support Vector Machine**
- Maximum margin hyperplane — widest possible boundary
- Support vectors — only boundary points define the margin
- C parameter — large C = strict = narrow margin = overfits
- Kernel trick — computes dot products in high dimensions without going there
- RBF kernel — measures similarity (close=1, far=0)
- Gamma — controls neighbourhood size (large=local, small=global)
- C and Gamma must be tuned together with GridSearchCV

**4.6 Naive Bayes**
- Bayes theorem: P(class|features) ∝ P(features|class) × P(class)
- Naive assumption: features are conditionally independent given class
- Three types: Multinomial (word counts), Bernoulli (presence/absence), Gaussian (continuous)
- Laplace smoothing (alpha=1) — prevents P=0 for unseen words
- Log trick — prevents numerical underflow during prediction
- Fastest algorithm — closed-form solution, no iteration

---

## Algorithm Selection Guide

| Situation | Algorithm |
|-----------|-----------|
| Any new binary problem — start here | Logistic Regression |
| Text data (spam, sentiment, news) | Naive Bayes |
| Need to explain the decision | Decision Tree |
| LR not good enough | Random Forest |
| High dimensions, complex boundary | SVM |
| Small dataset, low dimensions | KNN |

**Golden rule:** Always start simple. Complex models only justified when simple ones fail.

---

## Algorithms Needing Scaling

| Algorithm | Needs Scaling? | Why |
|-----------|---------------|-----|
| Logistic Regression | YES | Gradient descent converges faster |
| KNN | YES | Uses distance — unscaled features dominate |
| SVM | YES | Margin calculation uses distance |
| Decision Tree | No | Splits on thresholds, not distances |
| Random Forest | No | Collection of Decision Trees |
| Naive Bayes | No | Uses probabilities, not distances |

---

## Datasets Used

| Dataset | Problem | Notebook |
|---------|---------|----------|
| Pima Indians Diabetes (768 patients, 8 features) | Predict diabetes | 4.2-knn.ipynb |
| Chennai Bank loan data | Credit risk prediction | 4.2-knn-credit-risk-solved.ipynb |
| Bangalore IT company HR data | Employee attrition | 4.3-decision-tree.ipynb |
| Wayanad crop disease data | Crop disease classification | 4.4-random-forest.ipynb |
| Breast cancer biopsy (30 features) | Malignant vs benign | 4.5-svm.ipynb |
| SMS Spam Collection (5,572 messages) | Spam detection | 4.6-naive-bayes.ipynb + 4.7 project |

---

## Notes

- `diabetes.csv` — local copy of Pima Indians Diabetes dataset
- All notebooks follow the same structure: analogy → theory → from-scratch code → sklearn → visualisations → summary → practice task
- Real-world problems use Indian context throughout (₹, Kerala/Chennai/Bangalore settings)
