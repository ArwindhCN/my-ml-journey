# Chapter 9.8 — Project: Loan Default Prediction (Chennai Bank Dataset)

## The Problem

A Chennai bank has been approving and rejecting loans manually for years. Your job: build a neural network that looks at an applicant's details and predicts — **will this person default or not?**

This is a binary classification problem. Features include income, age, loan amount, credit history, employment type. Target: `default` (1 = defaulted, 0 = repaid).

This project ties together everything from Chapter 9:
- Perceptron & forward pass (9.1, 9.2) → the network architecture
- Activation functions (9.3) → ReLU in hidden layers, Sigmoid at output
- Loss functions (9.4) → Binary Cross-Entropy
- Backpropagation (9.5) → runs automatically inside `model.fit`
- Optimisers (9.6) → Adam with weight decay
- Keras basics (9.7) → the full workflow

---

## Preprocessing Decisions — Made Before Touching Code

| Feature type | Treatment | Why |
|---|---|---|
| Categorical (`employment_type`) | One-Hot Encoding | Label encoding implies false ordering (unemployed > self-employed is meaningless) |
| Numerical (`income`, `age`, `loan_amount` etc.) | Standard scaling (z-score) | Stops large-valued features dominating weights in the weighted sum |
| Target (`default`) | Nothing | Already 0/1 |

**Two rules that must not be broken:**
1. **Split before scaling** — if you scale on the full dataset first, test set statistics leak into training. The scaler must be fit on training data only, then applied to both.
2. **Stratify the split** — preserves the 0/1 ratio in both train and test sets. Without this, an unlucky split could put most defaults in one set.

---

## Stage 1 — Load & Explore

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('chennai_loan_data.csv')

# First look
print(df.shape)
print(df.head())
print(df.info())

# Class balance — critical first check
print(df['default'].value_counts())
print(df['default'].value_counts(normalize=True))
```

**Why check class balance first?** If 95% of applicants repaid and only 5% defaulted, a model that always predicts "no default" gets 95% accuracy without learning anything. Accuracy alone is misleading on imbalanced data — you need precision, recall, and F1 (Chapter 2).

```python
# Visualise class balance
df['default'].value_counts().plot(kind='bar', color=['steelblue', 'salmon'])
plt.title('Class Distribution — Repaid vs Defaulted')
plt.xticks([0, 1], ['Repaid (0)', 'Defaulted (1)'], rotation=0)
plt.ylabel('Count')
plt.show()

# Distribution of key numerical features
df[['income', 'age', 'loan_amount']].hist(bins=30, figsize=(12, 4))
plt.suptitle('Feature Distributions')
plt.tight_layout()
plt.show()
```

---

## Stage 2 — Preprocess

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features and target
X = df.drop('default', axis=1)
y = df['default']

# One-Hot Encode categorical columns
X = pd.get_dummies(X, columns=['employment_type'], drop_first=False)

# Check result
print(X.columns.tolist())
print(X.shape)

# Split BEFORE scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # preserves class balance in both splits
)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# Scale AFTER splitting — fit on train only
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # learns mean and std from training data
X_test  = scaler.transform(X_test)        # applies same scale, no new learning
```

---

## Stage 3 — Build the Network

```python
import tensorflow as tf
from tensorflow import keras

n_features = X_train.shape[1]

model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(n_features,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1,  activation='sigmoid')
])
```

**Architecture logic — the funnel:**
- Wide at the start (32 neurons) to capture many raw patterns from the input features
- Narrows in the middle (16 neurons) as the network builds abstractions
- Single output neuron with Sigmoid — outputs probability of default between 0 and 1

This is the junior-senior hierarchy from 9.2, just two levels deep instead of one.

**Activation choices:**
- Hidden layers → ReLU — derivative = 1 for z > 0, gradient flows intact through backprop (9.3)
- Output layer → Sigmoid — squashes to 0–1 probability, pairs with BCE loss (9.4)

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

---

## Stage 4 — Train

```python
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True   # rewind to best epoch, not last
)

history = model.fit(
    X_train, y_train,
    epochs=200,              # set high — early stopping cuts it short
    batch_size=32,
    validation_split=0.2,   # 20% of training data held out to monitor during training
    callbacks=[early_stop]
)

print(f"Training stopped at epoch: {len(history.history['loss'])}")
```

### Plot the training curve

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['loss'],     label='Train loss')
ax1.plot(history.history['val_loss'], label='Val loss')
ax1.set_title('Loss Curve')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(history.history['accuracy'],     label='Train accuracy')
ax2.plot(history.history['val_accuracy'], label='Val accuracy')
ax2.set_title('Accuracy Curve')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.tight_layout()
plt.show()
```

**How to read the loss curve:**

| What you see | What it means | Fix |
|---|---|---|
| Both losses falling together | Training well | Nothing |
| Train loss falls, val loss rises | Overfitting | Add dropout, reduce layers |
| Both barely moving | Underfitting | Bigger network, higher learning rate |
| Val loss jumpy/noisy | Unstable training | Smaller learning rate, larger batch size |

---

## Stage 5 — Evaluate

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Test set score
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy:  {test_acc:.4f}")
print(f"Test loss:      {test_loss:.4f}")

# Probabilities → hard predictions
y_prob = model.predict(X_test).flatten()
y_pred = (y_prob > 0.5).astype(int)

# Full classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Repaid', 'Defaulted']))

# ROC-AUC — threshold-independent quality measure
auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {auc:.4f}")
```

**Why ROC-AUC alongside accuracy?** AUC measures how well the model *ranks* applicants by risk — regardless of where you set the threshold. A score of 1.0 = perfect separation. 0.5 = random guessing. On imbalanced datasets, AUC is often more informative than accuracy.

```python
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred: Repaid', 'Pred: Defaulted'],
            yticklabels=['True: Repaid', 'True: Defaulted'])
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
```

---

## Stage 6 — Analyse Where It Fails

```python
results = pd.DataFrame({
    'true_label':  y_test.values,
    'predicted':   y_pred,
    'probability': y_prob
})

# False Negatives — predicted "will repay" but actually defaulted
# Most dangerous for the bank — approved loans that go bad
false_negatives = results[(results['true_label'] == 1) & (results['predicted'] == 0)]
print(f"False Negatives (missed defaults):      {len(false_negatives)}")

# False Positives — predicted "will default" but actually repaid
# Lost business — good customers wrongly rejected
false_positives = results[(results['true_label'] == 0) & (results['predicted'] == 1)]
print(f"False Positives (good customers rejected): {len(false_positives)}")

# Most uncertain predictions (near the decision boundary)
uncertain = results[(results['probability'] > 0.4) & (results['probability'] < 0.6)]
print(f"Uncertain predictions (0.4–0.6 range):  {len(uncertain)}")
```

**Why this analysis matters for the Chennai bank:**
- A False Negative (missed default) = bank approves a loan that goes bad → financial loss
- A False Positive (rejected good customer) = lost business, reputation risk

These two errors have very different costs. The bank may decide to lower the threshold from 0.5 to 0.3 — catching more defaults at the cost of rejecting more good customers. That's a business decision, not a model decision.

---

## Stage 7 — Threshold Tuning (Business Decision)

```python
# Try different thresholds and see how precision/recall shifts
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
print("-" * 48)

from sklearn.metrics import precision_score, recall_score, f1_score

for t in thresholds:
    y_pred_t = (y_prob > t).astype(int)
    p = precision_score(y_test, y_pred_t, zero_division=0)
    r = recall_score(y_test, y_pred_t, zero_division=0)
    f = f1_score(y_test, y_pred_t, zero_division=0)
    print(f"{t:<12.1f} {p:<12.4f} {r:<12.4f} {f:<12.4f}")
```

Lower threshold → higher recall (catches more defaults) → lower precision (more false alarms).
Higher threshold → higher precision (fewer false alarms) → lower recall (misses more defaults).
The bank chooses where to sit on this tradeoff based on the relative cost of each error.

---

## The Takeaway

This project is Chapter 9 end to end, in one pipeline:

| Stage | Chapter 9 concept it uses |
|---|---|
| One-Hot Encoding, scaling | Chapter 2 + Chapter 8 preprocessing |
| Network architecture (32→16→1) | Forward pass, junior-senior layers (9.2) |
| ReLU hidden, Sigmoid output | Activation functions (9.3) |
| `loss='binary_crossentropy'` | BCE cancels sigmoid saturation (9.4) |
| `model.fit` running 200 epochs | Backpropagation under the hood (9.5) |
| `optimizer=Adam` | Adaptive per-weight updates (9.6) |
| `EarlyStopping(patience=10)` | Keras callbacks (9.7) |
| Precision, recall, confusion matrix | Chapter 2 classification metrics |
| Threshold tuning | Business framing around model outputs |

**Chapter 9 complete.** Next: Chapter 10 — Deep Learning Essentials (Dropout, Batch Normalisation, Embeddings, PyTorch fundamentals).
