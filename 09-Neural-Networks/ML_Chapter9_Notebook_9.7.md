# Chapter 9.7 — Keras Basics: Building Networks in Code

## What Keras Is

You now understand how a neural network works from the inside — weighted sums, activations, forward pass, loss, backprop, optimiser. You could implement all of that from scratch in raw Python. But doing that for every project would take hundreds of lines just to build one network.

**Keras is the shortcut.** It's a high-level library that wraps all of that machinery. You describe *what* you want (3 layers, ReLU activations, Adam optimiser) and Keras handles the forward pass, the backprop chain rule, the weight updates — everything under the hood.

The analogy: in Chapter 5 you understood gradient descent fully from scratch. In Chapter 7 you used `GridSearchCV` — you didn't re-implement cross-validation yourself, you just called the tool. Keras is the same move, but for neural networks.

---

## The Four Core Keras Concepts

| Keras concept | What it maps to from theory |
|---|---|
| `Sequential` | The forward pass architecture — data flows left to right through layers |
| `Dense` | A fully connected layer — every neuron connects to every neuron in the previous layer |
| `compile` | Your three training decisions: optimiser (9.6) + loss function (9.4) + what to report |
| `fit` | The training loop: forward pass → loss → backprop → optimiser step, repeated N times |

---

## Step-by-Step Keras Workflow

### Step 1 — Split your data

Always split before building the model. You know this from Chapter 6.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

- **Training set (80%):** model learns from this — weights get updated here
- **Test set (20%):** model never sees this during training — honest final exam

The analogy: you study from a textbook (training set), then sit an exam with unseen questions (test set). Testing yourself on the same textbook questions only tells you how well you memorised, not how well you understood.

---

### Step 2 — Build the network

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(10,)),
    keras.layers.Dense(8,  activation='relu'),
    keras.layers.Dense(1,  activation='sigmoid')
])
```

**`Sequential`** — layers go in a straight line. Data flows forward through them in order.

**`Dense(16, activation='relu')`** — a fully connected hidden layer with 16 neurons and ReLU activation. "Fully connected" = every neuron sees all inputs from the previous layer. Exactly the junior-senior setup from 9.2.

**`input_shape=(10,)`** — tells Keras how many features each row has (e.g. 10 loan application columns: age, salary, credit score...). Only needed on the first layer — Keras infers the rest automatically.

**Activation choices and why:**
- Hidden layers → **ReLU** — derivative = 1 for z > 0, no vanishing gradient, deep networks can train (9.3)
- Output layer → **Sigmoid** — squashes to 0–1 probability, pairs cleanly with BCE loss via the cancellation from 9.4

---

### Step 3 — Check the architecture

```python
model.summary()
```

Always run this before training — it's a sanity check. Output:

```
Layer (type)          Output Shape       Param #
Dense (16 neurons)    (None, 16)         176    ← 10 inputs × 16 + 16 biases
Dense (8 neurons)     (None, 8)          136    ← 16 inputs × 8  + 8  biases
Dense (1 neuron)      (None, 1)          9      ← 8  inputs × 1  + 1  bias
Total params: 321
```

`None` in the output shape means "any batch size" — Keras handles batches of any size automatically.

---

### Step 4 — Compile

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

Three decisions you've already made in theory:
- **optimizer** → Adam (adaptive per-weight learning rate, 9.6)
- **loss** → binary_crossentropy (BCE — cancels sigmoid saturation cleanly, 9.4)
- **metrics** → what Keras prints during training (accuracy here; doesn't affect weight updates)

---

### Step 5 — Train with Early Stopping

```python
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',          # watch validation loss
    patience=10,                 # stop if no improvement for 10 consecutive epochs
    restore_best_weights=True    # rewind to the best epoch, not the last
)

history = model.fit(
    X_train, y_train,
    epochs=200,              # set high — early stopping cuts it short automatically
    batch_size=32,           # 32 random rows per gradient update (the "stochastic" in Adam)
    validation_split=0.2,    # holds out 20% of training data to monitor during training
    callbacks=[early_stop]
)
```

**`validation_split=0.2`** — Keras carves out 20% of training data as a validation set. After every epoch it checks loss on this slice. This is how you catch overfitting *during* training.

**Validation set vs test set:**
- Validation set → watched during training, used to make decisions (when to stop)
- Test set → never touched until the very end, gives the honest final score

**`restore_best_weights=True`** — without this, you get weights from the *last* epoch, which may be worse than epoch 47 where the model peaked. This rewinds automatically to the best checkpoint.

**`patience=10`** — exactly what your GNN used. Now you know what it means: "give the model 10 epochs to recover before giving up."

---

### Step 6 — Plot the training curve

```python
import matplotlib.pyplot as plt

plt.plot(history.history['loss'],     label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

This plot is the health check. Read it like this:

| What you see | What it means | Fix |
|---|---|---|
| Both losses falling together | Training well | Nothing |
| Train loss falls, val loss rises | Overfitting | Add dropout, reduce layers, more data |
| Both barely moving | Underfitting | Bigger network, higher learning rate |
| Val loss jumpy/noisy | Unstable training | Smaller learning rate, larger batch size |

---

### Step 7 — Evaluate on test set

```python
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")
```

`model.evaluate` runs one forward pass on the test set — **no weight updates happen here.** Pure evaluation. This is the number that matters. Training accuracy is what the model scored on its own study material. Test accuracy is the exam result.

---

### Step 8 — Make predictions

```python
# Returns probabilities (sigmoid output, between 0 and 1)
y_prob = model.predict(X_test)

# Convert to hard 0/1 predictions using 0.5 threshold
y_pred = (y_prob > 0.5).astype(int)

# Use your Chapter 2 metrics
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

`model.predict` runs the forward pass only — no backprop, no updates. Sigmoid outputs a probability; you apply the 0.5 threshold yourself to get hard predictions. From there, it's the same precision/recall/F1 from Chapter 2.

---

## Merits of Keras

- Clean, readable code — a full network in ~10 lines
- Hides the backprop and weight update math — you focus on architecture decisions
- `model.summary()` gives instant visibility into layer shapes and parameter counts
- `EarlyStopping` and other callbacks handle common training decisions automatically
- Runs on top of TensorFlow — production-ready, GPU support built in
- Easy to switch optimisers, loss functions, activations — just change the string

## Demerits of Keras

- Hides too much — if something goes wrong internally (NaN loss, exploding gradients), harder to debug than raw PyTorch
- Less flexible than PyTorch for custom architectures (e.g. your GNN used PyTorch Geometric, not Keras — heterogeneous graphs need lower-level control)
- `Sequential` API breaks down for complex topologies (multiple inputs, skip connections, branching) — need the Functional API instead
- Slightly slower than raw PyTorch for research/experimentation due to abstraction overhead

---

## Full Code — Everything in One Place

```python
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# 1. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Build
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(10,)),
    keras.layers.Dense(8,  activation='relu'),
    keras.layers.Dense(1,  activation='sigmoid')
])

# 3. Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 4. Check architecture
model.summary()

# 5. Train with early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)

# 6. Plot training curve
plt.plot(history.history['loss'],     label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 7. Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# 8. Predict
y_prob = model.predict(X_test)
y_pred = (y_prob > 0.5).astype(int)
print(classification_report(y_test, y_pred))
```

---

## The Takeaway

Keras is not new theory — it's 9.1 through 9.6 wrapped in clean code. Every line maps directly to something you already understand:

| Code | Theory it implements |
|---|---|
| `Dense(16, activation='relu')` | Weighted sum + ReLU (9.1, 9.3) |
| `activation='sigmoid'` on output | Sigmoid for binary probability (9.2, 9.3) |
| `loss='binary_crossentropy'` | BCE cancels sigmoid saturation (9.4) |
| `model.fit(...)` | Forward pass → backprop → optimiser, repeated (9.5, 9.6) |
| `optimizer='adam'` | Per-weight adaptive updates (9.6) |
| `EarlyStopping(patience=10)` | Stop when val loss stops improving (9.6, overfitting from Ch6) |
| `model.predict(...)` | Forward pass only, no weight updates (9.2) |

Next: **9.8 — Project: Loan Default Prediction** — where you build a real model on a Chennai bank dataset, using everything from 9.1 through 9.7 end to end.
