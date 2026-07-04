# Chapter 9.6 — Optimisers: SGD vs Adam

## The Setup — What an Optimiser Actually Does

Backprop (9.5) gave us `∂L/∂w` for every weight — the direction and size of the mistake. The optimiser's job is to decide: **given this gradient, how exactly do we update the weight?**

The base formula from Chapter 5:
```
w = w − η · ∂L/∂w
```

Every optimiser builds on this. The difference is *how they compute the effective step* before applying it.

---

## The Analogy: Navigating a Hilly Landscape in the Dark

Imagine you're trying to find the lowest valley in a hilly landscape — but it's pitch dark. All you can feel is the slope under your feet right now (that's your gradient). You're trying to reach the bottom (minimum loss).

**SGD** = walking downhill with a fixed step size every time, feeling only the ground directly under your feet.

**Adam** = walking downhill, but you carry a notebook. You've been recording every slope you've felt so far. You use that history to decide: "this direction has been consistently steep — slow down. That direction has barely moved — push harder."

---

## 1. SGD — Stochastic Gradient Descent

### What it does

The "stochastic" part means instead of computing the gradient over the entire dataset (slow), it picks a random mini-batch each step and computes the gradient on that. Faster, but noisier.

```
w = w − η · ∂L/∂w
```

One learning rate. One formula. Applied identically to every weight, every step.

### SGD with Momentum (the common version)

Plain SGD reacts only to the current gradient — like someone who forgets everything they felt a second ago. **Momentum** gives SGD a memory:

```
v = β · v_prev + (1 − β) · ∂L/∂w     # running average of gradients
w = w − η · v
```

`β` is typically 0.9 — meaning 90% of last step's direction carries forward. Like a ball rolling downhill that builds up speed in a consistent direction and resists getting knocked sideways by noise.

### Merits
- Simple and well understood
- With careful tuning, matches or beats Adam near convergence
- More predictable behaviour — good for research and reproducibility
- Generalises slightly better than Adam in some vision tasks (well-documented finding)

### Demerits
- Same learning rate for every weight — a poor fit for networks where different weights live in very different loss landscapes
- Requires manual tuning of η and β — bad choice and training stalls or diverges
- Slow on flat regions, prone to overshooting on steep ones
- "Stochastic" noise can make it unstable without momentum

---

## 2. Adam — Adaptive Moment Estimation

### The Analogy

Adam is like a navigator who keeps two notebooks:

- **Notebook 1 (m):** "What direction have I been going recently on average?" → smooths out noisy gradient directions
- **Notebook 2 (v):** "How wild/large have my steps been recently?" → scales down aggressive weights, scales up stalled ones

At each step:
```
m = β₁ · m_prev + (1 − β₁) · ∂L/∂w        # gradient direction average (β₁ ≈ 0.9)
v = β₂ · v_prev + (1 − β₂) · (∂L/∂w)²     # gradient magnitude average (β₂ ≈ 0.999)

m̂ = m / (1 − β₁ᵗ)    # bias correction (important early in training)
v̂ = v / (1 − β₂ᵗ)

w = w − η · m̂ / (√v̂ + ε)    # ε ≈ 1e-8, prevents division by zero
```

### What each piece does

| Term | Notebook analogy | Effect |
|---|---|---|
| `m` (1st moment) | "Which direction have I mostly been going?" | Smooths noisy gradients |
| `v` (2nd moment) | "How violently has this weight been moving?" | Scales the step size per weight |
| `m̂ / √v̂` | "Adjusted direction given my full history" | Per-weight adaptive step |
| `weight_decay` | "Slowly forget large weights" | L2 regularisation, prevents overfitting |

### The step size logic (from our conversation)

- Large `v` (weight has been getting big, consistent gradients) → divide by large √v → **smaller step** → cautious in steep regions
- Small `v` (weight has been getting tiny gradients) → divide by small √v → **larger step** → pushes through flat regions

**Caveat:** Adam cannot tell the difference between "tiny gradient because I'm near the minimum" vs "tiny gradient because I'm on a plateau far from the minimum." In the first case, its larger step can overshoot. This is Adam's most well-known weakness near final convergence.

### The Hybrid Fix

Some training pipelines exploit this:
```
Phase 1 (early): Adam   → fast progress across the loss landscape
Phase 2 (late):  SGD    → careful, precise steps near the minimum
```

### Merits
- Per-weight adaptive learning rates — no manual tuning per layer
- Handles sparse gradients well (useful in NLP, graphs — e.g. your GNN)
- Fast convergence in practice — the default choice for most deep learning
- Robust to learning rate choice — works across a wide range of η values

### Demerits
- Can overshoot near the minimum (see above — the flat-vs-close ambiguity)
- Known to generalise slightly worse than well-tuned SGD on some image benchmarks
- More memory: stores `m` and `v` for every weight (2× the parameters in memory)
- `weight_decay` inside Adam behaves differently from true L2 reg (AdamW fixes this — the version used in most modern transformers)

---

## SGD vs Adam — Head to Head

| | SGD (with momentum) | Adam |
|---|---|---|
| Learning rate | One for all weights | Per-weight, adaptive |
| Memory cost | Low | 2× (stores m and v) |
| Tuning needed | High (η, β both matter) | Low (η only, roughly) |
| Speed to converge | Slower | Faster |
| Final accuracy | Often slightly better | Slightly worse sometimes |
| Best for | CV/vision, fine-tuning | NLP, graphs, general use |
| Your GNN used | — | ✅ Adam + weight_decay |

---

## Code — Keras (SGD vs Adam)

```python
import tensorflow as tf
from tensorflow import keras

# --- Build a simple binary classifier (same structure as 9.7) ---
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(10,)),
    keras.layers.Dense(8,  activation='relu'),
    keras.layers.Dense(1,  activation='sigmoid')
])

# --- Option 1: SGD with momentum ---
sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

# --- Option 2: Adam ---
adam = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

# --- Option 3: Adam with weight decay (AdamW) ---
# Keras has AdamW built in from TF 2.11+
adamw = keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-5)
model.compile(optimizer=adamw, loss='binary_crossentropy', metrics=['accuracy'])
```

---

## Code — PyTorch (SGD vs Adam)

```python
import torch
import torch.nn as nn
import torch.optim as optim

# --- Build the same network in PyTorch ---
model = nn.Sequential(
    nn.Linear(10, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

# --- Option 1: SGD with momentum ---
sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# --- Option 2: Adam ---
adam = optim.Adam(model.parameters(), lr=0.001)

# --- Option 3: Adam with weight decay (what your GNN used) ---
adam_wd = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# --- Training loop skeleton ---
criterion = nn.BCELoss()

for epoch in range(100):
    model.train()

    # Forward pass
    y_pred = model(X_train).squeeze()
    loss = criterion(y_pred, y_train)

    # Backward pass
    optimizer.zero_grad()   # clear gradients from last step
    loss.backward()         # backprop — computes ∂L/∂w for every weight
    optimizer.step()        # apply the update: w = w − η · (Adam or SGD formula)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Key PyTorch lines to understand

```python
optimizer.zero_grad()   # PyTorch accumulates gradients by default — must clear each step
loss.backward()         # this IS backpropagation — runs the chain rule automatically
optimizer.step()        # this IS the optimiser — applies w = w − η · update
```

These three lines together are the entire training loop. Everything in 9.4 (loss), 9.5 (backprop), and 9.6 (optimiser) lives inside these three calls.

---

## Connection to Your GNN Research

In `train_hetero_gnn.py` you had:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
```

Now you know exactly what each part does:
- `Adam` → per-weight adaptive steps, good for heterogeneous graphs where different node types have very different gradient scales
- `lr=0.001` → global starting learning rate, Adam adapts per weight from here
- `weight_decay=1e-5` → L2 regularisation baked into every update, helping prevent your HANConv weights from exploding (alongside your early stopping)

---

## The Takeaway

Backprop gives you the gradient. The optimiser decides what to *do* with it. SGD is the simple, honest workhorse — same rule for everyone. Adam is the adaptive navigator — it reads the history of every weight and adjusts accordingly. For most deep learning projects, especially graphs and NLP, Adam is the default. For careful fine-tuning near convergence, SGD often wins.

Next: **9.7 Keras basics** — where you'll put loss functions, backprop, and optimisers together into actual working code for the first time.
