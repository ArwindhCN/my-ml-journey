# Chapter 9.4 — Loss Functions: Binary Cross-Entropy vs MSE

## The Setup

Ravi (our neuron) takes a loan application, computes `z = Σwᵢxᵢ + b`, passes it through **Sigmoid**, and outputs a probability `ŷ` — e.g. "80% confident this person defaults."

Once we know the true outcome `y` (0 or 1), we need a **loss function** to score how wrong Ravi was, so gradient descent knows how to correct him.

A key fact we'll use throughout: the sigmoid's derivative, written in terms of its own output:

```
dŷ/dz = ŷ(1 − ŷ)
```

---

## Why MSE Breaks Down on Probabilities

MSE worked fine in Chapter 1 for continuous targets (like house prices). Does it work for a 0–1 probability output? Let's test it on a bad case.

**Scenario:** true label `y = 1` (applicant really did default), but Ravi is confidently *wrong*: `ŷ = 0.01`.

- **Error:** `(ŷ − y)² = (0.01 − 1)² ≈ 0.98` — a big number, correctly flagging a bad mistake.
- **But the gradient that actually updates Ravi's weights** isn't just the error — it's the error multiplied by the sigmoid's derivative at that point (`dŷ/dz`).

At `ŷ = 0.01`:
```
dŷ/dz = 0.01 × 0.99 = 0.0099
```

That's tiny. When Ravi is sitting near the extreme ends of the sigmoid curve (very confident, right or wrong), the curve is nearly flat — so its slope is nearly zero.

**The actual gradient using MSE:**
```
dL/dŷ = 2(ŷ − y) = 2(0.01 − 1) = −1.98
dL/dz = dL/dŷ × dŷ/dz = −1.98 × 0.0099 ≈ −0.0196
```

**The problem:** Ravi is making his *worst possible mistake* here, and the correction he receives is almost nothing. The gradient gets crushed exactly when it's needed most. This is sigmoid *saturation* — the same vanishing-gradient mechanism from 9.3, just triggered by how extreme the prediction is, rather than by network depth.

---

## Binary Cross-Entropy (BCE) — a.k.a. Log Loss

```
L = −[y·ln(ŷ) + (1−y)·ln(1−ŷ)]
```

Same name, two terms you'll see used interchangeably everywhere (papers, Keras, sklearn): **Binary Cross-Entropy** and **Log Loss**.

### Does BCE fix the problem? Full derivation.

**Step 1 — Take dL/dŷ:**
```
dL/dŷ = −y/ŷ + (1−y)/(1−ŷ)
```

**Step 2 — Chain rule through the sigmoid** (multiply by `dŷ/dz = ŷ(1−ŷ)`):

```
dL/dz = [−y/ŷ + (1−y)/(1−ŷ)] × ŷ(1−ŷ)

       = −y/ŷ × ŷ(1−ŷ)   +   (1−y)/(1−ŷ) × ŷ(1−ŷ)

       = −y(1−ŷ)          +   (1−y)ŷ

       = −y + yŷ          +   ŷ − yŷ

       = ŷ − y
```

**Result:**
```
dL/dz = ŷ − y
```

The `ŷ` and `(1−ŷ)` denominators from BCE's own derivative cancel exactly against the `ŷ(1−ŷ)` from the sigmoid. **Nothing saturating survives.** The gradient is simply "how wrong Ravi is" — no shrinkage at the extremes.

### Checking it with numbers (same case: y=1, ŷ=0.01)

```
dL/dŷ = −1/0.01 + 0/0.99 = −100
dŷ/dz = 0.01 × 0.99 = 0.0099
dL/dz = −100 × 0.0099 = −0.99
```

Shortcut check: `ŷ − y = 0.01 − 1 = −0.99` ✓ — matches exactly.

### Side-by-side comparison

| Loss | Gradient at z (∂L/∂z) | Behavior |
|---|---|---|
| MSE | ≈ −0.0196 | Crushed by sigmoid saturation |
| BCE | −0.99 | Full-strength, proportional to actual error |

BCE's gradient is **~50x larger** for the exact same mistake. MSE gets choked by the `0.0099` sigmoid-slope factor; BCE's `−y/ŷ` term blows up by exactly enough to cancel that choke.

---

## The Takeaway

- **MSE** is the right loss for continuous targets (house prices, etc.) — Chapter 1 territory.
- **BCE (Log Loss)** is the right loss for binary probability outputs — its derivative is algebraically engineered to cancel the sigmoid's saturation, so `∂Loss/∂z = ŷ − y` always, no matter how confidently right or wrong the prediction is.
- This clean `ŷ − y` result isn't a coincidence — it's *why* BCE + Sigmoid is the standard pairing for binary classification, and it's also the gradient expression that backpropagation (9.5) will build on directly.

## For multi-class problems

Same idea generalizes to **Categorical Cross-Entropy**, paired with **Softmax** instead of Sigmoid (the multi-class output activation from 9.3). Same cancellation trick applies — covered when it becomes relevant.
