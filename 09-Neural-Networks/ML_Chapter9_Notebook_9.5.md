# Chapter 9.5 — Backpropagation: Chain Rule, Gradient Flow

## The Core Problem

After the forward pass, Ravi (the senior neuron) produces a prediction ŷ. The loss L tells us how wrong he is. But the network has dozens of weights — `wa`, `wb`, `wc` connecting the senior to his juniors, and `w1, w2, w3 ...` inside each junior connecting them to the raw inputs.

**The question:** how do we compute ∂L/∂w for *every* weight in the network — including weights that never directly touched the output?

## The Answer: Chain Rule, Applied Repeatedly

You already used chain rule in 9.4 to pull the gradient back through the sigmoid:

```
dL/dz = (dL/dŷ) × (dŷ/dz)
```

Backpropagation is just this — applied layer by layer, right to left, until every weight in the network has a gradient.

The key insight: **blame flows backward through the same weights that carried data forward**, scaled by how much each connection contributed.

---

## One Full Backward Pass — With Numbers

**Network:** 3 junior neurons → 1 senior neuron → loss

**Forward pass result:**
- Junior1 produced `h1` using weight `w1` on input `x1`
- Senior combined h1, h2, h3 using weights wa, wb, wc → produced ŷ
- Loss: `L = BCE(ŷ, y)` → we know from 9.4 that `∂L/∂z_out = ŷ − y`

### Step 1 — Start at the loss

```
∂L/∂z_out = ŷ − y
```

This is the error signal that kicks off the entire backward pass. (Derived in 9.4 — BCE + sigmoid cancel cleanly to give this.)

### Step 2 — Blame the senior's own weights (wa, wb, wc)

The senior computes `z_out = wa·h1 + wb·h2 + wc·h3 + b`. Taking the derivative with respect to wa:

```
∂z_out/∂wa = h1
```

So the full gradient for wa:
```
∂L/∂wa = ∂L/∂z_out × ∂z_out/∂wa = (ŷ − y) × h1
```

Interpretation: the bigger h1 was (the louder junior1 spoke), the more wa gets blamed for the mistake.

### Step 3 — Pass blame down to junior1

How much did h1 affect the loss? Through wa:
```
∂L/∂h1 = ∂L/∂z_out × ∂z_out/∂h1 = (ŷ − y) × wa
```

If wa was large (senior trusted junior1 a lot), junior1 receives a large blame signal.  
If wa was near zero (senior barely listened), junior1 gets almost no blame — regardless of how wrong the final answer was.

### Step 4 — Pull through junior1's activation (ReLU)

Junior1 computes `h1 = ReLU(z1)`. The derivative of ReLU:
```
ReLU′(z1) = 1 if z1 > 0, else 0
```

```
∂L/∂z1 = ∂L/∂h1 × ∂h1/∂z1 = (ŷ − y) × wa × ReLU′(z1)
```

If junior1 was in the "dead zone" (z1 ≤ 0), his ReLU derivative is 0 → no gradient passes through → his weights never update. This is the Dying ReLU problem from 9.3, seen from the backprop angle.

### Step 5 — Reach junior1's own weight (w1)

Junior1 computes `z1 = w1·x1 + ...`. Taking ∂z1/∂w1:
```
∂z1/∂w1 = x1
```

Final gradient for w1:
```
∂L/∂w1 = (ŷ − y) × wa × ReLU′(z1) × x1
```

---

## The Full Chain Visualised

```
Loss → senior → junior's activation → junior's weight
 ↑          ↑              ↑                  ↑
ŷ − y     × wa         × ReLU′(z1)         × x1
```

Each layer contributes exactly one local derivative. The network's gradient is all of them multiplied together.

---

## General Pattern — What Every Layer Does

| Term | What it is | Comes from |
|---|---|---|
| `ŷ − y` | How much loss changes with output | BCE + sigmoid (9.4 result) |
| `× wa` | How much senior leaned on h1 | The connection weight |
| `× ReLU′(z1)` | How much h1 changed with z1 | Junior's activation function |
| `× x1` | How much z1 changed with w1 | Junior's weighted sum formula |

For a deeper network (more hidden layers), you just keep multiplying. Each new layer adds:
- `× (weight connecting it forward)` — how much the next layer listened
- `× (its own activation derivative)` — whether the gradient can flow through

---

## Why Vanishing Gradient Happens Here

If every layer uses Sigmoid instead of ReLU, each `∂h/∂z` term contributes a sigmoid derivative ≤ 0.25.

In a 10-layer sigmoid network:
```
0.25¹⁰ = 0.000001
```

The gradient reaches the first layer as almost exactly zero — those early weights never update. This is why ReLU (derivative = 1 for z > 0) was the breakthrough that made deep networks trainable. Seen from backprop: ReLU lets the chain multiply by 1 instead of by 0.25, so the gradient arrives intact.

---

## The Takeaway

Backpropagation is not a new idea — it is the chain rule applied right-to-left, one layer at a time. Each layer receives a gradient from the layer ahead, multiplies by its own local derivative, and passes the result further back. By the end, every weight in the network has a `∂L/∂w` value.

What happens next with those gradients — how the weights actually get updated — is Chapter 9.6: Optimisers.
