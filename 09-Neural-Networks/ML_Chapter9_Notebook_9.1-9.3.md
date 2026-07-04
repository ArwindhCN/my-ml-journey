# Machine Learning — Chapter 9: Neural Networks
## Notebook: 9.1 · 9.2 · 9.3
### The Perceptron · The Forward Pass · Activation Functions

> Written so that anyone reading this — with a basic ML background — walks away with the same understanding. Every concept is built from first principles, with the real math, the real intuition, and the real "why."

---

## 9.1 — The Perceptron

### What is a Perceptron?

A perceptron is the **smallest possible neural network** — a single artificial neuron. Before touching any math, let's understand what it actually does using a real-world analogy.

---

### The Ravi Analogy

Imagine **Ravi**, a loan officer at a Chennai bank. When a customer sits across from him and asks for a loan, Ravi does the following:

1. **Gathers signals** — he checks the customer's income, loan repayment history, and bank balance
2. **Weighs each signal differently** — he doesn't treat all three equally. In his experience, past loan repayment history is the strongest predictor of future default. So he mentally gives it more importance than income or bank balance.
3. **Computes a total score** — he combines all three signals, weighted by their importance, into one number
4. **Makes a decision** — if the total score crosses a threshold, he approves. Otherwise, he rejects.

A perceptron does exactly this — and nothing more.

---

### The Math

**Step 1 — Weighted sum**

Each input is multiplied by its weight and added together, plus a bias term:

```
z = w₁x₁ + w₂x₂ + w₃x₃ + b
```

| Symbol | Meaning | Ravi equivalent |
|---|---|---|
| x₁, x₂, x₃ | Input features | Income, loan history, bank balance |
| w₁, w₂, w₃ | Weights | Importance Ravi gives each signal |
| b | Bias | Ravi's baseline suspicion (explained below) |
| z | Weighted sum | Ravi's total score before decision |

**Concrete example:**

Ravi assigns: w₁=3 (income), w₂=5 (loan history), w₃=2 (balance)

Customer walks in: income score=8, history score=2, balance score=6

```
z = (3×8) + (5×2) + (2×6)
z = 24 + 10 + 12
z = 46
```

---

**Step 2 — The Bias Term**

The bias `b` is Ravi's **baseline suspicion** — his default lean, completely independent of any input. Even if a customer had a perfect score on all three signals, Ravi might still be slightly cautious based on macroeconomic conditions. That default caution is the bias. It shifts the entire weighted sum up or down regardless of what the inputs are.

Without bias, the network can only draw decision boundaries that pass through the origin — a severe limitation. Bias gives the network the flexibility to shift those boundaries anywhere.

---

**Step 3 — The Activation Function (turning z into a decision)**

`z` is just a raw number — could be -200, could be 846. We need to convert it into something meaningful: either a decision or a probability.

**The original approach — Step Function:**

The 1950s perceptron used a step function:
```
output = 1   if z > threshold
output = 0   if z ≤ threshold
```

Ravi picks a threshold of 40. Score is 46 → approve. Simple.

**Why the step function was abandoned:**

Training a neural network requires **gradient descent** — we nudge each weight in the direction that reduces the loss. But to nudge correctly, we need to know: "if I change this weight by a tiny amount, does the output change?"

With a step function, the answer is almost always **no**. The output stays frozen at 0 or 1. The gradient is zero everywhere. Gradient descent has nothing to grip. The network cannot learn.

**The fix — Sigmoid:**

```
σ(z) = 1 / (1 + e⁻ᶻ)
```

Sigmoid is a smooth S-curve. It maps any z — no matter how large or small — to a number between 0 and 1. Nudge a weight by 0.0001, and the output changes by a tiny but nonzero amount. The gradient is always defined. Gradient descent can always learn.

Output now means: "probability of loan default." ŷ = 0.73 = 73% chance of default. You then apply a threshold (usually 0.5) separately to make the final yes/no call.

---

### How Do Weights Get to the Right Values?

Ravi didn't know from day one that loan history deserved w=5. He learned it. The mechanism is **gradient descent** — same as Chapter 5:

```
w = w − η · ∂Loss/∂w
```

In English: compute how wrong the prediction was (the loss), find which direction each weight should move to reduce that loss (the gradient), nudge every weight a small step in that direction (learning rate η), repeat thousands of times.

Every time Ravi approved a loan that defaulted because he underweighted loan history, gradient descent nudged w₂ upward. Over thousands of customers, w₂ converged to 5.

---

### 9.1 Summary — The Full Perceptron

```
Input features (x)
    ↓
Weighted sum:  z = Σ(wᵢxᵢ) + b
    ↓
Activation:    ŷ = σ(z)
    ↓
Output (probability)
    ↓
Loss (how wrong were we?)
    ↓
Gradient descent (adjust every weight)
    ↓
Repeat
```

One perceptron = one Ravi = one decision. To handle complex, real-world patterns, we need many Ravis — that's the next section.

---

---

## 9.2 — The Forward Pass

### Why One Ravi Isn't Enough

Ravi can draw one straight line between "approve" and "reject." But real customers are messy. A customer with high income AND a terrible loan history is a completely different risk from someone with medium scores on both. One straight line can't capture these complex, curved patterns.

Solution: **many Ravis, organised in layers.**

---

### The Two-Floor Bank

Imagine the bank has two floors:

**Ground floor — Junior officers (hidden layer):**
- Three Ravis, each specialising in a different angle of risk
- Ravi A: credit risk specialist — pays extra attention to loan history
- Ravi B: income risk specialist — focuses on income stability
- Ravi C: asset risk specialist — focuses on bank balance and collateral
- **Crucially:** each junior sees ALL the raw inputs, not just "their" signal. Ravi A still glances at income and balance — he just weights loan history more heavily. Same data, different weights.

**First floor — Senior officer (output layer):**
- Never touches the raw customer data at all
- His inputs are the junior officers' conclusions: h₁, h₂, h₃
- He synthesises those intermediate judgements into one final decision

This is why neural networks are powerful — each layer builds on abstractions from the previous layer, not on raw data.

---

### The Math of Stacking Layers

**Ground floor (hidden layer) — three juniors:**

```
z₁ = w₁·x₁ + w₂·x₂ + w₃·x₃ + b₁    →    h₁ = ReLU(z₁)
z₂ = w₄·x₁ + w₅·x₂ + w₆·x₃ + b₂    →    h₂ = ReLU(z₂)
z₃ = w₇·x₁ + w₈·x₂ + w₉·x₃ + b₃    →    h₃ = ReLU(z₃)
```

Each junior does the same two steps: weighted sum → activation. h₁, h₂, h₃ are their outputs.

**First floor (output layer) — the senior:**

```
z_out = w_a·h₁ + w_b·h₂ + w_c·h₃ + b_out
ŷ = σ(z_out)
```

The senior's inputs are h₁, h₂, h₃ — junior outputs, not raw x values. Same neuron formula as always — it's just that the "inputs" are now the previous layer's activations.

**Adding a 4th junior (h₄)?** Just one more term:
```
z_out = w_a·h₁ + w_b·h₂ + w_c·h₃ + w_d·h₄ + b_out
```

**Important:** Bias is **one per layer**, not one per neuron. The senior has one bias regardless of how many juniors feed into him.

---

### The Forward Pass — Left to Right, No Loops

The full left-to-right flow — raw inputs → hidden layer → output layer → prediction — is called the **forward pass**.

```
Raw inputs (x₁, x₂, x₃)
         ↓
Hidden layer: each junior computes z → applies ReLU → outputs h
         ↓
Output layer: senior computes z_out → applies Sigmoid → outputs ŷ
         ↓
Final prediction (probability of default)
```

Data flows in one direction only. No loops. No going backward. That's what makes it "forward."

---

### What About Going Backward?

Forward pass gives us ŷ. We compare ŷ to the true label y, compute a loss (how wrong we were), then work backward through the network assigning blame to every weight. That backward process is backpropagation — covered in depth in section 9.5. The forward pass is just the left-to-right prediction step.

---

---

## 9.3 — Activation Functions

### The Problem Without Activation Functions

Before explaining what activation functions are, let's understand why removing them breaks everything.

**Without activation, stacking layers collapses into a single linear equation:**

Junior computes: `z₁ = w₁x₁ + w₂x₂ + b₁`

Senior takes that and computes: `z₂ = w₃z₁ + b₂`

Substitute z₁ into z₂:
```
z₂ = w₃(w₁x₁ + w₂x₂ + b₁) + b₂
z₂ = (w₃w₁)x₁ + (w₃w₂)x₂ + (w₃b₁ + b₂)
```

That's just another straight line. Stack 100 layers with no activation — still one straight line. You could replace 100 Ravis with one Ravi and get the identical result. No depth, no power.

**Activation functions break this collapse** by wrapping z in a nonlinear function before passing it forward. Now the senior can't substitute and simplify. The layers stay genuinely separate.

---

### Three Reasons We Need Activation Functions

---

#### Reason 1 — Prevent Layer Collapse

As shown above: without activation, depth is an illusion. The entire network collapses into a single linear equation regardless of how many layers you add.

Activation functions make each layer's output irreducibly nonlinear, so stacking layers actually gives you more expressive power.

---

#### Reason 2 — Model Nonlinear Patterns in Real Data

A straight line can only draw one decision boundary. Real data is curved, twisted, irregular. Consider:
- A customer with **medium income AND medium loan history** → defaults
- A customer with **very high income AND very bad loan history** → doesn't default

No single straight line can separate these groups. You need a curved boundary.

**How ReLU creates curves — the actual math:**

ReLU on a single neuron is simple: `max(0, z)`. Flat zero on the left, straight slope on the right. Not very exciting alone.

But look at what happens when you combine **two ReLU neurons with different biases:**

- Neuron A: `hA = max(0, x)` — fires at x=0
- Neuron B: `hB = max(0, x−3)` — fires at x=3 (shifted by bias)
- Senior output: `hA + hB`

| x | hA = max(0, x) | hB = max(0, x−3) | output = hA + hB | slope |
|---|---|---|---|---|
| −2 | 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 | 0 |
| 1 | 1 | 0 | 1 | 1 |
| 2 | 2 | 0 | 2 | 1 |
| 3 | 3 | 0 | 3 | 1 |
| 4 | 4 | 1 | 5 | 2 |
| 5 | 5 | 2 | 7 | 2 |

At x=3, neuron B switches on. The slope of the combined output **changes from 1 to 2**. That change in slope is the **kink** — the nonlinearity.

Two neurons = one kink. 100 neurons with different biases = 100 kinks, each occurring at a different x value. Enough kinks close together can **approximate any curve** you want.

This is actually a famous result called the **Universal Approximation Theorem**: a neural network with enough neurons can approximate any function, no matter how complex.

The key insight: **the bend doesn't come from one ReLU — it comes from many ReLUs switching on at different points across the input space.**

---

#### Reason 3 — Keep Gradients Alive (Why ReLU Replaced Sigmoid in Hidden Layers)

During backpropagation, gradients flow backward through every layer. At each neuron, the gradient gets multiplied by the neuron's derivative.

**Sigmoid's derivative:**
```
σ'(z) = σ(z) · (1 − σ(z))
```
Maximum value: when σ(z) = 0.5 → 0.5 × 0.5 = **0.25**

So sigmoid's derivative is always ≤ 0.25. Every hidden layer multiplies the gradient by at most 0.25.

In a 10-layer network:
```
gradient at early layer = original_gradient × 0.25¹⁰ = 0.000001
```

The gradient essentially dies. Early weights receive near-zero updates. Early layers stop learning. This is called the **vanishing gradient problem** — it's why deep networks were nearly impossible to train before ~2010.

**ReLU's derivative:**
```
ReLU'(z) = 1   for z > 0
           0   for z < 0
```

For positive z, the gradient passes through **unchanged** (multiplied by 1). No shrinkage. The gradient arrives at early layers with full strength, regardless of how deep the network is.

This is the single biggest reason ReLU enabled modern deep learning — it made genuinely deep networks trainable.

---

### Where Activation Sits in Every Neuron

Every neuron in the network does this exact two-step — hidden layers and output layer alike, no exceptions:

```
Step 1:  z = w·x + b          ← weighted sum (always linear)
Step 2:  h = activation(z)    ← nonlinearity applied to z
Step 3:  h passed to next layer as input
```

The activation function is the only thing that makes neural networks more than fancy linear regression.

---

### The Three Activation Functions You Must Know

**1. Step Function (historical, now abandoned)**
```
f(z) = 1   if z > 0
       0   if z ≤ 0
```
Hard binary jump. Gradient = 0 everywhere. Cannot train with gradient descent. Used in the original 1950s perceptron. Abandoned.

---

**2. Sigmoid**
```
σ(z) = 1 / (1 + e⁻ᶻ)
```
Smooth S-curve. Output always between 0 and 1. Derivative ≤ 0.25 — causes vanishing gradient in deep networks. Used at the **output layer for binary classification** (yes/no) because it naturally outputs a probability.

Output: ŷ = 0.73 → "73% probability of loan default." The network outputs a probability — you apply a threshold (e.g. 0.5) separately to make the final binary decision.

---

**3. ReLU (Rectified Linear Unit)**
```
ReLU(z) = max(0, z)
```
Dead flat for negative z, straight slope for positive z. Derivative = 1 for z > 0. No gradient shrinkage. Used at **all hidden layers** as the default choice. Enabled deep networks.

**Dying ReLU problem:** if a neuron's z is always negative, its output is always 0, its gradient is always 0, and it never updates. The neuron is effectively dead. Fix: Leaky ReLU uses `max(0.01z, z)` instead, giving a tiny nonzero gradient even for negative z.

---

**4. Softmax (for multi-class output)**
```
softmax(zᵢ) = e^zᵢ / Σ e^zⱼ
```
Takes N raw scores and converts them into N probabilities that sum to exactly 1. Used at the **output layer for multi-class classification**.

Example — 5 loan categories:
```
Raw scores:    personal=2.1, home=0.5, auto=3.3, education=0.9, business=1.2
After Softmax: personal=0.14, home=0.03, auto=0.57, education=0.06, business=0.18
               ↑ all add up to exactly 1.0
```
Network picks highest — auto loan, 57% confident.

---

### The Practical Rule — Which Activation Where

| Layer | Activation | Why |
|---|---|---|
| Any hidden layer | **ReLU** | Derivative = 1, gradient stays full strength through deep networks |
| Output — binary (yes/no) | **Sigmoid** | Outputs smooth probability 0–1 for binary classification |
| Output — multi-class (N categories) | **Softmax** | Outputs N probabilities summing to 1 |
| Hidden layers (if ReLU neurons keep dying) | **Leaky ReLU** | Small nonzero gradient for negative z, prevents dead neurons |

---

### Tying It All Together

The three reasons for activation functions are really three angles on the same truth:

**Activation functions are what make neural networks more than fancy linear regression.**

Without them:
- Layers collapse → no depth
- Only straight-line boundaries → can't model real patterns
- Gradients vanish → deep networks can't train

With ReLU in hidden layers and Sigmoid/Softmax at the output:
- Layers stay genuinely separate → real depth
- Curved, complex decision boundaries → model real-world patterns
- Gradients stay full strength → deep networks train successfully

---

## Key Formulas Reference

```
Neuron output:        z = Σ(wᵢxᵢ) + b
                      h = activation(z)

Weight update:        w = w − η · ∂Loss/∂w

Sigmoid:              σ(z) = 1 / (1 + e⁻ᶻ)        → range (0, 1)
Sigmoid derivative:   σ'(z) = σ(z) · (1 − σ(z))    → max 0.25

ReLU:                 max(0, z)                      → range [0, ∞)
ReLU derivative:      1 for z>0, 0 for z<0

Binary Cross-Entropy: Loss = −[y·log(ŷ) + (1−y)·log(1−ŷ)]

Chain rule:           ∂Loss/∂w₁ = ∂Loss/∂ŷ × ∂ŷ/∂h₁ × ∂h₁/∂w₁
```

---

*Chapter 9 · Topics 9.1–9.3 · June 2026*
