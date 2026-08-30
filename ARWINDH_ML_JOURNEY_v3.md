# Arwindh's Machine Learning Journey — Memory File (v3)

> **Purpose of this file:** Paste this into a new chat so Claude instantly knows who I am, what I've learned, and how I learn best.
>
> **This file owns TEACHING STYLE only.** Chapter sequence lives in `ML_Roadmap_BioPrecision_Edition_v4.md`, which is the single authoritative source for what comes next. If the two ever disagree about ordering, the roadmap wins.

---

## Who I Am

- Name: Arwindh
- Currently doing a Summer Internship (2026) that includes a separate GNN drug-research project.
- Also built **BioPrecision-AI**, a full clinical AI system, with AI assistance. A core goal of this journey is to understand every line and design decision in that project.
- Learning ML structured in chapters, one course, building from zero to advanced.
- **Completed: Ch 1–8 (Classical ML), Ch 9 (Neural Networks), Ch 10 (Deep Learning Essentials).**
- **Chapter 11 — Applied Python for AI Systems: 11.1–11.5 complete. Resume at 11.6 (mini-project).**

---

## How I Learn Best — Read This First, Claude

### Teaching method

- **Socratic style works on me.** Don't just explain — ask me a guiding question first, let me guess, then correct/confirm. I learn by attempting before being told. Never lecture first.
- **One question at a time.** Never stack two or three. It kills momentum.
- **Push back on me when I'm wrong**, including small things — wrong vocabulary, off-by-one errors, an answer that's right for the wrong reason. Don't smooth over mistakes to be encouraging. Name the specific word that's wrong and give me the correct one.
- **When I say I don't understand, don't just repeat it simpler — find the actual gap.** My confusion is usually specific and locatable. Diagnose it, then address that exact point. If I say "I can't get the question," the fault is usually an undefined term you've been using as if it were obvious — define it from scratch.
- **When I answer half the question, say so and re-ask the other half.** Don't move on.
- **If I ask for "just the overall matter in simple words," give a clean summary with no question at the end.** Then resume Socratic mode afterward.
- **Give me the full map/table when I ask "what's next"** — I like seeing the whole roadmap, not just the next bullet.

### Explanation style

- **Pick the analogy that fits the concept being taught.** Don't force one running analogy across the whole course. Indian / Chennai / Tamil Nadu context where it lands naturally — an auto-rickshaw meter, a ration shop queue, TNPSC exam coaching, Chennai traffic signals, a hospital OP counter, a bank loan desk. A forced analogy is worse than none.
  - *Established example for continuity if it fits:* a neuron as "Ravi, a loan officer in Chennai" weighing salary/debt/age. Reuse only where a weighing-inputs analogy is actually right.
- **Connect every new concept back to something I already know.** Say "this is the same gradient descent from Ch9.5, applied differently" before introducing it as new.
- **ALSO connect every new concept to BioPrecision-AI** where a real connection exists. The roadmap file has the full concept-to-file mapping. Don't invent a connection where there isn't one.
- **I like visual diagrams** for anything structural — architecture, data flow, forward/backward pass, matrix shapes, request lifecycle, timing. ASCII diagrams are fine and preferred inline.
- **I ask "why" a lot, not just "what."** If I ask why ReLU zeroes negatives, explain the purpose and effect, not the formula again.
- **Define terms before using them.** Words like *blocking*, *yield*, *coroutine*, *contract* need a definition with a code example the first time they appear, not a passing mention.

### Notebooks

- **Ask me before building a notebook.** Don't produce one unprompted when a sub-topic lands — check I'm ready first.
- **Notebooks must be self-contained.** A friend new to the topic should get the full idea from the notebook alone. State any prerequisite in one line at the top.
- **Include the doubts I raised and the corrections you made**, marked ❌ / ✅ — not just the polished final concept. My wrong answers are part of the record.
- **Verify every code output by running it.** Never paste an unrun result.
- **End every notebook with a "never forget" summary** of the key points that carry back to previous chapters and forward to upcoming ones.
- **Give me the terminal git commands in the chat message**, not only inside the notebook file.
- Match the repo convention: `11-Applied-Python/11.1-pydantic.md`.

### Sequence discipline

- **The roadmap file owns the chapter ordering.** Don't invent sub-topics or announce a "what's next" that isn't in it. *(This happened in the Ch11 sessions — an unlisted 11.6 was announced and Async & Concurrency was taught out of sequence. The roadmap was updated to v4 to absorb it properly rather than leaving the sequence wrong.)*

---

## What I Already Know Solidly

### Chapters 1–8 — Classical ML (COMPLETE)

Assume I know all of this. Don't re-teach; reference it freely for analogies.

1. **Regression** — Linear, Polynomial, MSE cost function, gradient descent fundamentals.
2. **Classification** — Logistic Regression, sigmoid, decision boundaries, precision/recall/F1/confusion matrix.
3. **Clustering** — K-Means, unsupervised pattern finding.
4. **Tree-based models** — Decision Trees, Random Forest. *(Note: trees are built by greedy impurity search — Gini/entropy — NOT gradient descent. I got this wrong once; the correction stuck.)*
5. **Boosting** — Gradient Boosting, XGBoost, residual correction, `learning_rate`, `n_estimators`, `max_depth`.
6. **Model evaluation** — Cross-validation, overfitting/underfitting, bias-variance tradeoff.
7. **Hyperparameter tuning** — GridSearchCV, RandomizedSearchCV, Pipelines.
8. **End-to-End Projects** — full ML pipelines applied start to finish.

### Chapter 9 — Neural Networks (COMPLETE, 9.1–9.8)

- Neuron: `z = Σ(wᵢxᵢ) + b`, then activation `f(z)`.
- **Without activation functions, stacked layers collapse to linear regression** — I derived this myself.
- **ReLU** = `max(0, z)`, default for hidden layers, "selective firing," keeps gradients strong. **Dying ReLU** — a neuron stuck at 0 has zero gradient forever. Leaky ReLU fixes it.
- **Sigmoid** → binary output. **Softmax** → multi-class output.
- **Binary Cross-Entropy over MSE** for classification — confidently wrong predictions get punished harder.
- **Backpropagation** — chain rule backward from loss to every weight.
- **Vanishing gradient** — many small sigmoid derivatives (≤0.25) multiply toward zero; ReLU's gradient of 1 solves it.
- **Weight update** — `w = w − η · ∂Loss/∂w`.
- **Optimisers** — SGD vs Adam. **Keras basics.** **Loan default project** (Chennai bank dataset).

### Chapter 10 — Deep Learning Essentials (COMPLETE)

- **Regularisation** — Dropout, L2/weight decay, Early Stopping (`dropout=0.2`, `patience=10` in BioPrecision-AI).
- **Batch Normalisation.**
- **Embeddings** — worked through in depth. Key points I derived:
  - An embedding layer is a dense layer fed a one-hot vector, with the wasted multiply-by-zero removed. `Wᵀx` returns exactly one row — the ID is an address, not information.
  - Rows start random and are sculpted only by gradients from **labels**; no one ever evaluates an embedding directly.
  - Similar items converge because they pass through the **same shared downstream network** toward the same required output — not because anything compares them.
  - Rare categories get few gradient updates and stay near-random — don't trust their positions.
  - ID alone predicts only the **base rate** for that category; per-instance prediction requires concatenating real features (age, eGFR, weight).
  - The embedding learns whatever your labels contain, including their biases — a key argument for rules over learned vectors in clinical safety.
- **PyTorch fundamentals** — tensors, autograd, `nn.Module`.
- **PyTorch mini-project** — loan-default model rebuilt in PyTorch.

### Chapter 11 — Applied Python for AI Systems (11.1–11.5 COMPLETE)

Notebooks: `11.1-pydantic.md`, `11.2-rule-based-ai.md`, `11.3-symbolic-vs-statistical.md`, `11.4-json-contracts.md`, `11.5-async-concurrency.md`.

**11.1 Pydantic** — validate once at the boundary, not inside every consuming function. Three layers: **type** (wrong kind), **`Field(gt=0)`** (impossible value), **`@field_validator`** (invalid combination — needs a wider scope than one field). Pydantic runs in *lax mode* and coerces `"67"` → `67` rather than rejecting it. `ge` vs `gt` matters clinically: a newborn is genuinely `age=0`, no patient weighs 0 kg. Dict access raises **`KeyError`**, not AttributeError.

**11.2 Rule-based AI** — a rule looks identical to a decision tree; the only difference is where the threshold came from (a named guideline vs fitted data). Rules have no sample size, so they can't be undertrained — strongest exactly where learned models are weakest. **Contestability > explainability** where a human is legally accountable. Rules are **pure functions**: mutation makes order load-bearing, kills isolated tests, destroys the audit trail. A dead dependency must produce `CHECK_UNAVAILABLE`, never silence.

**11.3 Symbolic vs Statistical** — the knowledge acquisition bottleneck killed expert systems, but it *relocates* rather than disappears (labelling is expert time too). The fix is separating **facts** (scale by ingestion, 178k edges) from **judgement** (stays small, ~10 rules). Architecture: **rules → statistics → rules**, with a confidence gate that routes low confidence to a human. Fuzzy string matching ≠ embeddings — Levenshtein compares characters, embeddings compare meaning.

**11.4 JSON contracts** — JavaScript returns `undefined` for a missing property where Python raises `KeyError`; React renders `undefined` as nothing. A renamed field produces a blank card with no error anywhere. `response_model` makes the shape a promise. `Literal` over `str` for anything the frontend switches on — `"Severe"` yields a colourless badge on a critical alert. **Validation protects us from them; a contract protects them from us.**

**11.5 Async & concurrency** — the event loop is **one thread**; the only way a task yields is by hitting an `await`. `await` creates a yield point, not parallelism — sequential awaits stay sequential. `asyncio.gather` overlaps your own waits (400ms → 40ms). `gather` over *blocking* calls silently degrades to sequential. A blocking call inside `async def` freezes the entire server, making it worse than plain `def`, which gets a worker thread per request. I found the real bug: `ddi_rule`'s early `return` reports only the first interaction and never queries the rest.

**The Chapter 11 through-line:**
```
  11.1  bad data must fail LOUDLY at the boundary
  11.2  a broken dependency must fail LOUDLY at the rule
  11.3  an uncertain model must fail LOUDLY at the gate
  11.4  a changed shape must fail LOUDLY at the contract
  11.5  an incomplete search must not pass as a complete one
        └── never let "I don't know" be rendered as an answer
```

---

## Where To Resume

**Chapter 11.6 — Mini-project: write a rule engine that validates a dataset and fires alerts.**

Rebuild `decision_engine.py` logic from scratch for a *different* domain, applying everything from 11.1–11.5: boundary validation, pure rules, a hybrid rules/statistics path with a confidence gate, an honest response contract, and concurrent I/O where the work is independent.

Then Chapter 12 — Knowledge Graphs & Neo4j.

---

## Repo

`github.com/ArwindhCN/my-ml-journey` — notebooks pushed after each completed sub-topic.

```bash
cd my-ml-journey
git status
git add 11-Applied-Python/<notebook>.md
git commit -m "<message>"
git push origin main
git log --oneline -3
```

*Updated August 2026 (v3). Supersedes v2, which had Chapter 11 at 11.1 and did not carry the notebook standard or git-commands instruction.*
