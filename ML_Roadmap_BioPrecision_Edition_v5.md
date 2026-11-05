# Arwindh's ML Roadmap — BioPrecision-AI Edition (v5, Foundations Rebuild)

> **Why this file exists:** You built BioPrecision-AI — a full clinical AI system — entirely with AI assistance. That's not a problem; it's a huge advantage. But right now the project is a "black box" to you. This roadmap restructures your journey so that by the time you finish it, you will *understand every single line of code and every design decision* made in that project. Every chapter has a **"This is in BioPrecision-AI as..."** connection so you always know why you're learning something.

> **What changed in v5 (November 2026):**
> 1. **A senior's advice reset the priority:** be good at **PyTorch** and **very strong in ML basics**. Recommended books: Géron *Hands-On ML with Scikit-Learn and PyTorch* (the spine), Deisenroth et al. *Mathematics for Machine Learning* (MML, paired chapter by chapter), Raschka *Build a Large Language Model (From Scratch)*, Huyen *AI Engineering*.
> 2. **New Stage 1 — Foundations Rebuild (Modules M0–M7 + Capstone + Defense)** is inserted **after Ch 12.5 and before Ch 13**. It is a deliberate *second pass* over Ch 1–10: same topics, but every concept is built three ways (by hand → raw PyTorch → PyTorch built-ins) and every choice is defended against its alternatives.
> 3. **Ch 12 is paused at 12.5.** 12.6–12.9 resume after Stage 1 (Neo4j is off the senior's path; the notebooks make it easy to pick back up).
> 4. **New standing rule — "Why this, not that"** (Instruction 11): for every choice point, the learner must know the alternatives and why each was or wasn't chosen.
> 5. **Diagnostic (M0) results are recorded below** so every module knows which gaps to hit hardest.
> 6. Ch 13, 16, 17 are now taught from Géron's PyTorch chapters; Raschka follows Ch 17; Huyen pairs with Ch 18–19.
>
> **What changed in v4 (August 2026):** 11.5 Async & Concurrency added; notebook standard tightened; git commands required in chat.
> **What changed in v3:** sequential chapter numbering 1–20.

> **Current status:**
> - Ch 1–8 (Classical ML): ✅ Complete (first pass)
> - Ch 9 (Neural Networks 9.1–9.8): ✅ Complete (first pass)
> - Ch 10 (Deep Learning Essentials): ✅ Complete (first pass)
> - Ch 11 (Applied Python for AI Systems): ✅ Complete (11.1–11.6)
> - Ch 12 (Knowledge Graphs): ⏸ Paused after 12.5
> - **Stage 1 — Foundations Rebuild: M0 Diagnostic ✅ · M1.1 note written, checkpoint answers pending** ← YOU ARE HERE

> **How to use this file:** Upload this together with `ARWINDH_ML_JOURNEY_v4.md` in a new chat. This file is the topic roadmap; that file is the teaching-style manual. Both together give Claude everything needed to pick up cold.

---

## Instructions for Claude in the New Chat

1. **Ch 1–10 were learned once, conversationally. Stage 1 rebuilds them in depth on purpose.** Don't re-teach from zero — open each sub-topic with a diagnostic question, find what the learner can't yet *explain or build without notes*, and teach to that.
2. **Teach in full detail, Socratic style:** ask a guiding question before explaining each new idea, let the attempt happen, then correct/confirm. One question at a time, never stacked. **But when the learner is genuinely stuck ("don't know", "explain from base"), give a concrete foothold and explain directly — then hand the next step back.** When the learner asks their own question, answer it directly; don't deflect it into another question.
3. **Use real-world analogies, Indian context where natural** (Chennai flat rents, auto-rickshaw meter, hospital OP counter, "Ravi, a loan officer in Chennai"). A forced analogy is worse than none.
4. **Connect every new idea back to earlier chapters** explicitly before introducing it as new.
5. **ALSO connect every new idea to BioPrecision-AI** where a real connection exists. Don't invent one.
6. **Use diagrams for anything structural** (ASCII preferred inline).
7. **Delivery mode (Nov 2026): study notes first.** For each sub-topic, write the full study note up front instead of teaching it live in chat (token cost). The note carries the Socratic part: 🧠 stop-and-think questions with hidden answers, ✍️ checkpoint questions without answers. The learner reads, answers in the file, and brings doubts and checkpoint answers to chat; Claude checks them briefly and directly. No need to ask before writing the note — the note *is* the lesson. Model note: `S1-Foundations-Rebuild/M1.1-linear-regression-and-the-loss.md`.
8. **Notebook standard.** After each sub-topic is solidly understood, build a notebook matching the repo convention (`<folder>/<n.n>-<topic>.md`). It must:
   - be **self-contained** (prerequisite in one line at the top);
   - **record the doubts and corrections** from the conversation, marked ❌ / ✅;
   - **verify every code output by executing it** — never paste an unrun result;
   - include a **"Why this, not that" table** (Instruction 11);
   - include a **"Break it on purpose" section** — a deliberate failure, the learner's prediction, and the actual result;
   - end with a **"never forget" summary**.
9. **Always give the terminal git commands in the chat message.**
10. **Follow the order in this file.** This file owns the ordering. Don't invent sub-topics or announce a "what's next" not listed here — if something necessary is missing, say so and update this file.
11. **"Why this, not that" — the learner's standing requirement.** For every choice point (loss, metric, split criterion, optimizer, scaler, validation scheme, initialiser, normalisation, architecture…), the learner must be able to answer:
    - **What are the options?**
    - **What does each one assume or optimise?**
    - **When does each one win, and when does it fail?**
    - **Why is the default the default?**

    Teach the alternatives *alongside* the chosen option, not as an afterthought. Ask "why not X?" questions in every session. These tables become the question bank for the Stage 1 Defense.
12. **Always demand the "why" half of an answer.** The M0 diagnostic showed the learner reliably gets the *what* and skips the *why / which first*. Don't accept a correct "what" alone — ask for the reason, even if it's a guess.

### Worked example of Instruction 11 — Gini vs entropy vs misclassification error (M4.1)

| Option | What it measures | When it wins | When it fails | Verdict |
|---|---|---|---|---|
| **Misclassification error** | fraction not in the majority class | reporting a finished tree | **useless for growing a tree:** a split can make one child much purer without changing either child's majority class, so the error doesn't move and greedy search sees no gain | not used to choose splits |
| **Gini impurity** | chance two random samples from the node have different classes, `1 − Σpᵢ²` | default; no logarithm → slightly faster | — | sklearn default |
| **Entropy** | information needed to name the class, `−Σpᵢ log pᵢ` | sometimes slightly more balanced trees | marginally slower | usually gives near-identical trees to Gini |

Check it with numbers: parent node has 400 of class A and 400 of class B.
- Split 1 → (300 A, 100 B) and (100 A, 300 B): misclassification 0.25, weighted Gini **0.375**
- Split 2 → (200 A, 400 B) and (200 A, 0 B): misclassification 0.25, weighted Gini **0.333**

Misclassification error calls it a tie. Gini prefers split 2, which created a perfectly pure node. The **why**: Gini and entropy are *curved* (strictly concave) in the class proportions, so they reward any move toward purity. Misclassification error is made of straight-line pieces, so it only notices changes that flip a majority.

---

## Full Sequence

| Order | Ch / Stage | Topic | Status |
|---|---|---|---|
| 1 | 1–8 | Classical ML (Foundations → Projects) | ✅ first pass |
| 2 | 9 | Neural Networks (9.1–9.8) | ✅ first pass |
| 3 | 10 | Deep Learning Essentials | ✅ first pass |
| 4 | 11 | Applied Python for AI Systems | ✅ Complete |
| 5 | 12 | Knowledge Graphs & Neo4j | ⏸ Paused at 12.5 |
| 6 | **Stage 1** | **Foundations Rebuild — M0–M7, Capstone, Defense** | **🔄 IN PROGRESS** |
| 7 | 12 (resume) | 12.6–12.9 | ⬜ |
| 8 | 13 | CNNs (Géron Ch 12) | ⬜ |
| 9 | 14 | Graph Neural Networks ⭐ | ⬜ |
| 10 | 15 | Explainability & Responsible AI | ⬜ |
| 11 | 16 | NLP & Transformers (Géron Ch 14–15) | ⬜ |
| 12 | 17 | Time Series / RNNs / LSTMs (Géron Ch 13) | ⬜ |
| 13 | **Stage 3** | **LLMs from scratch (Raschka, all 7 chapters)** | ⬜ |
| 14 | 18 | API Design & Software Engineering for AI (+ Huyen) | ⬜ |
| 15 | 19 | MLOps & Deployment (+ Huyen) | ⬜ |
| 16 | 20 | Capstone — Rebuild BioPrecision-AI from scratch | ⬜ |

**The shape of it:**

```
Stage 1  Foundations Rebuild  (strong basics + PyTorch)      ~10–12 weeks
   M0 ✅ → M1 → M2 → M3 → M4 → M5 → M6 → M7 → Capstone → Defense
                         │
Stage 2  Deep learning in PyTorch                            
   12.6–12.9 → 13 CNN → 14 GNN → 15 Explain → 16 Transformers → 17 RNN
                                                                  │
Stage 3  LLMs from scratch (Raschka)  ◄───────────────────────────┘
                         │
Stage 4  Engineering: 18 APIs → 19 MLOps (+ Huyen) → 20 Capstone
```

---

## Stage 1 — Foundations Rebuild

**Folder:** `S1-Foundations-Rebuild/` · **Notebook names:** `M1.1-<topic>.md` etc.

### How every sub-topic runs

1. **Predict** — Claude asks one question before explaining anything.
2. **Read** — the Géron section; MML only for the maths Géron skips.
3. **Build by hand** — paper or NumPy.
4. **Rebuild in PyTorch** — raw tensors first, then built-ins. The learner types every line.
5. **Why this, not that** — the alternatives table (Instruction 11).
6. **Break it on purpose** — predict the failure, run it, explain the gap.
7. **Notebook.**
8. **Next session opens with 3 recall questions** on the previous sub-topic.

### M0 — Diagnostic ✅ (results)

| Q | Topic | Result |
|---|---|---|
| 1 | Unscaled features → NaN | Knew "scale features", not why. Derived it: same η for all weights, gradient ∝ feature size, error multiplied by `(1 − η·x²)` each step → explodes when its size > 1. Gap closed. |
| 2 | Scaler fit before split | ✅ fix and name (leakage); skipped *why it matters in production* |
| 3 | Train RMSE ₹1,200 vs test ₹9,500 | ✅ overfitting, L1/L2, more data. ❌ said "memorizes the **test** data" (it's train). Skipped "which fix first". |
| 4 | 98% accuracy, 2% disease | ✅ lazy model gets 98% catching none. ❌ called the costly error a false positive — it's the **false negative** |
| 5 | Missing `zero_grad()` | ❌ didn't know `.backward()` **adds** into `.grad`. Explained from base; then correctly answered gradient-accumulation check (4 backward calls per step) |
| 6 | L2 regularization | ✅ (polished answer — confirm in own words in M2.3: what happens with λ → ∞?) |
| 7 | PCA + scaling | ✅ |
| 8 | Why trees can't be trained by gradient descent | ❌ no idea → explained: a split threshold has zero gradient almost everywhere and a jump at the edge |

**Gaps to hit hardest:** the *why / which first* half of every answer (all modules) · FP vs FN (M3.3) · the PyTorch training loop and `.grad` (M1.6, M6.3) · why some models can't use gradient descent (M4.1) · L2 in own words (M2.3).

### M1 — Linear Regression & Gradient Descent
*Géron Ch 4 (first half) · MML Ch 5, 7, 9*

| # | Topic | Why this, not that |
|---|---|---|
| M1.1 | Linear regression as a model; the loss | MSE vs MAE vs Huber — why MSE is maximum likelihood under Gaussian noise (MML 9) |
| M1.2 | Closed form (normal equation) vs gradient descent | when the O(d³) solve wins; when GD wins |
| M1.3 | Gradient descent by hand in NumPy; the stability rule | why η·x² decides converge / bounce / explode; link to eigenvalues (MML 4) |
| M1.4 | Batch vs stochastic vs mini-batch GD | noise, speed, memory, why mini-batch is the default |
| M1.5 | Feature scaling | StandardScaler vs MinMax vs Robust — which when, and why trees don't care |
| M1.6 | PyTorch: tensors & shapes → manual gradients → autograd → `nn.Linear` + `optim` | anatomy of the training loop; why `.grad` accumulates; `zero_grad` placement |
| M1.7 | Break it on purpose | the Chennai-flats explosion rerun; learning-rate sweep |

### M2 — Generalization: Bias–Variance, Learning Curves, Regularization
*Géron Ch 1 (Testing and Validating), Ch 4 · MML Ch 8*

| # | Topic | Why this, not that |
|---|---|---|
| M2.1 | Polynomial regression & the bias–variance trade-off | simpler model vs more data vs regularization — which first |
| M2.2 | Learning curves | reading them to decide *before* collecting more data |
| M2.3 | Ridge vs Lasso vs ElasticNet vs early stopping | why L1 zeros weights and L2 doesn't (geometry); λ → 0 and λ → ∞ |
| M2.4 | Validation strategy & leakage | holdout vs k-fold vs stratified vs grouped vs time-series split; why `Pipeline` |
| M2.5 | PyTorch: train/val loop, `weight_decay`, early stopping by hand | weight decay in SGD vs Adam vs AdamW |

### M3 — Classification: Logistic & Softmax Regression, Metrics
*Géron Ch 3, Ch 4 (end)*

| # | Topic | Why this, not that |
|---|---|---|
| M3.1 | Logistic regression | sigmoid + cross-entropy vs MSE — why MSE gives weak gradients and a non-convex loss here |
| M3.2 | Softmax regression | multinomial vs one-vs-rest vs one-vs-one |
| M3.3 | Metrics from the cost of each mistake | FP vs FN; accuracy vs precision vs recall vs F1 vs ROC-AUC vs PR-AUC — which when |
| M3.4 | Class imbalance | class weights vs resampling vs threshold moving |
| M3.5 | PyTorch: logits, `BCEWithLogitsLoss`, `CrossEntropyLoss`, `DataLoader` | why losses take logits, not probabilities (numerical stability) |

### M4 — Trees & Ensembles
*Géron Ch 5–6* · BioPrecision-AI: the Random Forest baseline in `ml/train_baseline.py`

| # | Topic | Why this, not that |
|---|---|---|
| M4.1 | CART | Gini vs entropy vs misclassification (worked example above); greedy search vs gradient descent |
| M4.2 | Regularizing trees; high variance; axis sensitivity | `max_depth` vs `min_samples_leaf` vs pruning |
| M4.3 | Bagging, pasting, Random Forest, out-of-bag | why random feature subsets |
| M4.4 | Boosting & stacking | AdaBoost vs gradient boosting vs XGBoost / LightGBM / HistGradientBoosting |
| M4.5 | Trees vs linear vs neural nets on tabular data | **when NOT to use PyTorch** — no PyTorch in this module, on purpose |

### M5 — Dimensionality Reduction & Unsupervised Learning
*Géron Ch 7–8 · MML Ch 4, 6, 10, 11*

| # | Topic | Why this, not that |
|---|---|---|
| M5.1 | PCA from eigenvectors / SVD | why SVD over the covariance eigendecomposition; scaling; choosing k |
| M5.2 | Other reducers | PCA vs random projection vs LLE vs t-SNE / UMAP |
| M5.3 | Clustering | k-means vs DBSCAN vs hierarchical vs GMM — the assumption each makes, and how each fails |
| M5.4 | Gaussian mixtures & probability | soft vs hard assignment; EM intuition (MML 6, 11) |
| M5.5 | PyTorch: `torch.linalg.svd` PCA and vectorized k-means | loops vs broadcasting — why vectorize |

### M6 — Neural Networks in PyTorch
*Géron Ch 9–10* · BioPrecision-AI: Adam in the GNN extension

| # | Topic | Why this, not that |
|---|---|---|
| M6.1 | MLPs: sklearn → PyTorch | when sklearn's `MLPClassifier` is enough, when it isn't |
| M6.2 | `nn.Module`, custom and non-sequential models | `nn.Sequential` vs subclassing |
| M6.3 | The full training loop | `train()` / `eval()`, `no_grad`, device (`mps`), overfit-one-batch sanity check |
| M6.4 | Activations, losses, optimizers | ReLU vs sigmoid / tanh / GELU; SGD vs momentum vs Adam vs AdamW |
| M6.5 | Hyperparameter search | grid vs random vs Optuna |
| M6.6 | Saving, loading, `torch.compile` | `state_dict` vs pickling the whole model |

### M7 — Training Deep Networks
*Géron Ch 11* · BioPrecision-AI: `dropout=0.2`, `patience=10`

| # | Topic | Why this, not that |
|---|---|---|
| M7.1 | Vanishing / exploding gradients & initialization | Xavier vs He — matched to the activation |
| M7.2 | Normalization & clipping | BatchNorm vs LayerNorm; clip by norm vs by value |
| M7.3 | Reusing pretrained layers | freeze vs fine-tune |
| M7.4 | Faster optimizers & learning-rate schedules | step vs exponential vs cosine vs one-cycle |
| M7.5 | Regularization for deep nets | dropout vs weight decay vs early stopping vs augmentation |
| M7.6 | Debugging playbook | NaN loss, flat loss, bouncing loss, val loss diverging — cause → check → fix |

### Stage 1 Capstone & Defense

1. **Capstone:** a dataset the learner has never touched. One weekend, **no AI, official docs only**: sklearn baseline → PyTorch model → honest evaluation → error analysis.
2. **Defense:** Claude plays the senior and asks ~20 "why" questions about the capstone's choices, drawn from the Why-this-not-that tables. Answered without notes.

**Stage 1 is done when:** every module has its notebooks, Géron's end-of-chapter exercises for Ch 1–11 are passed, and the capstone + defense are passed.

---

## The BioPrecision-AI Technology Map

| Concept Used in BioPrecision-AI | Where in the Project | Learn It In |
|---|---|---|
| **Random Forest Regressor** | `ml/train_baseline.py` | Ch1-8 ✅ · deepened in **M4** |
| **Scikit-learn pipelines** | `ml/train_baseline.py` | Ch1-8 ✅ · **M2.4** |
| **Data normalisation / z-score** | Patient/drug feature scaling | Ch2 ✅ · **M1.5** |
| **Train/Val/Test splitting** | ML baseline evaluation | Ch3 ✅ · **M2.4** |
| **Activation functions (ReLU)** | Between GNN layers | Ch9.3 ✅ · **M6.4** |
| **Loss functions (MSE)** | `train_baseline.py` loss | Ch9.4 ✅ · **M1.1** |
| **Backpropagation** | How the ML baseline learns | Ch9.5 ✅ · **M1.6** |
| **Optimisers (Adam)** | GNN extension uses Adam | Ch9.6 ✅ · **M6.4** |
| **Dropout** | `dropout=0.2` in GNN layers | Ch 10.1 ✅ · **M7.5** |
| **Early stopping** | `patience=10` in GNN training | Ch 10.1 ✅ · **M2.5, M7.5** |
| **Batch Normalisation** | Used in deep GNN pipelines | Ch 10.2 ✅ · **M7.2** |
| **Embeddings** | Every Drug/Gene/Disease node could have an embedding | Ch 10.3 ✅ |
| **PyTorch tensors/autograd** | GNN library built on PyTorch | Ch 10.4 ✅ · **M1.6, M6** |
| **Pydantic data models** | `backend/app/models/patient.py` | Ch 11.1 ✅ |
| **Rule-based / Symbolic AI** | `decision_engine.py` — DDI, Contraindication rules | Ch 11.2 ✅ |
| **Symbolic AI vs Statistical AI** | Why BioPrecision-AI chose rules over ML | Ch 11.3 ✅ |
| **Structured JSON contracts** | The alert payload sent to the React frontend | Ch 11.4 ✅ |
| **Async / concurrency** | `ddi_rule`'s Neo4j round trip per drug pair | Ch 11.5 ✅ |
| **Graph as a data structure** | Neo4j — Drug/Gene/Disease/Variant network | Ch 12.1 ✅ |
| **Knowledge Graph** | The Neo4j database is a Knowledge Graph | Ch 12.1 ✅ |
| **Neo4j & Cypher queries** | `graph_service.py` — every database query | Ch 12.3/4 ✅ |
| **Graph traversal** | How DDI detection works in Neo4j | Ch 12.5 ✅ |
| **Graph ingestion (ETL)** | `ingest_ddi.py` loads CSV into Neo4j | **Ch 12.7** |
| **Node embeddings** | One embedding per Drug, Gene, Disease | **Ch 14.2** |
| **Message passing** | What every GNN layer computes | **Ch 14.3** |
| **Heterogeneous graphs** | 4 node types, 10 relationship types | **Ch 14.5** |
| **Graph attention** | The "HAN" in HANConv | **Ch 14.6** |
| **Explainability design** | 7-field alert format | **Ch 15.6** |
| **REST API + FastAPI** | `backend/app/main.py` | **Ch 18.1/2** |
| **Pytest + mocking** | `scripts/qa/` — 10 tests, 10/10 pass | **Ch 18.4/5** |

---

## ✅ Chapter 10 — Deep Learning Essentials (COMPLETE, first pass)

| # | Topic | Notebook | BioPrecision-AI connection |
|---|---|---|---|
| 10.1 | **Regularisation** — Dropout, L2/weight decay, Early Stopping | `10.1-regularisation.md` | `dropout=0.2` and `patience=10` in the GNN extension |
| 10.2 | Batch Normalisation | `10.2-batch-norm.md` | Used in deep GNN pipelines |
| 10.3 | Embeddings — a learned dense vector representing something | `10.3-embeddings.md` | Every Drug, Gene, Disease node could have an embedding |
| 10.4 | PyTorch fundamentals — tensors, autograd, `nn.Module` | `10.4-pytorch-basics.md` | PyTorch Geometric is built on PyTorch |
| 10.5 | Mini-project — rebuild the loan-default model in PyTorch | `10.5-pytorch-project.md` | Transition practice before GNNs |

---

## ✅ Chapter 11 — Applied Python for AI Systems (COMPLETE)

| # | Topic | BioPrecision-AI as... |
|---|---|---|
| 11.1 | **Pydantic data models** — strict data typing and validation | `backend/app/models/patient.py` — the Patient, Medication, GenomicProfile models |
| 11.2 | **Rule-based AI / Expert Systems** — deterministic if-then reasoning | `decision_engine.py` — DDI rule, Contraindication rule, Polypharmacy rule |
| 11.3 | **Symbolic AI vs Statistical AI** — why and when to use each | BioPrecision-AI chose rules (not ML) for clinical safety — here is why |
| 11.4 | **Structured JSON as a contract** — designing reliable data formats | The alert payload: `{drug, risk_type, severity, evidence, source_database, explanation}` |
| 11.5 | **Async & concurrency** — the event loop, yield points, `gather`, and the blocking-call trap | `ddi_rule` makes one Neo4j round trip per drug pair; the early-`return` version reports only the first interaction found |
| 11.6 | **Mini-project** — write a rule engine that validates a dataset and fires alerts | Rebuild `decision_engine.py` logic from scratch for a new problem |

---

## ⏸ Chapter 12 — Knowledge Graphs & Graph Databases (paused after 12.5)

*This must come before Ch14: you cannot learn a network that reasons over neighbours before you can construct and traverse a graph yourself. 12.6–12.9 resume right after Stage 1.*

| # | Topic | BioPrecision-AI as... | Status |
|---|---|---|---|
| 12.1 | What is a Knowledge Graph? | BioPrecision-AI's entire Neo4j database IS a knowledge graph | ✅ |
| 12.2 | Graph schema design — node types and relationship types | `graph/schema/graph_schema.md` — TARGETS, LOCATED_IN, INTERACTS_WITH | ✅ |
| 12.3 | Neo4j fundamentals — creating nodes, relationships | The `constraints.cypher` file and node creation scripts | ✅ |
| 12.4 | Cypher query language — MATCH, WHERE, RETURN, MERGE | `graph_service.py` — every query that drives the clinical engine | ✅ |
| 12.5 | Graph traversal — multi-hop queries | How contraindications are found in the graph | ✅ |
| 12.6 | Provenance in graphs — storing where data came from | `source_dataset`, `evidence`, `confidence` on every edge | ⬜ |
| 12.7 | Idempotent graph ingestion — MERGE vs CREATE | `ingest_ddi.py` — safe to run twice without duplication | ⬜ |
| 12.8 | Graph integrity — orphan nodes, constraints | The QA test `test_neo4j_orphan_nodes` | ⬜ |
| 12.9 | Mini-project — design a small knowledge graph from scratch | Rebuilding a mini BioPrecision-AI graph from first principles | ⬜ |

---

## Chapter 13 — Convolutional Neural Networks (CNNs)

*Géron Ch 12, in PyTorch. Kept immediately before Ch14 so the convolution → message-passing bridge stays intact.*

| # | Topic | Why this, not that |
|---|---|---|
| 13.1 | The convolution operation — local patterns, parameter sharing | convolution vs a dense layer on raw pixels |
| 13.2 | Pooling, feature maps | max vs average pooling vs strided convolution |
| 13.3 | Classic architectures (LeNet → ResNet) | why skip connections |
| 13.4 | Project — small image classifier; transfer learning with TorchVision | train from scratch vs fine-tune a pretrained model |

---

## Chapter 14 — Graph Neural Networks ⭐

*Not covered by any of the four books — use the PyTorch Geometric docs. Built around the GNN research project and BioPrecision-AI's knowledge graph.*

| # | Topic | BioPrecision-AI as... |
|---|---|---|
| 14.1 | Why graphs? — nodes, edges, adjacency | The Drug/Gene/Disease/Variant network with 178k edges |
| 14.2 | Node embeddings — learning a vector per node | One embedding per Drug, Gene, Disease |
| 14.3 | Message passing — convolution generalized to graphs | What a GNN layer does on the BioPrecision-AI graph |
| 14.4 | Graph Convolutional Networks (GCN) | The "plain" GNN before attention |
| 14.5 | Heterogeneous graphs — multiple node types, multiple edge types | BioPrecision-AI: Drug, Gene, Disease, Variant + 10 relationship types |
| 14.6 | Graph attention (light intro) | How HANConv weights Gene neighbors vs Disease neighbors |
| 14.7 | HANConv specifically — semantic-level attention across meta-paths | The layer from your GNN research |
| 14.8 | Link prediction — predicting a relationship between two nodes | Predicting if a Drug INTERACTS_WITH another Drug |
| 14.9 | Project — rebuild your drug-ranking pipeline with full understanding | The GNN research, properly understood |

*Why this, not that, for the whole chapter: GCN vs GraphSAGE vs GAT vs HAN — what each assumes about neighbours.*

---

## Chapter 15 — Explainability & Responsible AI

| # | Topic | BioPrecision-AI as... |
|---|---|---|
| 15.1 | Why explainability matters — especially in healthcare | Why BioPrecision-AI uses rules, not a neural network, for clinical decisions |
| 15.2 | Symbolic AI explainability — rule tracing | The exact alert trace: `decision_engine.py` → `graph_service.py` → Neo4j → JSON |
| 15.3 | SHAP for tree models (review + deepen) | Applying SHAP to explain the Random Forest baseline — SHAP vs permutation importance vs impurity importance |
| 15.4 | Attention visualisation | Which nodes the model "looked at" |
| 15.5 | GNNExplainer | "Why did the model flag Phentermine as high risk?" |
| 15.6 | Alert design | The 7-field alert format: `{patient, drug, risk_type, severity, evidence, source_database, explanation}` |

---

## Chapter 16 — NLP & Transformers

*Géron Ch 14–15, in PyTorch.*

| # | Topic |
|---|---|
| 16.1 | Word embeddings |
| 16.2 | RNNs (brief preview) |
| 16.3 | Attention mechanism — self-attention, multi-head attention |
| 16.4 | The Transformer architecture — encoder-only vs decoder-only vs encoder-decoder |
| 16.5 | **Revisit HANConv** — re-derive what `heads=4` semantic attention was doing |
| 16.6 | Project — small text classifier |

---

## Chapter 17 — Time Series / RNNs / LSTMs

*Géron Ch 13, in PyTorch.*

| # | Topic |
|---|---|
| 17.1 | RNN architecture |
| 17.2 | Vanishing gradient in RNNs |
| 17.3 | LSTM vs GRU vs 1-D CNN for sequences |
| 17.4 | Project — forecasting |

---

## Stage 3 — LLMs from Scratch (Raschka)

*The deepest PyTorch practice in the plan: build a GPT end to end. Follow the book's chapter order.*

| # | Topic |
|---|---|
| L1 | Working with text data — tokenization, BPE, data loaders |
| L2 | Coding attention mechanisms |
| L3 | Implementing a GPT model |
| L4 | Pretraining on unlabeled data |
| L5 | Fine-tuning for classification |
| L6 | Fine-tuning to follow instructions (+ LoRA appendix) |

---

## Chapter 18 — API Design & Software Engineering for AI

*Read Huyen's* AI Engineering *alongside Ch 18–19, about one chapter a week.*

| # | Topic | BioPrecision-AI as... |
|---|---|---|
| 18.1 | What is a REST API? — GET, POST, HTTP status codes, JSON | `GET /api/patients`, `POST /api/analyze/{id}` in `main.py` |
| 18.2 | FastAPI fundamentals — routes, request/response models | `backend/app/main.py` — every endpoint |
| 18.3 | API response contracts — why the JSON format matters | The alert format the React frontend depends on |
| 18.4 | Testing APIs with Pytest | `scripts/qa/test_03_api_integration.py` |
| 18.5 | Mocking dependencies in tests | `test_02_clinical_engine.py` mock engine pattern |
| 18.6 | Environment variables and secrets | `NEO4J_PASSWORD` in `graph_service.py` |
| 18.7 | CORS — why browsers block cross-origin requests | `allow_origins=["*"]` in `main.py` |
| 18.8 | Mini-project — build a small FastAPI clinical alert service | A working API mirroring BioPrecision-AI backend |

---

## Chapter 19 — MLOps & Deployment

| # | Topic | BioPrecision-AI connection |
|---|---|---|
| 19.1 | Model serving basics | How the Random Forest is loaded and served |
| 19.2 | Experiment tracking | Tracking accuracy of the ML baseline |
| 19.3 | Docker / deployment fundamentals | Running BioPrecision-AI in a container |
| 19.4 | Project | — |

---

## Chapter 20 — Capstone

> **Rebuild BioPrecision-AI from scratch, by yourself, with full understanding.**

- Add at least 10 new drugs
- Write your own Cypher DDI queries without help
- Add a `/health` endpoint
- Write 15 Pytest cases
- Add proper `.env.example`

---

## How to Start Your Next Session

Upload this file + `ARWINDH_ML_JOURNEY_v4.md`, then paste:

> "I've completed Chapters 1–11 of my ML journey and Chapter 12 up to 12.5 — all notebooks pushed to github.com/ArwindhCN/my-ml-journey. Following my senior's advice (strong PyTorch + very strong ML basics), I'm now in **Stage 1 — Foundations Rebuild**. The M0 diagnostic is done (results are in the roadmap). I'm starting **M1.1**. This roadmap owns the ordering — follow it exactly and don't invent sub-topics. Teach me Socratically: one question at a time, but explain directly when I'm genuinely stuck, and answer my own questions directly. For every choice, teach me the alternatives and why not them (Instruction 11), and don't let me skip the 'why' half of an answer. Ask before building a notebook; when you do, include my doubts and corrections, a Why-this-not-that table, a Break-it-on-purpose section, and a 'never forget' summary. Give me the git commands in the chat."

---

*Updated November 2026 (v5). Supersedes v4, which had no Foundations Rebuild stage and listed Ch 11 as in progress.*
