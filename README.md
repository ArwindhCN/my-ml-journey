# Arwindh's ML Journey — BioPrecision-AI Edition

Learning machine learning from the ground up — starting with the maths, then applying algorithms to real-world problems to understand what works best and when.

> **Why this roadmap exists:** You built BioPrecision-AI — a full clinical AI system — entirely with AI assistance. That's not a problem; it's a huge advantage. But right now the project is a "black box" to you. This roadmap restructures your journey so that by the time you finish it, you will *understand every single line of code and every design decision* made in that project. Every chapter has a **"This is in BioPrecision-AI as..."** connection so you always know why you're learning something.

> **Current status (November 2026):**
> - Ch 1–8 (Classical ML): ✅ Complete (first pass)
> - Ch 9 (Neural Networks 9.1–9.8): ✅ Complete (first pass)
> - Ch 10 (Deep Learning Essentials): ✅ Complete (first pass)
> - Ch 11 (Applied Python for AI Systems 11.1–11.6): ✅ Complete
> - Ch 12 (Knowledge Graphs & Neo4j): ⏸ Paused after 12.5 — 12.6–12.9 resume after Stage 1
> - **Stage 1 — Foundations Rebuild: 🔄 In progress** ← you are here

> **What changed (v5):** A senior's advice reset the priority to **PyTorch + very strong ML basics**. So a new **Stage 1 — Foundations Rebuild** sits between Ch 12.5 and Ch 13: a deliberate second pass over Ch 1–10 where every concept is built three ways (by hand → raw PyTorch → PyTorch built-ins) and every choice is defended against its alternatives ("why this, not that").
>
> **Books:** Géron *Hands-On ML with Scikit-Learn and PyTorch* (the spine) · Deisenroth et al. *Mathematics for Machine Learning* (paired) · Raschka *Build a Large Language Model (From Scratch)* · Huyen *AI Engineering*.

---

## 🗺️ Progress & Full Sequence

| Order | Chapter / Stage | Topic | Status |
|---|---|---|---|
| 1 | **01** | Foundations — maths & tools | ✅ Complete |
| 2 | **02** | Data skills — cleaning & feature engineering | ✅ Complete |
| 3 | **03** | Evaluation & metrics | ✅ Complete |
| 4 | **04** | Classification algorithms | ✅ Complete |
| 5 | **05** | Regression algorithms | ✅ Complete |
| 6 | **06** | Unsupervised learning | ✅ Complete |
| 7 | **07** | Model selection & tuning | ✅ Complete |
| 8 | **08** | End-to-end projects | ✅ Complete |
| 9 | **09** | Neural Networks (Keras 9.1–9.8) | ✅ Complete |
| 10 | **10** | Deep Learning Essentials (PyTorch 10.1–10.5) | ✅ Complete |
| 11 | **11** | Applied Python for AI Systems (11.1–11.6) | ✅ Complete |
| 12 | **12** | Knowledge Graphs & Neo4j | ⏸ Paused at 12.5 |
| 13 | **Stage 1** | **Foundations Rebuild — M0–M7, Capstone, Defense** | 🔄 **In progress** |
| 14 | **12 (resume)** | Knowledge Graphs 12.6–12.9 | ⬜ Planned |
| 15 | **13** | Convolutional Neural Networks (Géron Ch 12) | ⬜ Planned |
| 16 | **14** | Graph Neural Networks (PyTorch Geometric) ⭐ | ⬜ Planned |
| 17 | **15** | Explainability & Responsible AI | ⬜ Planned |
| 18 | **16** | NLP & Transformers (Géron Ch 14–15) | ⬜ Planned |
| 19 | **17** | Time Series / RNNs / LSTMs (Géron Ch 13) | ⬜ Planned |
| 20 | **Stage 3** | LLMs from scratch (Raschka) | ⬜ Planned |
| 21 | **18** | API Design & Software Engineering for AI (+ Huyen) | ⬜ Planned |
| 22 | **19** | MLOps & Deployment (+ Huyen) | ⬜ Planned |
| 23 | **20** | Capstone — Rebuild BioPrecision-AI from scratch | ⬜ Planned |

**The shape of it:**

```
Stage 1  Foundations Rebuild  (strong basics + PyTorch)
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

## 🚀 Structure

```
my-ml-journey/
├── 01-Foundations/              # Linear algebra, calculus, stats, NumPy, Pandas
├── 02-Data-Skills/              # Missing values, encoding, scaling, EDA
│   └── titanic-survival.ipynb  # Chapter 2 project — 81% accuracy
├── 03-Evaluation/               # Confusion matrix, precision, recall, F1, ROC-AUC, CV
├── 04-Classification/           # LR, KNN, DT, RF, SVM, NB
├── 05-Regression/               # Linear, Ridge, Lasso, Polynomial, DT, RF, XGBoost
├── 06-Unsupervised-Learning/    # K-Means, Hierarchical, DBSCAN, PCA, t-SNE
├── 07-Model Selection & Tuning/ # GridSearchCV, RandomizedSearchCV, Pipelines
├── 09-Neural-Networks/          # Perceptron, Backpropagation, Optimisers, Keras
├── 10-Deep-Learning-Essentials/ # Regularisation, BatchNorm, Embeddings, PyTorch Basics
├── 11-Applied-Python/           # Pydantic, rule-based AI, JSON contracts, async
│   └── placement_engine.py     # Chapter 11 project — placement eligibility engine
├── 12-Knowledge-Graphs/         # Knowledge graphs, schema design, Neo4j, Cypher, traversal
├── S1-Foundations-Rebuild/      # Stage 1 — second pass over Ch 1–10, built in PyTorch
└── PROBLEMS/                    # Extra practice problems (house rent prediction)
```

---

## 🧬 BioPrecision-AI Technology Map

Every concept used in BioPrecision-AI, and exactly where in this roadmap you learn it:

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
| **Dropout** | `dropout=0.2` in GNN layers | Ch10.1 ✅ · **M7.5** |
| **Early stopping** | `patience=10` in GNN training | Ch10.1 ✅ · **M2.5, M7.5** |
| **Batch Normalisation** | Used in deep GNN pipelines | Ch10.2 ✅ · **M7.2** |
| **Embeddings** | Every Drug/Gene/Disease node could have an embedding | Ch10.3 ✅ |
| **PyTorch tensors/autograd** | GNN library built on PyTorch | Ch10.4 ✅ · **M1.6, M6** |
| **Pydantic data models** | `backend/app/models/patient.py` | Ch11.1 ✅ |
| **Rule-based / Symbolic AI** | `decision_engine.py` — DDI, Contraindication rules | Ch11.2 ✅ |
| **Symbolic AI vs Statistical AI** | Why BioPrecision-AI chose rules over ML | Ch11.3 ✅ |
| **Structured JSON contracts** | The alert payload sent to the React frontend | Ch11.4 ✅ |
| **Async / concurrency** | `ddi_rule`'s Neo4j round trip per drug pair | Ch11.5 ✅ |
| **Graph as a data structure** | Neo4j — Drug/Gene/Disease/Variant network | Ch12.1 ✅ |
| **Knowledge Graph** | The Neo4j database is a Knowledge Graph | Ch12.1 ✅ |
| **Neo4j & Cypher queries** | `graph_service.py` — every database query | Ch12.3/4 ✅ |
| **Graph traversal** | How DDI detection works in Neo4j | Ch12.5 ✅ |
| **Graph ingestion (ETL)** | `ingest_ddi.py` loads CSV into Neo4j | **Ch 12.7** |
| **Node embeddings** | One embedding per Drug, Gene, Disease | **Ch 14.2** |
| **Message passing** | What every GNN layer computes | **Ch 14.3** |
| **Heterogeneous graphs** | 4 node types, 10 relationship types | **Ch 14.5** |
| **Graph attention** | The "HAN" in HANConv | **Ch 14.6** |
| **Explainability design** | 7-field alert format | **Ch 15.6** |
| **REST API + FastAPI** | `backend/app/main.py` | **Ch 18.1/2** |
| **Pytest + mocking** | `scripts/qa/` — 10 tests, 10/10 pass | **Ch 18.4/5** |

---

## 📖 Chapter Breakdowns

### Chapter 01–08: Classical Machine Learning ✅
Foundations, data cleaning, evaluation metrics, classification, regression, clustering, PCA, hyperparameter tuning, and end-to-end projects. All complete.

### Chapter 09: Neural Networks ✅
Where classical ML ends and deep learning begins. Stacking neurons in layers to learn complex patterns.
- Perceptron, activations, BCE/MSE loss, Backpropagation, SGD/Adam, and Keras implementation.

### Chapter 10: Deep Learning Essentials ✅
- **Regularisation (10.1)**: Fight overfitting via dropout, weight decay, and early stopping.
- **Batch Normalisation (10.2)**: Stabilise activations inside deep networks.
- **Embeddings (10.3)**: Learned dense representations for categorical entities.
- **PyTorch Fundamentals (10.4)**: Tensors, autograd, and `nn.Module` abstraction.
- **Mini-Project (10.5)**: Rebuilt the loan default model from scratch in PyTorch.

### Chapter 11: Applied Python for AI Systems ✅

*Covers the Python patterns used in BioPrecision-AI's backend.*

| # | Topic | BioPrecision-AI as... |
|---|---|---|
| 11.1 | **Pydantic data models** — strict data typing and validation | `backend/app/models/patient.py` — Patient, Medication, GenomicProfile |
| 11.2 | **Rule-based AI / Expert Systems** — deterministic reasoning | `decision_engine.py` — DDI, Contraindication, Polypharmacy rules |
| 11.3 | **Symbolic AI vs Statistical AI** — why and when to use each | Rationale behind rules vs ML for safety |
| 11.4 | **Structured JSON as a contract** — designing reliable formats | Alert payload format |
| 11.5 | **Async & concurrency** — the event loop, `gather`, the blocking-call trap | `ddi_rule`'s Neo4j round trip per drug pair |
| 11.6 | **Mini-project** — campus placement eligibility engine | Rebuilding `decision_engine.py` logic for a new problem |

---

## ⏸ Chapter 12 — Knowledge Graphs & Graph Databases (paused after 12.5)

| # | Topic | Status |
|---|---|---|
| 12.1 | What is a Knowledge Graph? | ✅ |
| 12.2 | Graph schema design — node and relationship types | ✅ |
| 12.3 | Neo4j fundamentals — CREATE vs MERGE, identity, constraints | ✅ |
| 12.4 | Cypher — MATCH, WHERE, RETURN | ✅ |
| 12.5 | Graph traversal — multi-hop queries | ✅ |
| 12.6 | Provenance in graphs — storing where data came from | ⬜ after Stage 1 |
| 12.7 | Idempotent graph ingestion — MERGE vs CREATE | ⬜ after Stage 1 |
| 12.8 | Graph integrity — orphan nodes, constraints | ⬜ after Stage 1 |
| 12.9 | Mini-project — design a small knowledge graph from scratch | ⬜ after Stage 1 |

---

## 🔄 Stage 1 — Foundations Rebuild (in progress)

A second, deeper pass over Ch 1–10. Every sub-topic: **predict → read (Géron + MML) → build by hand → rebuild in PyTorch → "why this, not that" → break it on purpose → notebook.**

| Module | Topic | Status |
|---|---|---|
| **M0** | Diagnostic | ✅ |
| **M1** | Linear Regression & Gradient Descent | 🔄 In progress |
| **M2** | Generalization — Bias–Variance, Learning Curves, Regularization | ⬜ |
| **M3** | Classification — Logistic & Softmax Regression, Metrics | ⬜ |
| **M4** | Trees & Ensembles | ⬜ |
| **M5** | Dimensionality Reduction & Unsupervised Learning | ⬜ |
| **M6** | Neural Networks in PyTorch | ⬜ |
| **M7** | Training Deep Networks | ⬜ |
| — | Stage 1 Capstone & Defense | ⬜ |

---

## ⬜ Chapter 13 — Convolutional Neural Networks (CNNs)
*Géron Ch 12, in PyTorch.*
- **13.1–13.3**: Convolution, pooling, feature maps, classic architectures (LeNet → ResNet).
- **13.4**: Project: small image classifier; transfer learning with TorchVision.

---

## ⬜ Chapter 14 — Graph Neural Networks (GNNs) ⭐
- **14.1–14.4**: Graphs, node embeddings, message passing, and GCNs.
- **14.5–14.7**: Heterogeneous graphs, GAT, and HANConv (from GNN research).
- **14.8–14.9**: Link prediction and drug-ranking capstone.

---

## ⬜ Chapter 15 — Explainability & Responsible AI
- **15.1–15.3**: Why explainability matters, rule tracing, and SHAP.
- **15.4–15.6**: Attention maps, GNNExplainer, and clinical alert design.

---

## ⬜ Chapter 16 — NLP & Transformers
*Géron Ch 14–15, in PyTorch.*
- **16.1–16.4**: Word embeddings, attention mechanism, Transformer architecture.
- **16.5–16.6**: Revisit HANConv's semantic attention; project: text classifier.

---

## ⬜ Chapter 17 — Sequence Models (RNNs / LSTMs)
*Géron Ch 13, in PyTorch.*
- **17.1–17.3**: Recurrent cells, vanishing gradient, LSTM/GRU.
- **17.4**: Project: sequential patient data prediction.

---

## ⬜ Stage 3 — LLMs from Scratch (Raschka)
- **L1–L3**: Text data and tokenization (BPE), coding attention, implementing a GPT model.
- **L4–L6**: Pretraining, fine-tuning for classification, instruction fine-tuning (+ LoRA).

---

## ⬜ Chapter 18 — API Design & Software Engineering for AI
*Read alongside Huyen's* AI Engineering.
- **18.1–18.4**: REST APIs, FastAPI, response contracts, testing APIs with Pytest.
- **18.5–18.8**: Mocking dependencies in tests, environment variables, CORS, and mini-project (clinical alert service).

---

## ⬜ Chapter 19 — MLOps & Deployment
- **19.1–19.3**: Model serving, experiment tracking, Docker containerisation.
- **19.4**: Project.

---

## ⬜ Chapter 20 — Capstone
- **Rebuild BioPrecision-AI from scratch, by yourself, with full understanding.**
