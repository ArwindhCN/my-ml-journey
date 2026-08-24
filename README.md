# Arwindh's ML Roadmap — Post Chapter 10 (BioPrecision-AI Edition)

Learning machine learning from the ground up — starting with the maths, then applying algorithms to real-world problems to understand what works best and when.

> **Why this roadmap exists:** You built BioPrecision-AI — a full clinical AI system — entirely with AI assistance. That's not a problem; it's a huge advantage. But right now the project is a "black box" to you. This roadmap restructures your journey so that by the time you finish it, you will *understand every single line of code and every design decision* made in that project. Every chapter has a **"This is in BioPrecision-AI as..."** connection so you always know why you're learning something.

> **Current status:**
> - Ch 1–7 (Classical ML): ✅ Complete
> - Ch 9 (Neural Networks 9.1–9.8): ✅ Complete — all notebooks pushed
> - Ch 10 (Deep Learning Essentials): ✅ Complete — all notebooks pushed
> - Ch 9B (Applied Python for AI Systems): 🔄 Next up — start here

---

## 🗺️ Progress & Full Chapter Sequence

| Chapter | Topic | Status |
|---|---|---|
| **01** | Foundations — maths & tools | ✅ Complete |
| **02** | Data skills — cleaning & feature engineering | ✅ Complete |
| **03** | Evaluation & metrics | ✅ Complete |
| **04** | Classification algorithms | ✅ Complete |
| **05** | Regression algorithms | ✅ Complete |
| **06** | Unsupervised learning | ✅ Complete |
| **07** | Model selection & tuning | ✅ Complete |
| **09** | Neural Networks (Keras 9.1–9.8) | ✅ Complete |
| **10** | Deep Learning Essentials (PyTorch 10.1–10.5) | ✅ Complete |
| **9B** | **Applied Python for AI Systems** | 🔄 **Up Next** |
| **11** | Convolutional Neural Networks (CNNs) | ⬜ Planned |
| **12** | Graph Neural Networks (PyTorch Geometric) ⭐ | ⬜ Planned |
| **12B** | Knowledge Graphs & Neo4j | ⬜ Planned |
| **12C** | Explainability & Responsible AI | ⬜ Planned |
| **13** | NLP & Transformers | ⬜ Planned |
| **14** | Time Series / RNNs / LSTMs | ⬜ Planned |
| **15** | MLOps & Deployment | ⬜ Planned |
| **15B** | API Design & Software Engineering for AI | ⬜ Planned |
| **16** | Capstone — Rebuild BioPrecision-AI from scratch | ⬜ Planned |

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
├── 07-Model-Selection-&-Tuning/ # GridSearchCV, RandomizedSearchCV, Pipelines
├── 09-Neural-Networks/          # Perceptron, Backpropagation, Optimisers, Keras
└── 10-Deep-Learning-Essentials/ # Regularisation, BatchNorm, Embeddings, PyTorch Basics
```

---

## 🧬 BioPrecision-AI Technology Map

Every concept used in BioPrecision-AI, and exactly where in this roadmap you will learn it:

| Concept Used in BioPrecision-AI | Where in the Project | Learn It In |
|---|---|---|
| **Random Forest Regressor** | `ml/train_baseline.py` | Ch1-7 ✅ Already know! |
| **Scikit-learn pipelines** | `ml/train_baseline.py` | Ch1-7 ✅ |
| **Data normalisation / z-score** | Patient/drug feature scaling | Ch2 ✅ |
| **Train/Val/Test splitting** | ML baseline evaluation | Ch3 ✅ |
| **Activation functions (ReLU)** | Between GNN layers | Ch9.3 ✅ |
| **Loss functions (MSE)** | `train_baseline.py` loss | Ch9.4 ✅ |
| **Backpropagation** | How the ML baseline learns | Ch9.5 ✅ |
| **Optimisers (Adam)** | GNN extension uses Adam | Ch9.6 ✅ |
| **Dropout** | `dropout=0.2` in GNN layers | Ch10.1 ✅ |
| **Early stopping** | `patience=10` in GNN training | Ch10.1 ✅ |
| **Batch Normalisation** | Used in deep GNN pipelines | Ch10.2 ✅ |
| **Embeddings** | Every Drug/Gene/Disease node could have an embedding | Ch10.3 ✅ |
| **PyTorch tensors/autograd** | GNN library built on PyTorch | Ch10.4 ✅ |
| **Pydantic data models** | `backend/app/models/patient.py` | **Ch 9B.1** (Next) |
| **Rule-based / Symbolic AI** | `decision_engine.py` — DDI, Contraindication rules | **Ch 9B.2** |
| **Symbolic AI vs Statistical AI** | Why BioPrecision-AI chose rules over ML | **Ch 9B.3** |
| **Graph as a data structure** | Neo4j — Drug/Gene/Disease/Variant network | **Ch 12.1** |
| **Knowledge Graph** | The Neo4j database is a Knowledge Graph | **Ch 12.1 + 12B** |
| **Node embeddings** | One embedding per Drug, Gene, Disease | **Ch 12.2** |
| **Message passing** | What every GNN layer computes | **Ch 12.3** |
| **Heterogeneous graphs** | 4 node types, 10 relationship types | **Ch 12.5** |
| **Graph attention** | The "HAN" in HANConv | **Ch 12.6** |
| **Neo4j & Cypher queries** | `graph_service.py` — every database query | **Ch 12B.3/4** |
| **Graph traversal** | How DDI detection works in Neo4j | **Ch 12B.5** |
| **Graph ingestion (ETL)** | `ingest_ddi.py` loads CSV into Neo4j | **Ch 8 + 12B.7** |
| **Explainability design** | 7-field alert format | **Ch 12C.6** |
| **REST API + FastAPI** | `backend/app/main.py` | **Ch 15B.1/2** |
| **Pytest + mocking** | `scripts/qa/` — 10 tests, 10/10 pass | **Ch 15B.4/5** |

---

## 📖 Chapter Breakdowns

### Chapter 01–07: Classical Machine Learning ✅
Foundations, data cleaning, evaluation metrics, classification, regression, clustering, PCA, and hyperparameter tuning. All complete.

### Chapter 09: Neural Networks ✅
Where classical ML ends and deep learning begins. Stacking neurons in layers to learn complex patterns.
- Perceptron, activations, BCE/MSE loss, Backpropagation, SGD/Adam, and Keras implementation.

### Chapter 10: Deep Learning Essentials ✅
- **Regularisation (10.1)**: Fight overfitting via dropout, weight decay, and early stopping.
- **Batch Normalisation (10.2)**: Stabilise activations inside deep networks.
- **Embeddings (10.3)**: Learned dense representations for categorical entities.
- **PyTorch Fundamentals (10.4)**: Tensors, autograd, and `nn.Module` abstraction.
- **Mini-Project (10.5)**: Rebuilt the loan default model from scratch in PyTorch.

---

## 🔄 Upcoming: Chapter 9B — Applied Python for AI Systems

*Covers the Python patterns used in BioPrecision-AI's backend.*

| # | Topic | BioPrecision-AI as... |
|---|---|---|
| 9B.1 | **Pydantic data models** — strict data typing and validation | `backend/app/models/patient.py` — Patient, Medication, GenomicProfile |
| 9B.2 | **Rule-based AI / Expert Systems** — deterministic reasoning | `decision_engine.py` — DDI, Contraindication, Polypharmacy rules |
| 9B.3 | **Symbolic AI vs Statistical AI** — why and when to use each | Rationale behind rules vs ML for safety |
| 9B.4 | **Structured JSON as a contract** — designing reliable formats | Alert payload format |
| 9B.5 | **Mini-project** — write a rule engine that validates and alerts | Rebuilding `decision_engine.py` from scratch |

---

## ⬜ Chapter 11 — Convolutional Neural Networks (CNNs)
- **11.1–11.3**: Convolution, pooling, feature maps, classic architectures (LeNet/ResNet).
- **11.4**: Mini-project: Image classifier in PyTorch.

---

## ⬜ Chapter 12 — Graph Neural Networks (GNNs) ⭐
- **12.1–12.4**: Graphs, node embeddings, message passing, and GCNs.
- **12.5–12.7**: Heterogeneous graphs, GAT, and HANConv (from GNN research).
- **12.8–12.9**: Link prediction and drug-ranking capstone.

---

## ⬜ Chapter 12B — Knowledge Graphs & Graph Databases
- **12B.1–12B.4**: Knowledge Graphs, schema design, Neo4j, Cypher queries.
- **12B.5–12B.8**: Graph traversal, data provenance, idempotent ingestion, data integrity.
- **12B.9**: Mini-project: Build a knowledge graph.

---

## ⬜ Chapter 12C — Explainability & Responsible AI
- **12C.1–12C.3**: Why explainability matters, rule tracing, and SHAP.
- **12C.4–12C.6**: Attention maps, GNNExplainer, and clinical alert design.

---

## ⬜ Chapter 13 — NLP & Transformers
- Word embeddings, attention mechanism, Transformer architecture, and semantic attention in HANConv.

---

## ⬜ Chapter 14 — Sequence Models (RNNs / LSTMs)
- Recurrent cells, vanishing gradient, LSTM/GRU, and sequential patient data prediction.

---

## ⬜ Chapter 15 — MLOps & Deployment
- Model serving, experiment tracking, Docker containerisation, and production pipelines.

---

## ⬜ Chapter 15B — API Design & Software Engineering for AI
- REST APIs, FastAPI, Pytest integration testing, mocking dependencies, environments, and CORS.

---

## ⬜ Chapter 16 — Capstone
- **Rebuild BioPrecision-AI from scratch, by yourself, with full understanding.**
