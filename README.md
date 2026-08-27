# Arwindh's ML Roadmap — Post Chapter 10 (BioPrecision-AI Edition)

Learning machine learning from the ground up — starting with the maths, then applying algorithms to real-world problems to understand what works best and when.

> **Why this roadmap exists:** You built BioPrecision-AI — a full clinical AI system — entirely with AI assistance. That's not a problem; it's a huge advantage. But right now the project is a "black box" to you. This roadmap restructures your journey so that by the time you finish it, you will *understand every single line of code and every design decision* made in that project. Every chapter has a **"This is in BioPrecision-AI as..."** connection so you always know why you're learning something.

> **Current status:**
> - Ch 1–8 (Classical ML): ✅ Complete
> - Ch 9 (Neural Networks 9.1–9.8): ✅ Complete — all notebooks pushed
> - Ch 10 (Deep Learning Essentials): ✅ Complete — all notebooks pushed
> - Ch 11 (Applied Python for AI Systems): 🔄 Next up — start here

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
| **08** | End-to-end projects | ✅ Complete |
| **09** | Neural Networks (Keras 9.1–9.8) | ✅ Complete |
| **10** | Deep Learning Essentials (PyTorch 10.1–10.5) | ✅ Complete |
| **11** | **Applied Python for AI Systems** | 🔄 **Up Next** |
| **12** | Knowledge Graphs & Neo4j | ⬜ Planned |
| **13** | Convolutional Neural Networks (CNNs) | ⬜ Planned |
| **14** | Graph Neural Networks (PyTorch Geometric) ⭐ | ⬜ Planned |
| **15** | Explainability & Responsible AI | ⬜ Planned |
| **16** | NLP & Transformers | ⬜ Planned |
| **17** | Time Series / RNNs / LSTMs | ⬜ Planned |
| **18** | API Design & Software Engineering for AI | ⬜ Planned |
| **19** | MLOps & Deployment | ⬜ Planned |
| **20** | Capstone — Rebuild BioPrecision-AI from scratch | ⬜ Planned |

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
└── 10-Deep-Learning-Essentials/ # Regularisation, BatchNorm, Embeddings, PyTorch Basics
```

---

## 🧬 BioPrecision-AI Technology Map

Every concept used in BioPrecision-AI, and exactly where in this roadmap you will learn it:

| Concept Used in BioPrecision-AI | Where in the Project | Learn It In |
|---|---|---|
| **Random Forest Regressor** | `ml/train_baseline.py` | Ch1-8 ✅ Already know! |
| **Scikit-learn pipelines** | `ml/train_baseline.py` | Ch1-8 ✅ |
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
| **Pydantic data models** | `backend/app/models/patient.py` | **Ch 11.1** (Next) |
| **Rule-based / Symbolic AI** | `decision_engine.py` — DDI, Contraindication rules | **Ch 11.2** |
| **Symbolic AI vs Statistical AI** | Why BioPrecision-AI chose rules over ML | **Ch 11.3** |
| **Graph as a data structure** | Neo4j — Drug/Gene/Disease/Variant network | **Ch 12.1** |
| **Knowledge Graph** | The Neo4j database is a Knowledge Graph | **Ch 12.1** |
| **Neo4j & Cypher queries** | `graph_service.py` — every database query | **Ch 12.3/4** |
| **Graph traversal** | How DDI detection works in Neo4j | **Ch 12.5** |
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

---

## 🔄 Upcoming: Chapter 11 — Applied Python for AI Systems

*Covers the Python patterns used in BioPrecision-AI's backend.*

| # | Topic | BioPrecision-AI as... |
|---|---|---|
| 11.1 | **Pydantic data models** — strict data typing and validation | `backend/app/models/patient.py` — Patient, Medication, GenomicProfile |
| 11.2 | **Rule-based AI / Expert Systems** — deterministic reasoning | `decision_engine.py` — DDI, Contraindication, Polypharmacy rules |
| 11.3 | **Symbolic AI vs Statistical AI** — why and when to use each | Rationale behind rules vs ML for safety |
| 11.4 | **Structured JSON as a contract** — designing reliable formats | Alert payload format |
| 11.5 | **Mini-project** — write a rule engine that validates and alerts | Rebuilding `decision_engine.py` from scratch |

---

## ⬜ Chapter 12 — Knowledge Graphs & Graph Databases
- **12.1–12.4**: Knowledge Graphs, schema design, Neo4j, Cypher queries.
- **12.5–12.8**: Graph traversal, data provenance, idempotent ingestion, data integrity.
- **12.9**: Mini-project: Build a knowledge graph.

---

## ⬜ Chapter 13 — Convolutional Neural Networks (CNNs)
- **13.1–13.3**: Convolution, pooling, feature maps, classic architectures (LeNet/ResNet).
- **13.4**: Mini-project: Image classifier in PyTorch.

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
- **16.1–16.4**: Word embeddings, attention mechanism, Transformer architecture, and semantic attention.
- **16.5–16.6**: Revisit HANConv, project: text classifier.

---

## ⬜ Chapter 17 — Sequence Models (RNNs / LSTMs)
- **17.1–17.3**: Recurrent cells, vanishing gradient, LSTM/GRU.
- **17.4**: Project: sequential patient data prediction.

---

## ⬜ Chapter 18 — API Design & Software Engineering for AI
- **18.1–18.4**: REST APIs, FastAPI, response contracts, testing APIs with Pytest.
- **18.5–18.8**: Mocking dependencies in tests, environment variables, CORS, and mini-project (clinical alert service).

---

## ⬜ Chapter 19 — MLOps & Deployment
- **19.1–19.3**: Model serving, experiment tracking, Docker containerisation.
- **19.4**: Project.

---

## ⬜ Chapter 20 — Capstone
- **Rebuild BioPrecision-AI from scratch, by yourself, with full understanding.**
