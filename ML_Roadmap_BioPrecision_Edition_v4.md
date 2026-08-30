# Arwindh's ML Roadmap — BioPrecision-AI Edition (v4, numbered-sequentially)

> **Why this file exists:** You built BioPrecision-AI — a full clinical AI system — entirely with AI assistance. That's not a problem; it's a huge advantage. But right now the project is a "black box" to you. This roadmap restructures your journey so that by the time you finish it, you will *understand every single line of code and every design decision* made in that project. Every chapter has a **"This is in BioPrecision-AI as..."** connection so you always know why you're learning something.

> **What changed in v4 (August 2026):**
> 1. **Chapter 11 gained a new sub-topic: 11.5 Async & Concurrency.** It was taught out of sequence during the Ch11 sessions and is genuinely required — `ddi_rule` makes one Neo4j round trip per drug pair, and the original early-`return` implementation was found to report only the *first* interaction. The mini-project moves from 11.5 to **11.6**.
> 2. **Notebook standard tightened** (Instruction 7) — notebooks must be self-contained for a reader new to the topic, must record the doubts and corrections from the conversation, and must end with a "never forget" summary.
> 3. **Git commands are now required** in the chat message, not only inside the notebook (Instruction 8).
>
> **What changed in v3 (sequential numbering):**
> 1. **Chapters 1-8 are now grouped as Classical ML (Foundations → Projects).** This resolves the missing Chapter 8 discrepancy and keeps the completed Neural Networks and Deep Learning chapters as Chapters 9 and 10, matching your existing repo layout.
> 2. **Chapters after Chapter 10 are numbered sequentially (11 to 20).** Out-of-order and suffixed numbers (9B, 12B, 12C, 15B) have been eliminated.
> 3. **The chronological flow is fully maintained.** Chapter 12 (Knowledge Graphs) is before Chapter 13 (CNNs), which leads directly into Chapter 14 (GNNs), and Chapter 18 (API Design) is before Chapter 19 (MLOps/Deployment).
> 4. **All internal links, cross-references, diagrams, and technology map mappings are updated to reflect the new sequential numbering.**

> **Current status:**
> - Ch 1–8 (Classical ML): ✅ Complete
> - Ch 9 (Neural Networks 9.1–9.8): ✅ Complete
> - Ch 10 (Deep Learning Essentials): ✅ Complete
> - Ch 11 (Applied Python for AI Systems): 🔄 In progress — **11.1–11.5 complete, next is 11.6 Mini-project**

> **How to use this file:** Upload this together with `ARWINDH_ML_JOURNEY.md` in a new chat. This file is the topic roadmap; that file is the teaching-style manual. Both together give Claude everything needed to pick up cold.

---

## Instructions for Claude in the New Chat

1. **Chapters 1–10 are COMPLETE. Do not re-teach them.** Resume inside Chapter 11.
2. **Teach in full detail, Socratic style:** ask a guiding question before explaining each new idea, let the attempt happen, then correct/confirm — never lecture first. One question at a time, never stacked.
3. **Use real-world analogies, Indian context where natural** (the "Ravi, a loan officer in Chennai" neuron analogy is the established example — reuse or extend it for continuity).
4. **Connect every new idea back to earlier chapters** explicitly before introducing it as new (e.g. "this is the same gradient descent from Ch9.5, just applied differently").
5. **ALSO connect every new idea to BioPrecision-AI** — for example, when teaching Dropout, say "this is the `dropout=0.2` parameter used in the GNN layers of BioPrecision-AI's future extension."
6. **Use diagrams for anything structural** (network architecture, data flow, forward/backward pass).
7. **Ask before building a notebook.** Once a sub-topic lands, confirm the learner is ready rather than producing the notebook unprompted.
8. **Notebook standard.** After each sub-topic is solidly understood, build a notebook — markdown explanation + working code + the learner's own derivations — matching the repo convention (e.g. `11-Applied-Python/11.1-pydantic.md`). It must additionally:
   - be **self-contained**, so a friend new to the topic gets the full idea from the notebook alone (state any prerequisite in one line at the top);
   - **record the doubts clarified and corrections made during the conversation**, marked ❌ / ✅, not just the polished final concept;
   - **verify every code output by executing it** — never paste an unrun result;
   - end with a **"never forget" summary** of the points that carry back to previous chapters and forward to upcoming ones.
9. **Always give the terminal git commands in the chat message**, not only inside the notebook, so the learner can commit and push to `github.com/ArwindhCN/my-ml-journey` without opening the file.
10. **Follow the sequential chapter order in this file.** This file owns the ordering. Do not invent sub-topics or announce a "what's next" that is not listed here — if a genuinely necessary topic is missing, say so explicitly and update this file rather than slipping it into the sequence.

---

## Full Chapter Sequence (CORRECTED ORDER)

| Order | Ch | Topic | Status |
|---|---|---|---|
| 1 | 1–8 | Classical ML (Foundations → Projects) | ✅ Complete |
| 2 | 9 | Neural Networks (9.1–9.8) | ✅ Complete |
| 3 | 10 | Deep Learning Essentials | ✅ Complete |
| 4 | **11** | **Applied Python for AI Systems** | **🔄 IN PROGRESS** |
| 5 | **12** | **Knowledge Graphs & Neo4j** | ⬜ |
| 6 | 13 | Convolutional Neural Networks (CNNs) | ⬜ |
| 7 | 14 | Graph Neural Networks ⭐ | ⬜ |
| 8 | 15 | Explainability & Responsible AI | ⬜ |
| 9 | 16 | NLP & Transformers | ⬜ |
| 10 | 17 | Time Series / RNNs / LSTMs | ⬜ |
| 11 | **18** | **API Design & Software Engineering for AI** | ⬜ |
| 12 | 19 | MLOps & Deployment | ⬜ |
| 13 | 20 | Capstone — Rebuild BioPrecision-AI from scratch | ⬜ |

**Dependency logic:**

```
11  ──► 12  ──► 13 ──► 14 ──► 15
(Pydantic,  (graphs as   (convolution) (message      (explaining
 rules,      data,                      passing)      what 14 did)
 JSON)       Cypher)
                                          │
                    16 ──► 17             │
                 (attention,              │
                  revisits 14.7) ◄────────┘

                    18  ──► 19 ──► 20
                  (REST,   (serving,  (rebuild
                   FastAPI) Docker)    everything)
```

---

## The BioPrecision-AI Technology Map

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
| **Dropout** | `dropout=0.2` in GNN layers | Ch 10.1 ✅ |
| **Early stopping** | `patience=10` in GNN training | Ch 10.1 ✅ |
| **Batch Normalisation** | Used in deep GNN pipelines | Ch 10.2 ✅ |
| **Embeddings** | Every Drug/Gene/Disease node could have an embedding | Ch 10.3 ✅ |
| **PyTorch tensors/autograd** | GNN library built on PyTorch | Ch 10.4 ✅ |
| **Pydantic data models** | `backend/app/models/patient.py` | **Ch 11.1** ← Next! |
| **Rule-based / Symbolic AI** | `decision_engine.py` — DDI, Contraindication rules | **Ch 11.2** |
| **Symbolic AI vs Statistical AI** | Why BioPrecision-AI chose rules over ML | **Ch 11.3** |
| **Structured JSON contracts** | The alert payload sent to the React frontend | **Ch 11.4** |
| **Async / concurrency** | `ddi_rule`'s Neo4j round trip per drug pair | **Ch 11.5** |
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

## ✅ Chapter 10 — Deep Learning Essentials (COMPLETE)

*Bridged plain neural networks (Ch9) to bigger architectures, and Keras (9.7) to PyTorch (what GNNs use).*

| # | Topic | Notebook | BioPrecision-AI connection |
|---|---|---|---|
| 10.1 | **Regularisation** — Dropout, L2/weight decay, Early Stopping | `10.1-regularisation.md` | `dropout=0.2` and `patience=10` in the GNN extension |
| 10.2 | Batch Normalisation | `10.2-batch-norm.md` | Used in deep GNN pipelines |
| 10.3 | Embeddings — a learned dense vector representing something | `10.3-embeddings.md` | Every Drug, Gene, Disease node could have an embedding |
| 10.4 | PyTorch fundamentals — tensors, autograd, `nn.Module` | `10.4-pytorch-basics.md` | PyTorch Geometric is built on PyTorch |
| 10.5 | Mini-project — rebuild the loan-default model in PyTorch | `10.5-pytorch-project.md` | Transition practice before GNNs |

---

## 🔄 Chapter 11 — Applied Python for AI Systems ← YOU ARE HERE

**Status: 11.1 ✅ · 11.2 ✅ · 11.3 ✅ · 11.4 ✅ · 11.5 ✅ · 11.6 ⬜ next**

*Covers the Python patterns used in BioPrecision-AI's backend — not covered anywhere else in the roadmap. You've seen this code — now you'll understand it.*

| # | Topic | BioPrecision-AI as... |
|---|---|---|
| 11.1 | **Pydantic data models** — strict data typing and validation | `backend/app/models/patient.py` — the Patient, Medication, GenomicProfile models |
| 11.2 | **Rule-based AI / Expert Systems** — deterministic if-then reasoning | `decision_engine.py` — DDI rule, Contraindication rule, Polypharmacy rule |
| 11.3 | **Symbolic AI vs Statistical AI** — why and when to use each | BioPrecision-AI chose rules (not ML) for clinical safety — here is why |
| 11.4 | **Structured JSON as a contract** — designing reliable data formats | The alert payload: `{drug, risk_type, severity, evidence, source_database, explanation}` |
| 11.5 | **Async & concurrency** — the event loop, yield points, `gather`, and the blocking-call trap | `ddi_rule` makes one Neo4j round trip per drug pair; the early-`return` version reports only the first interaction found |
| 11.6 | **Mini-project** — write a rule engine that validates a dataset and fires alerts | Rebuild `decision_engine.py` logic from scratch for a new problem |

---

## Chapter 12 — Knowledge Graphs & Graph Databases

*Covers what actually runs in BioPrecision-AI. GNNs (Ch14) learn from graphs in Python. This chapter teaches how to BUILD and QUERY graph databases — exactly what Neo4j does. **This must come before Ch14:** you cannot learn a network that reasons over neighbours before you can construct and traverse a graph yourself.*

| # | Topic | BioPrecision-AI as... |
|---|---|---|
| 12.1 | What is a Knowledge Graph? | BioPrecision-AI's entire Neo4j database IS a knowledge graph |
| 12.2 | Graph schema design — node types and relationship types | `graph/schema/graph_schema.md` — TARGETS, LOCATED_IN, INTERACTS_WITH |
| 12.3 | Neo4j fundamentals — creating nodes, relationships | The `constraints.cypher` file and node creation scripts |
| 12.4 | Cypher query language — MATCH, WHERE, RETURN, MERGE | `graph_service.py` — every query that drives the clinical engine |
| 12.5 | Graph traversal — multi-hop queries | How contraindications are found in the graph |
| 12.6 | Provenance in graphs — storing where data came from | `source_dataset`, `evidence`, `confidence` on every edge |
| 12.7 | Idempotent graph ingestion — MERGE vs CREATE | `ingest_ddi.py` — safe to run twice without duplication |
| 12.8 | Graph integrity — orphan nodes, constraints | The QA test `test_neo4j_orphan_nodes` |
| 12.9 | Mini-project — design a small knowledge graph from scratch | Rebuilding a mini BioPrecision-AI graph from first principles |

---

## Chapter 13 — Convolutional Neural Networks (CNNs)

*Light pass — just enough to firmly grasp convolution before Ch14 generalizes it to graphs. **Kept immediately before Ch14** so the convolution → message-passing bridge stays intact.*

| # | Topic |
|---|---|
| 13.1 | The convolution operation — local patterns, parameter sharing |
| 13.2 | Pooling, feature maps |
| 13.3 | Classic architecture walkthrough (LeNet-style) |
| 13.4 | Project — small image classifier |

---

## Chapter 14 — Graph Neural Networks ⭐

*Built around your GNN research project AND around BioPrecision-AI's knowledge graph. By now you will have already built a real graph in Ch 12 — this chapter puts learning on top of it.*

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

*Note: 14.1 will now be partly revision after 12.1 — that repetition is intentional and cheap. It re-frames the graph you already built as a learning substrate.*

---

## Chapter 15 — Explainability & Responsible AI

*Placed after Ch14 because 15.4 and 15.5 explain the models built there.*

| # | Topic | BioPrecision-AI as... |
|---|---|---|
| 15.1 | Why explainability matters — especially in healthcare | Why BioPrecision-AI uses rules, not a neural network, for clinical decisions |
| 15.2 | Symbolic AI explainability — rule tracing | The exact alert trace: `decision_engine.py` → `graph_service.py` → Neo4j → JSON |
| 15.3 | SHAP for tree models (review + deepen) | Applying SHAP to explain the Random Forest baseline |
| 15.4 | Attention visualisation | Which nodes the model "looked at" |
| 15.5 | GNNExplainer | "Why did the model flag Phentermine as high risk?" |
| 15.6 | Alert design | The 7-field alert format: `{patient, drug, risk_type, severity, evidence, source_database, explanation}` |

---

## Chapter 16 — NLP & Transformers

| # | Topic |
|---|---|
| 16.1 | Word embeddings |
| 16.2 | RNNs (brief preview) |
| 16.3 | Attention mechanism — self-attention, multi-head attention |
| 16.4 | The Transformer architecture |
| 16.5 | **Revisit HANConv** — re-derive what `heads=4` semantic attention was doing |
| 16.6 | Project — small text classifier |

---

## Chapter 17 — Time Series / RNNs / LSTMs

| # | Topic |
|---|---|
| 17.1 | RNN architecture |
| 17.2 | Vanishing gradient in RNNs |
| 17.3 | LSTM / GRU |
| 17.4 | Project |

---

## Chapter 18 — API Design & Software Engineering for AI

*Deployment is downstream of API design. Ch19's "model serving" and Docker chapters assume you already know what an endpoint, a status code, and a JSON contract are — that is taught here.*

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

By this point you will understand every single file in the project. The capstone version should go further:
- Add at least 10 new drugs
- Write your own Cypher DDI queries without help
- Add a `/health` endpoint
- Write 15 Pytest cases
- Add proper `.env.example`

---

## How to Start Your Next Session

Upload this file + `ARWINDH_ML_JOURNEY.md`, then paste:

> "I've completed Chapters 1–10 of my ML journey (Ch1–8 Classical ML + Ch9 Neural Networks + Ch10 Deep Learning Essentials), and Chapter 11 sub-topics 11.1 Pydantic, 11.2 Rule-based AI, 11.3 Symbolic vs Statistical, 11.4 JSON contracts and 11.5 Async & concurrency — all notebooks pushed to github.com/ArwindhCN/my-ml-journey. I'm resuming at **11.6 — the Chapter 11 mini-project**. Read both attached files for my full background and teaching style. This roadmap owns the chapter ordering — follow it exactly and don't invent sub-topics. Teach me Socratically as usual: one guiding question at a time, don't explain before I attempt, correct me directly when I'm wrong. Ask me before building a notebook, and when you do build one make it self-contained with my doubts and corrections included plus a 'never forget' summary. Give me the git commands in the chat, not just in the file."

---

*Updated August 2026 (v4). Supersedes v3, which listed the mini-project at 11.5 and did not include Async & Concurrency.*
