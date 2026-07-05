# Chapter 10 — Deep Learning Essentials

Bridging plain neural networks (Chapter 9) to bigger architectures (CNNs, GNNs, Transformers), and bridging Keras to PyTorch — the framework used for the rest of this journey, including the heterogeneous GNN drug-ranking research.

## Status: 🔄 In progress

## Contents

| File | Topic | Status |
|---|---|---|
| [10.1_Regularisation - Dropout, L2, and Early Stopping.md](<10.1_Regularisation - Dropout, L2, and Early Stopping.md>) | Dropout, L2 (weight decay), Early Stopping — three ways to fight overfitting | done |
| [10.2_Batch Normalisation.md](<10.2_Batch Normalisation.md>) | Stabilising the shifting distribution of activations inside deep networks | done |
| [10.3_Embeddings.md](<10.3_Embeddings.md>) | Compact, learned dense vectors for categorical entities, replacing one-hot encoding | done |
| [10.4_PyTorch Fundamentals - Tensors, Autograd, and nn.Module.md](<10.4_PyTorch Fundamentals - Tensors, Autograd, and nn.Module.md>) | Tensors, autograd, `nn.Module` — the framework underneath everything from here on | done |
| [10.5_Mini-Project - Rebuilding the Loan Default Model in PyTorch.md](<10.5_Mini-Project - Rebuilding the Loan Default Model in PyTorch.md>) | Chapter 9.8's Keras model, rebuilt from scratch in PyTorch | done |

## What this chapter covers

**10.1 — Regularisation.** Why gradient descent, left unrestrained, always finds a way to memorize noise in the training set given enough capacity. Three fixes that attack the problem from different angles: Dropout (prevents neurons from co-adapting into fragile cliques), L2 / weight decay (penalizes large weights directly in the loss function), and Early Stopping (halts training the moment validation loss stops improving). All three are typically combined rather than chosen between.

**10.2 — Batch Normalisation.** Deep networks suffer from *internal covariate shift* — the input distribution to any given layer keeps drifting because earlier layers keep updating. BatchNorm forces each layer's output back to a stable distribution (mean 0, variance 1) on every mini-batch, then lets the network learn its own optimal scale and shift (`γ`, `β`) back on top of that stability.

**10.3 — Embeddings.** One-hot encoding breaks down at scale — thousands of categories means thousands of sparse, meaningless dimensions with zero notion of similarity. Embeddings replace this with small, dense, *learned* vectors where semantically similar entities end up close together in the vector space, purely as a side effect of training. This is the exact mechanism behind every node (patient, drug, gene, disease) in the GNN research.

**10.4 — PyTorch Fundamentals.** Keras hides the training loop inside `.fit()`. PyTorch exposes it: tensors (`requires_grad`), autograd (`loss.backward()` computing every gradient automatically via the chain rule), and `nn.Module` (bundling layers into a network, the equivalent of Keras' `Sequential`). This is the exact skeleton used in the real GNN research code.

**10.5 — Mini-Project.** The Chapter 9.8 Chennai bank loan-default classifier, rebuilt end-to-end in PyTorch — combining Dropout, L2, Batch Normalisation, and a hand-written training loop with manual early stopping, all in one architecture.

## Key connection

Everything in this chapter — the explicit training loop, `nn.Module`, Dropout, weight decay, and embeddings — appears unmodified in the real heterogeneous GNN drug-ranking research code. Chapter 10 is the last stop before Chapter 11 (CNNs) and Chapter 12 (Graph Neural Networks), where the same skeleton gets a new kind of layer.

---

*Part of [my-ml-journey](../README.md).*
