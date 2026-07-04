# 09 — Neural Networks

Where classical ML ends and deep learning begins. A single neuron learns the same way linear regression does — gradient descent on a loss. Stack many neurons in layers and you get a system that can learn any pattern, no matter how complex.

## Topics

| # | Topic | Notebook |
|---|-------|----------|
| 9.1 | Perceptron — weighted sum, bias, activation | [9.1-9.3-perceptron-forwardpass-activations.md](./ML_Chapter9_Notebook_9.1-9.3.md) |
| 9.2 | Forward pass — data flowing layer to layer | [9.1-9.3-perceptron-forwardpass-activations.md](./ML_Chapter9_Notebook_9.1-9.3.md) |
| 9.3 | Activation functions — ReLU, Sigmoid, Softmax | [9.1-9.3-perceptron-forwardpass-activations.md](./ML_Chapter9_Notebook_9.1-9.3.md) |
| 9.4 | Loss functions — Binary Cross-Entropy vs MSE | [9.4-loss-functions.md](./ML_Chapter9_Notebook_9.4.md) |
| 9.5 | Backpropagation — chain rule, gradient flow | [9.5-backpropagation.md](./ML_Chapter9_Notebook_9.5.md) |
| 9.6 | Optimisers — SGD vs Adam | [9.6-optimisers.md](./ML_Chapter9_Notebook_9.6.md) |
| 9.7 | Keras basics — building networks in code | [9.7-keras-basics.md](./ML_Chapter9_Notebook_9.7.md) |
| 9.8 | Project — Loan default prediction (Chennai bank) | [9.8-loan-default-project.md](./ML_Chapter9_Notebook_9.8.md) |

## Key Rules

> Use **ReLU** in hidden layers, never Sigmoid — sigmoid derivative ≤ 0.25 kills the gradient in deep networks.

> Use **BCE + Sigmoid** for binary output, never MSE — MSE gradient gets crushed to near-zero exactly when the model is most wrong.

> **Split before scaling.** Fit the scaler on training data only. Breaking this leaks test statistics into training — results become silently invalid.

---

*Part of [my-ml-journey](../README.md)*
