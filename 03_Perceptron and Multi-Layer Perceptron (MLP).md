# ⚛️ 3. Perceptron and Multi-Layer Perceptron (MLP)

---

## 🧠 What is a Perceptron?

A **Perceptron** is the most basic type of neural network, introduced by Frank Rosenblatt in 1958. It is a **binary classifier** that makes decisions by computing a weighted sum of the input features and applying an activation function.

---

## 🧩 Structure of a Perceptron

A single-layer perceptron consists of:
- Input features (`x1, x2, ..., xn`)
- Weights (`w1, w2, ..., wn`)
- Bias (`b`)
- Activation Function (e.g., step function)

**Equation:**
```

y = f(w · x + b)

```
Where:
- `x` = input vector
- `w` = weight vector
- `b` = bias
- `f` = activation function (e.g., step or sign function)

---

## ✅ Example of Perceptron Logic Gate

Perceptrons can solve simple linearly separable problems like:
- AND
- OR

But they **fail** on problems like **XOR**, which are not linearly separable.

---

## ❌ Limitations of Single-Layer Perceptrons

- Can only model linearly separable data
- Cannot solve XOR or complex problems
- Lacks hidden layers for abstraction

---

## 🧠 What is a Multi-Layer Perceptron (MLP)?

A **Multi-Layer Perceptron (MLP)** is a feedforward neural network with one or more **hidden layers**. It overcomes the limitations of a single-layer perceptron by introducing **non-linear activation functions** and depth.

---

## 🏗️ Architecture of an MLP

An MLP typically consists of:
- Input Layer  
- One or more Hidden Layers  
- Output Layer

Each layer is fully connected to the next.

```

Input → Hidden Layer(s) → Output

```

Example (3-4-1 architecture):
- Input: 3 neurons
- Hidden: 4 neurons with ReLU
- Output: 1 neuron with Sigmoid (for binary classification)

---

## 🔁 Forward Pass in MLP

For each layer:
```

Z = W · X + b
A = Activation(Z)

````

Where:
- `Z` = linear transformation
- `A` = non-linear output
- `W`, `b` = weights and biases

---

## 🔙 Backpropagation in MLP

Backpropagation uses the **chain rule** to compute gradients of the loss function with respect to weights. It enables the network to **learn by updating weights** using gradient descent.

---

## ⚙️ Training an MLP

1. Initialize weights and biases
2. Forward pass to compute predictions
3. Compute loss
4. Backward pass to compute gradients
5. Update weights using an optimizer

---

## 🛠 PyTorch Example

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3, 4)  # Input layer: 3 → Hidden: 4
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)  # Hidden: 4 → Output: 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
````

---

## 🔬 Applications of MLP

* Image and digit classification
* Sentiment analysis
* Fraud detection
* Predictive modeling

---

## ✅ Key Points

* Perceptron is the simplest neural model for binary classification.
* MLP introduces hidden layers and non-linearity to solve complex problems.
* MLPs are fully connected feedforward networks.
* Trained using backpropagation and optimization algorithms.

---

