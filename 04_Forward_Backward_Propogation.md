
# 🔁 4. Forward and Backward Propagation

---

## 📘 What is Forward Propagation?

**Forward Propagation** is the process where input data flows through the neural network layer by layer — from input to output — to produce predictions.

### 🔢 Step-by-Step:
For each layer in the network:
```

Z = W · X + b
A = activation(Z)

```
- `X`: Input to the layer  
- `W`: Weights  
- `b`: Biases  
- `Z`: Linear transformation  
- `A`: Output after applying the activation function  

The output `A` is passed as input to the next layer.

---

## 📘 What is Backward Propagation?

**Backward Propagation** (Backpropagation) is the method used to train neural networks by calculating the **gradient of the loss** with respect to each parameter (weights and biases) using the **chain rule** of calculus.

The gradients are then used to **update the weights** in the direction that minimizes the loss function.

---

## 🧠 Why Use Backpropagation?

- Efficiently computes gradients for all layers
- Works with any differentiable loss and activation functions
- Enables the use of gradient-based optimizers like SGD and Adam

---

## 🧩 Key Terminology

| Term               | Description |
|--------------------|-------------|
| **Loss Function**  | Measures how far predictions are from actual values |
| **Gradient**       | Partial derivative of loss w.r.t. weights |
| **Chain Rule**     | Used to compute gradients through multiple layers |
| **Learning Rate**  | Controls how much weights change during updates |

---

## 🔙 Backpropagation Steps

1. **Forward Pass** – Compute outputs and loss  
2. **Compute Gradients** – Use chain rule to compute dL/dW, dL/db  
3. **Update Parameters** – Use optimizer (e.g., SGD)  
```

W = W - learning\_rate \* dL/dW
b = b - learning\_rate \* dL/db

```

---

## 🧮 Mathematical Example

Let’s say we have:
```

Z = W · X + b
A = ReLU(Z)
L = Loss(A, Y)

```

To update weights `W`, compute:
```

dL/dA → dA/dZ → dZ/dW

```
Final gradient:
```

dL/dW = dL/dA \* dA/dZ \* dZ/dW

````

This is computed for every layer, moving backward from the output layer to the input layer.

---

## 🛠 PyTorch Example with Autograd

```python
import torch
import torch.nn as nn

# Sample forward + backward pass
model = nn.Sequential(
    nn.Linear(3, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

# Example input and label
x = torch.randn(1, 3)
y = torch.tensor([[1.0]])

# Loss function
criterion = nn.BCELoss()

# Forward pass
output = model(x)
loss = criterion(output, y)

# Backward pass
loss.backward()

# Access gradients
for param in model.parameters():
    print(param.grad)
````

---

## ⚠️ Important Notes

* Always call `loss.backward()` before updating weights.
* Don’t forget to zero gradients before each backpropagation step.

  ```python
  optimizer.zero_grad()
  ```

---

## ✅ Key Points

* Forward propagation computes the prediction.
* Backward propagation calculates gradients using the chain rule.
* Together, they form the training loop of a neural network.
* PyTorch handles most of this using `autograd`.

---
