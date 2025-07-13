# ⚡ 5. Activation Functions

---

## 📘 What is an Activation Function?

An **activation function** introduces non-linearity into a neural network. It helps the model **learn complex patterns** and decision boundaries that cannot be captured using just linear transformations.

Without activation functions, even a deep neural network would behave like a linear model.

---

## 🧠 Why Are They Important?

- Allow networks to learn **non-linear** mappings.
- Enable stacking of multiple layers to create **deep architectures**.
- Without activation functions, the entire network would collapse into a **single linear function** regardless of the number of layers.

---

## 🔢 Common Activation Functions

### 1. **Sigmoid (Logistic)**

```

f(x) = 1 / (1 + e^(-x))

```

- Output: (0, 1)
- Used in binary classification (output layer)
- **Problems**: Vanishing gradient, not zero-centered

---

### 2. **Tanh (Hyperbolic Tangent)**

```

f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

```

- Output: (-1, 1)
- Zero-centered → better than sigmoid
- Still suffers from vanishing gradient

---

### 3. **ReLU (Rectified Linear Unit)**

```

f(x) = max(0, x)

```

- Most commonly used
- Sparse activation → fast computation
- **Problem**: Dying ReLU (neurons stuck at 0)

---

### 4. **Leaky ReLU**

```

f(x) = x if x > 0 else 0.01 \* x

```

- Solves Dying ReLU by allowing a small gradient when x < 0

---

### 5. **Softmax**

```

f(xi) = e^(xi) / Σ(e^xj) for all j

````

- Converts raw outputs into probabilities
- Used in the **output layer of multi-class classification**

---

## 📊 Comparison Table

| Function     | Output Range | Zero-Centered | Used In              |
|--------------|--------------|---------------|----------------------|
| Sigmoid      | (0, 1)       | ❌ No          | Binary classification|
| Tanh         | (-1, 1)      | ✅ Yes         | Hidden layers        |
| ReLU         | [0, ∞)       | ❌ No          | Most hidden layers   |
| Leaky ReLU   | (-∞, ∞)      | ✅ Yes         | Hidden layers        |
| Softmax      | (0, 1)       | ❌ No          | Multi-class output   |

---

## 🛠 PyTorch Examples

```python
import torch
import torch.nn as nn

# ReLU activation
relu = nn.ReLU()
print(relu(torch.tensor([-2.0, 0.0, 2.0])))

# Sigmoid activation
sigmoid = nn.Sigmoid()
print(sigmoid(torch.tensor([-2.0, 0.0, 2.0])))

# Softmax activation
softmax = nn.Softmax(dim=0)
print(softmax(torch.tensor([1.0, 2.0, 3.0])))
````

---

## 🚫 Vanishing Gradient Problem

* **Sigmoid** and **tanh** squash values into small ranges.
* Their derivatives become very small, causing gradients to vanish.
* This slows down or completely halts learning in deep networks.

**Solution:** Use ReLU or variants in hidden layers.

---

## ✅ Key Points

* Activation functions allow networks to model non-linear relationships.
* ReLU is the most widely used in hidden layers.
* Softmax is used in the final layer for multi-class classification.
* The choice of activation function impacts convergence and accuracy.

---

