# ⚖️ 6. Loss Functions

---

## 📘 What is a Loss Function?

A **loss function** measures the difference between the predicted output of a neural network and the actual ground truth label. It quantifies **how wrong the model's predictions are**, and serves as the feedback signal during training.

---

## 🎯 Why Are Loss Functions Important?

- They guide the optimization process.
- Help the model **learn better weights** through gradient descent.
- The goal of training is to **minimize the loss**.

---

## 🧩 Types of Loss Functions

### 🔢 For Regression Tasks

---

### 1. **Mean Squared Error (MSE)**

```

L = (1/n) \* Σ(y\_pred - y\_true)^2

```

- Penalizes large errors more heavily
- Smooth and differentiable
- Sensitive to outliers

---

### 2. **Mean Absolute Error (MAE)**

```

L = (1/n) \* Σ|y\_pred - y\_true|

```

- More robust to outliers than MSE
- Doesn’t penalize large errors as heavily

---

### 3. **Huber Loss**

```

If |error| < δ:
L = 0.5 \* error^2
Else:
L = δ \* (|error| - 0.5 \* δ)

```

- Combines benefits of MSE and MAE
- Tunable parameter `δ`

---

### 🔠 For Classification Tasks

---

### 4. **Binary Cross-Entropy (Log Loss)**

```

L = -\[y \* log(p) + (1 - y) \* log(1 - p)]

```

- Used in binary classification  
- `y`: true label (0 or 1)  
- `p`: predicted probability

---

### 5. **Categorical Cross-Entropy**

```

L = -Σ y\_true \* log(y\_pred)

````

- Used for multi-class classification (with softmax)  
- One-hot encoded labels

---

### 6. **Sparse Categorical Cross-Entropy**

- Like categorical cross-entropy but uses **integer labels** instead of one-hot vectors  
- More memory-efficient

---

## 🧠 Choosing the Right Loss Function

| Task Type      | Loss Function               |
|----------------|-----------------------------|
| Binary Classification | Binary Cross-Entropy      |
| Multi-class Classification | Categorical Cross-Entropy |
| Regression (sensitive to large errors) | MSE         |
| Regression (robust to outliers)        | MAE or Huber Loss |

---

## 🛠 PyTorch Examples

```python
import torch
import torch.nn as nn

# MSE Loss for Regression
mse = nn.MSELoss()
y_pred = torch.tensor([2.5])
y_true = torch.tensor([3.0])
print(mse(y_pred, y_true))  # Output: 0.25

# Binary Cross-Entropy
bce = nn.BCELoss()
pred = torch.tensor([0.8])
target = torch.tensor([1.0])
print(bce(pred, target))  # Output: -log(0.8)

# CrossEntropyLoss for Multi-Class
ce = nn.CrossEntropyLoss()
preds = torch.tensor([[1.2, 0.8, 2.5]])  # logits
label = torch.tensor([2])  # class index
print(ce(preds, label))
````

---

## ✅ Key Points

* The loss function defines what the model is optimizing.
* Regression → MSE, MAE, or Huber
* Binary Classification → Binary Cross-Entropy
* Multi-class → Categorical Cross-Entropy
* Must match the model output (e.g., sigmoid with BCE, softmax with CE)

---
