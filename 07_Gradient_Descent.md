# 🧮 8. Gradient Descent and Optimization Algorithms

---

## 📘 What is Gradient Descent?

**Gradient Descent** is an optimization algorithm used to minimize the **loss function** by updating the model’s parameters (weights and biases) in the direction of the **negative gradient**.

---

## 🔁 Gradient Descent Formula

```

θ = θ - α \* ∇L(θ)

````

Where:
- `θ` = model parameters (weights)
- `α` = learning rate (step size)
- `∇L(θ)` = gradient of loss w.r.t. parameters

---

## 🚦 Types of Gradient Descent

| Type              | Description |
|-------------------|-------------|
| **Batch GD**      | Uses the entire dataset to compute gradients (slow but stable) |
| **Stochastic GD** | Uses one sample at a time (fast but noisy) |
| **Mini-batch GD** | Uses a small batch of samples (best of both worlds) |

---

## ⚙️ Optimization Algorithms

### 1. **Stochastic Gradient Descent (SGD)**

- Updates weights using one data point at a time  
- Highly fluctuating but often finds good minima

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
````

---

### 2. **Momentum**

* Accelerates SGD by adding a fraction of the previous update
* Helps escape local minima and smooths oscillations

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

---

### 3. **AdaGrad**

* Adapts learning rate individually for each parameter
* Good for sparse data
* Downside: learning rate shrinks too much

---

### 4. **RMSProp**

* Similar to AdaGrad, but uses exponential moving average of squared gradients
* Handles non-stationary objectives well

```python
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
```

---

### 5. **Adam (Adaptive Moment Estimation)**

* Combines Momentum and RMSProp
* Maintains moving average of gradients and squared gradients
* Widely used and generally performs well

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

---

## 📊 Comparison Table

| Optimizer | Adaptive LR | Momentum | Common Use         |
| --------- | ----------- | -------- | ------------------ |
| SGD       | ❌ No        | Optional | Simple tasks       |
| Momentum  | ❌ No        | ✅ Yes    | Faster convergence |
| AdaGrad   | ✅ Yes       | ❌ No     | Sparse data        |
| RMSProp   | ✅ Yes       | ❌ No     | RNNs, NLP          |
| Adam      | ✅ Yes       | ✅ Yes    | Most DL tasks      |

---

## 🧠 Learning Rate Tuning

* Too high → overshooting, unstable
* Too low → slow convergence
* Can use **schedulers** to adjust dynamically

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
```

---

## ✅ Key Points

* Gradient descent is core to learning in deep networks.
* Optimizers affect speed, stability, and final performance.
* Adam is the go-to optimizer in most cases.
* Choosing the right optimizer and learning rate is critical.

---
