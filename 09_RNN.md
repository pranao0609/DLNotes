# 🔄 14. Recurrent Neural Networks (RNNs)

---

## 📘 What is an RNN?

A **Recurrent Neural Network (RNN)** is a type of neural network designed to handle **sequential data**, such as time series, text, or audio.

RNNs have a **"memory"** — they take previous outputs as part of the input for the current step, making them suitable for tasks where order and context matter.

---

## 🔁 RNN Architecture

An RNN processes input one element at a time and maintains a hidden state across time steps.

At time step `t`:

```

h\_t = f(Wx \* x\_t + Wh \* h\_{t-1} + b)
y\_t = Why \* h\_t + by

```

Where:
- `x_t`: input at time `t`
- `h_t`: hidden state at time `t`
- `f`: activation function (usually `tanh` or `ReLU`)
- `y_t`: output at time `t`

---

## 🔄 RNN Unrolled

For input sequence: `x₁, x₂, x₃`

```

x₁ → h₁ → y₁
↓
x₂ → h₂ → y₂
↓
x₃ → h₃ → y₃

````

Each time step **shares weights**, allowing the model to generalize across sequence lengths.

---

## 🧠 Why Use RNNs?

- Suitable for **language modeling**, **machine translation**, **speech recognition**, etc.  
- Models **temporal dependencies** in sequences  
- Can be used for **many-to-one**, **one-to-many**, or **many-to-many** sequence problems

---

## 🧩 Types of Sequence Modeling

| Type           | Input       | Output      | Use Case                      |
|----------------|-------------|-------------|-------------------------------|
| One-to-One     | x → y       | Image → Label                |
| One-to-Many    | x → y₁, y₂  | Image Captioning             |
| Many-to-One    | x₁…xₙ → y   | Sentiment Analysis           |
| Many-to-Many   | x₁…xₙ → y₁…yₙ | Machine Translation        |

---

## ⚠️ Limitations of Vanilla RNNs

1. **Vanishing Gradient Problem**  
2. **Short-term memory** — struggles to retain long dependencies  
3. **Slow training** due to sequential processing

---

## 🔍 Improvements Over RNN

- **LSTM (Long Short-Term Memory)** — handles long dependencies  
- **GRU (Gated Recurrent Unit)** — simplified LSTM with similar performance  
- **Bidirectional RNN** — processes sequence forward and backward

---

## 🛠 PyTorch RNN Example

```python
import torch
import torch.nn as nn

rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=1, batch_first=True)
x = torch.randn(5, 3, 10)  # (batch, seq_len, input_size)
h0 = torch.zeros(1, 5, 20)  # (num_layers, batch, hidden_size)

out, hn = rnn(x, h0)
print(out.shape)  # → (5, 3, 20)
````

---

## ✅ Key Points

* RNNs are ideal for sequential/temporal data.
* They maintain a hidden state to remember past information.
* Suffer from vanishing gradients → use LSTM or GRU for long sequences.
* Useful in NLP, speech, music, and time series forecasting.

---
