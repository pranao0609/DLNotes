
# 🧠 15. LSTM and GRU

---

## 📘 Why LSTM & GRU?

Vanilla RNNs struggle with **long-term dependencies** due to the **vanishing gradient problem**.  
**LSTM (Long Short-Term Memory)** and **GRU (Gated Recurrent Unit)** were designed to solve this by introducing **gates** that control information flow.

---

## 🔁 LSTM (Long Short-Term Memory)

An LSTM unit maintains two states:
- **Hidden state (`h_t`)**
- **Cell state (`c_t`)** — acts like memory

### 🧩 Gates in LSTM

1. **Forget Gate (`f_t`)**: Decides what to forget  
2. **Input Gate (`i_t`)**: Decides what new info to store  
3. **Cell State Update (`ĉ_t`)**: Creates new memory  
4. **Output Gate (`o_t`)**: Decides what to output

### 🔢 LSTM Equations

```

- \( x_t \): input at time step \( t \)  
- \( h_{t-1} \): previous hidden state  
- \( c_{t-1} \): previous cell state  
- \( f_t \): forget gate  
- \( i_t \): input gate  
- \( \tilde{c}_t \): candidate cell state  
- \( o_t \): output gate  
- \( c_t \): current cell state  
- \( h_t \): current hidden state  
- \( \sigma \): sigmoid activation  
- \( \tanh \): hyperbolic tangent activation

Equations:

```math
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)

```

---

## 🔁 GRU (Gated Recurrent Unit)

A simplified version of LSTM. Combines forget and input gates into an **update gate**.

### 🧩 Gates in GRU

1. **Update Gate (`z_t`)**: Controls how much of the past to keep  
2. **Reset Gate (`r_t`)**: Controls how to combine new input with past memory

### 🔢 GRU Equations

```

z\_t = σ(W\_z · \[h\_{t-1}, x\_t])
r\_t = σ(W\_r · \[h\_{t-1}, x\_t])
ĥ\_t = tanh(W · \[r\_t \* h\_{t-1}, x\_t])
h\_t = (1 - z\_t) \* h\_{t-1} + z\_t \* ĥ\_t

````

---

## 🧪 LSTM vs GRU

| Feature         | LSTM                 | GRU                  |
|------------------|----------------------|-----------------------|
| Gates            | 3 (input, forget, output) | 2 (update, reset)    |
| Memory Cell      | Yes (`c_t`)         | No                   |
| Computation      | Slightly slower     | Faster               |
| Accuracy         | Slightly better for long sequences | Competitive     |
| Simplicity       | More complex        | Simpler              |

---

## 🛠 PyTorch Example

```python
import torch
import torch.nn as nn

# LSTM
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=1, batch_first=True)
x = torch.randn(5, 3, 10)  # (batch, seq_len, input_size)
h0 = torch.zeros(1, 5, 20)  # initial hidden state
c0 = torch.zeros(1, 5, 20)  # initial cell state
out, (hn, cn) = lstm(x, (h0, c0))

# GRU
gru = nn.GRU(input_size=10, hidden_size=20, num_layers=1, batch_first=True)
out, hn = gru(x, h0)
````

---

## 📊 When to Use What?

* **LSTM**: Better for very long sequences (e.g., text generation, language modeling)
* **GRU**: Faster, good for real-time systems, shorter sequences

---

## ✅ Key Points

* Both LSTM and GRU are advanced RNNs that address memory limitations.
* LSTM uses a separate cell state, GRU combines gates for simplicity.
* GRU is faster; LSTM handles long dependencies better.
* Widely used in NLP, time series, speech recognition, etc.

---

