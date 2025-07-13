# 🧠 2. Artificial Neural Networks (ANN)

---

## 📘 What is an Artificial Neural Network?

An **Artificial Neural Network (ANN)** is a computational model inspired by the biological neural networks in the human brain. It is made up of layers of interconnected units (neurons), and is capable of learning to perform tasks like classification, regression, and pattern recognition from data.

---

## 🧩 Components of an ANN

1. **Input Layer**  
   - Receives the input features.  
   - Each neuron corresponds to one input feature.

2. **Hidden Layers**  
   - Perform intermediate computations.  
   - The "depth" of the network refers to the number of hidden layers.  
   - Each neuron in a hidden layer takes a weighted sum of inputs, adds a bias, and passes the result through an activation function.

3. **Output Layer**  
   - Produces the final prediction.  
   - Depends on the type of task:  
     - Classification → softmax/sigmoid output  
     - Regression → linear output

4. **Weights & Biases**  
   - **Weights** determine the importance of each input.  
   - **Biases** allow flexibility in shifting the activation function.

---

## 🔁 Working of an ANN (Forward Pass)

For each neuron:

```

Z = (w1 \* x1 + w2 \* x2 + ... + wn \* xn) + b
A = activation(Z)

```

Where:  
- `x` = input values  
- `w` = weights  
- `b` = bias  
- `Z` = weighted sum  
- `A` = activated output

---

## 🎯 Example: Simple ANN with One Hidden Layer

```

Input (X) → \[3 neurons]
↓
Hidden Layer → \[4 neurons, ReLU]
↓
Output Layer → \[1 neuron, Sigmoid]

````

- Input: 3 features  
- Hidden layer: Applies ReLU  
- Output: Binary classification using sigmoid

---

## 🧠 Activation Functions

Used in neurons to introduce **non-linearity**:
- Sigmoid  
- Tanh  
- ReLU  
- Leaky ReLU

---

## ⚙️ Learning in ANN

- **Loss Function**: Measures prediction error  
- **Backpropagation**: Computes gradient of loss w.r.t. weights  
- **Optimizer (e.g., SGD, Adam)**: Updates weights to reduce loss  
- **Epoch**: One full pass through the training dataset  

---

## 🧪 ANN vs Traditional ML

| Feature               | Traditional ML       | ANN                     |
|-----------------------|----------------------|--------------------------|
| Feature Engineering   | Manual               | Automatic (learned)     |
| Handles Non-Linearity | Limited              | Excellent                |
| Works on Raw Data     | Not directly         | Yes (e.g., pixels)       |
| Interpretability      | High                 | Lower                   |

---

## 🛠 PyTorch Example

```python
import torch
import torch.nn as nn

class SimpleANN(nn.Module):
    def __init__(self):
        super(SimpleANN, self).__init__()
        self.fc1 = nn.Linear(3, 4)  # 3 inputs → 4 neurons
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)  # 4 neurons → 1 output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
````

---

## ✅ Key Points

* ANN mimics the brain’s neuron structure using layers and connections.
* It transforms input through weighted sums and activation functions.
* It learns by minimizing loss via backpropagation and optimization.
* More layers and neurons → better capacity, but higher risk of overfitting.

---

