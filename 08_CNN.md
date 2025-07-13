# 🧠 8. Convolutional Neural Networks (CNNs)

---

## 📘 What is a CNN?

A **Convolutional Neural Network (CNN)** is a deep learning architecture designed primarily for **image and spatial data processing**. It automatically extracts features like edges, textures, and objects using specialized layers.

---

## 🧩 Key Components of CNN

### 1. **Convolutional Layer**

- Applies filters/kernels to the input image  
- Detects local features like edges, corners  
- Each filter produces a **feature map**

```

Output = (Input ⊛ Filter) + Bias

```

---

### 2. **Activation Function (ReLU)**

- Introduces non-linearity  
- Applied after each convolution  
- `f(x) = max(0, x)`

---

### 3. **Pooling Layer**

- Reduces spatial dimensions (downsampling)  
- Makes the model translation-invariant  
- Common types:
  - **Max Pooling**: takes the max value
  - **Average Pooling**: takes the mean value

---

### 4. **Fully Connected (FC) Layer**

- After convolutions, the data is flattened and passed through dense layers  
- Used for final classification or regression

---

### 5. **Softmax / Sigmoid Output**

- **Softmax**: multi-class classification  
- **Sigmoid**: binary classification

---

## 🖼️ CNN Architecture Example

```

Input Image (28x28x1)
→ Conv Layer (5x5 filter) → ReLU
→ Max Pooling (2x2)
→ Conv Layer (5x5 filter) → ReLU
→ Max Pooling (2x2)
→ Flatten
→ FC Layer → Output (Softmax)

````

---

## 🔁 Forward Pass in CNN

1. Apply convolution → feature maps  
2. Apply activation (ReLU)  
3. Apply pooling (optional)  
4. Repeat  
5. Flatten → FC → Output layer

---

## 🛠 PyTorch CNN Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # (1 input channel → 16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # for 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 → 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 → 7x7
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
````

---

## 📊 CNN vs ANN

| Feature                | CNN                  | ANN                    |
| ---------------------- | -------------------- | ---------------------- |
| Input Type             | Images, spatial data | Vector data            |
| Feature Extraction     | Automatic (filters)  | Manual or learned      |
| Parameters             | Fewer (shared)       | Many (fully connected) |
| Translation Invariance | ✅ Yes                | ❌ No                   |

---

## 🎯 Applications of CNNs

* Image Classification (e.g., CIFAR-10, MNIST)
* Object Detection (YOLO, SSD)
* Image Segmentation (U-Net)
* Medical Imaging
* Face Recognition

---

## ✅ Key Points

* CNNs are tailored for grid-like data such as images.
* Use convolutional and pooling layers for feature extraction.
* Reduce dimensionality while retaining important features.
* Typically end with fully connected layers and softmax/sigmoid.


## 📘 Why Architecture Variants?

Different CNN architectures have been designed to improve **accuracy**, **efficiency**, and **depth handling**. These variants solve challenges like vanishing gradients, computational cost, and feature extraction quality.

---

## 🔧 Common CNN Variants

---

### 1. **LeNet-5 (1998)**

- One of the earliest CNNs for digit recognition (MNIST)
- Architecture:
```

Input (32x32) → Conv → Pool → Conv → Pool → FC → Output

```
- Used tanh activation, average pooling

---

### 2. **AlexNet (2012)**

- First CNN to win ImageNet Challenge  
- Used ReLU, dropout, and data augmentation  
- Deeper and wider than LeNet

**Key Features:**
- 5 convolutional layers + 3 fully connected layers  
- ReLU activation  
- GPU usage for training

---

### 3. **VGGNet (2014)**

- Simpler architecture using only 3×3 convolutions  
- Depth increased to 16 or 19 layers  
- High number of parameters

**Architecture:**
```

\[Conv3x3 + ReLU] × N → MaxPool → FC → Softmax

```

---

### 4. **GoogLeNet (Inception v1, 2014)**

- Introduced **Inception modules**  
- Combines multiple filter sizes in parallel (1×1, 3×3, 5×5)
- Reduced computation using 1×1 convolutions

---

### 5. **ResNet (2015)**

- Solves **vanishing gradient** problem  
- Introduces **skip (residual) connections**:
```

F(x) + x

````
- Enables training of very **deep networks** (50, 101, 152 layers)

**Residual Block Example:**
```python
def forward(self, x):
  out = F.relu(self.conv1(x))
  out = self.conv2(out)
  return F.relu(out + x)  # Skip connection
````

---

### 6. **MobileNet**

* Lightweight model for mobile and embedded devices
* Uses **depthwise separable convolutions** to reduce computation

---

### 7. **DenseNet**

* Each layer receives input from all previous layers
* Encourages feature reuse
* Reduces number of parameters

---

## 🧠 Comparison Table

| Model     | Depth | Key Innovation         | Pros                |
| --------- | ----- | ---------------------- | ------------------- |
| LeNet     | 7     | Early design           | Simple, fast        |
| AlexNet   | 8     | ReLU, dropout          | High accuracy boost |
| VGGNet    | 16/19 | Uniform 3x3 conv       | Simple, deep        |
| GoogLeNet | 22    | Inception modules      | Efficient           |
| ResNet    | 50+   | Residual connections   | Very deep training  |
| MobileNet | —     | Depthwise convolutions | Mobile-friendly     |
| DenseNet  | —     | Dense connections      | Feature reuse       |

---

## 🖼️ ResNet Block Diagram (Concept)

```
Input
  ↓
Conv → BN → ReLU → Conv
  ↓            ↑
    ←←←← Skip Connection
  ↓
Output
```

---

## ✅ Key Points

* Different architectures solve different problems like vanishing gradients, depth limitations, and efficiency.
* **ResNet** is the most popular backbone for modern vision tasks.
* **MobileNet** is optimized for resource-constrained environments.
* **Inception** and **DenseNet** optimize feature extraction and reuse.

---


