# 🧠 ERA V4 Session 4 - MNIST Classification with CNN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rj555/era-v4-session-4/blob/main/ERA_Session_4.ipynb)

> **Assignment for TSAI ERA V4 course session 4** - Building a Convolutional Neural Network for MNIST digit classification

---

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [🏗️ Architecture](#️-architecture)
- [📊 Model Summary](#-model-summary)
- [📈 Training Logs](#-training-logs)
- [🔧 Technical Details](#-technical-details)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)

---

## 🎯 Project Overview

This project implements a **Convolutional Neural Network (CNN)** for classifying handwritten digits from the MNIST dataset. The model achieves high accuracy through a carefully designed architecture with data augmentation and proper training techniques.

### Key Features
- ✅ **Lightweight CNN Architecture** (21,844 parameters)
- ✅ **Data Augmentation** for improved generalization
- ✅ **Learning Rate Scheduling** for optimal convergence
- ✅ **Comprehensive Logging** and visualization
- ✅ **GPU Support** (CUDA compatible)

---

## 🏗️ Architecture

### Network Architecture

The CNN follows a **4-layer convolutional + 2-layer fully connected** design:

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3)    # 1→4 channels, 28×28→26×26
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3)    # 4→8 channels, 26×26→24×24
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3)   # 8→16 channels, 12×12→10×10
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3)  # 16→32 channels, 10×10→8×8
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(512, 30)  # 4×4×32 = 512 → 30
        self.fc2 = nn.Linear(30, 10)   # 30 → 10 (digits 0-9)
```

### Data Flow Architecture

```
Input (28×28×1) 
    ↓
Conv1 + ReLU → (26×26×4)
    ↓
Conv2 + ReLU + MaxPool → (12×12×8)
    ↓
Conv3 + ReLU → (10×10×16)
    ↓
Conv4 + ReLU + MaxPool → (4×4×32)
    ↓
Flatten → (512)
    ↓
FC1 + ReLU → (30)
    ↓
FC2 → (10)
    ↓
Log Softmax → Output Probabilities
```

### Data Augmentation Pipeline

```python
# Training Augmentations
train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22)], p=0.1),  # Random cropping
    transforms.Resize((28, 28)),                                 # Resize back
    transforms.RandomRotation((-15., 15.), fill=0),             # Random rotation
    transforms.ToTensor(),                                       # Convert to tensor
    transforms.Normalize((0.1307,), (0.3081,)),                 # MNIST normalization
])
```

---

## 📊 Model Summary

| **Layer** | **Type** | **Output Shape** | **Parameters** |
|-----------|----------|------------------|----------------|
| Conv2d-1  | Conv2d   | [-1, 4, 26, 26]  | 40             |
| Conv2d-2  | Conv2d   | [-1, 8, 24, 24]  | 296            |
| Conv2d-3  | Conv2d   | [-1, 16, 10, 10] | 1,168          |
| Conv2d-4  | Conv2d   | [-1, 32, 8, 8]   | 4,640          |
| Linear-5  | Linear   | [-1, 30]         | 15,390         |
| Linear-6  | Linear   | [-1, 10]         | 310            |

### Model Statistics
- **Total Parameters**: 21,844
- **Trainable Parameters**: 21,844
- **Model Size**: 0.08 MB
- **Input Size**: 0.00 MB
- **Forward/Backward Pass**: 0.08 MB
- **Estimated Total Size**: 0.17 MB

---

## 📈 Training Logs

### Training Configuration
- **Optimizer**: SGD (lr=0.01, momentum=0.9)
- **Scheduler**: StepLR (step_size=5, gamma=0.1)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 128
- **Epochs**: 20
- **Device**: CPU (CUDA Available: False)

### Performance Metrics

| **Epoch** | **Train Loss** | **Train Accuracy** | **Test Accuracy** |
|-----------|----------------|-------------------|-------------------|
| 1         | 0.2793         | 72.92%            | 95.25%            |
| 2         | 0.1856         | 96.27%            | 97.39%            |
| 3         | 0.1829         | 97.30%            | 97.82%            |
| 4         | 0.1775         | 97.74%            | 98.08%            |
| 5         | 0.1821         | 97.95%            | 97.96%            |
| 6         | 0.1605         | 98.55%            | 98.57%            |
| 7         | 0.1568         | 98.63%            | 98.65%            |
| 8         | 0.1467         | 98.64%            | 98.66%            |
| 9         | 0.1561         | 98.65%            | 98.67%            |
| 10        | 0.1802         | 98.72%            | 98.75%            |
| 11        | 0.1590         | 98.78%            | 98.80%            |
| 12        | 0.1714         | 98.78%            | 98.82%            |

### Key Observations
- 🚀 **Rapid Convergence**: Model reaches 95%+ accuracy within 2 epochs
- 📈 **Stable Training**: Consistent improvement with minimal overfitting
- 🎯 **High Performance**: Final test accuracy of **98.80%**
- ⚡ **Efficient**: Lightweight model with excellent performance

---

## 🔧 Technical Details

### Training Process
1. **Data Loading**: MNIST dataset with custom transforms
2. **Model Initialization**: CNN with Xavier initialization
3. **Training Loop**: 20 epochs with progress tracking
4. **Validation**: Real-time accuracy monitoring
5. **Visualization**: Loss and accuracy plotting

### Key Functions
- `train()`: Handles forward pass, loss calculation, and backpropagation
- `test()`: Evaluates model performance on test set
- `GetCorrectPredCount()`: Calculates prediction accuracy

### Dependencies
```python
torch
torchvision
matplotlib
tqdm
torchsummary
```

---

## 📁 Project Structure

```
era-v4-session-4/
├── 📓 ERA_Session_4.ipynb    # Main Jupyter notebook
├── 📖 README.md              # This documentation
└── 📊 data/                  # MNIST dataset (auto-downloaded)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- PyTorch
- Jupyter Notebook

### Installation
```bash
# Clone the repository
git clone https://github.com/rj555/era-v4-session-4.git
cd era-v4-session-4

# Install dependencies
pip install torch torchvision matplotlib tqdm torchsummary

# Launch Jupyter
jupyter notebook ERA_Session_4.ipynb
```

### Running the Model
1. Open the notebook in Jupyter or Google Colab
2. Run all cells sequentially
3. Monitor training progress in real-time
4. View final results and visualizations

---

## 📊 Results Summary

- ✅ **Final Test Accuracy**: 98.80%
- ✅ **Model Efficiency**: 21,844 parameters
- ✅ **Training Time**: ~20 minutes (CPU)
- ✅ **Convergence**: Stable after 6 epochs

---

<div align="center">

**🎓 TSAI ERA V4 - Session 4 Assignment**

*Building efficient CNNs for image classification*

</div>
