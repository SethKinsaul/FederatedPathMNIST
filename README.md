**Federated Learning on PathMNIST: FedAvg vs FedPer**
This project implements a federated learning simulation on the PathMNIST dataset (colon pathology images) to compare the performance of FedAvg and FedPer approaches. The goal is to demonstrate how personalization in federated learning can improve local model performance while maintaining competitive global accuracy.

**Features Non-IID dataset partitioning:** 
PathMNIST data split across three simulated hospitals, each with different class distributions.

**Federated learning algorithms:**
FedAvg: Traditional federated averaging of all model weights.
FedPer: Shared CNN layers with personalized output heads for each hospital.
Convolutional Neural Network (CNN):
Two shared convolutional blocks with ReLU and MaxPooling
Flatten and fully connected shared layers
Personalized output head per hospital for classification into 9 classes
Evaluation: Tracks both local hospital accuracy and global accuracy using soft-voting aggregation for FedPer.
PyTorch implementation with CUDA support for GPU acceleration.

**Training Details**
Input: 28×28 RGB images
Optimizer: Adam, learning rate 0.001
Batch size: 64
Local epochs: 3 per communication round
Communication rounds: 10
Loss function: Cross-Entropy

**Results:**
FedPer consistently achieves higher local test accuracy for each hospital compared to FedAvg.
Soft-voting aggregation improves global accuracy, though it can be sensitive to class imbalance across hospitals.

**Usage**
1. Clone the repository
2. Install dependencies: pip install -r requirements.txt (includes medmnist, torch, torchvision)
3. Run the simulation: python FederatedPathMNIST.py
