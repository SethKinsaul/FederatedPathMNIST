# =====================================================
# Federated Learning Simulation on PathMNIST
# Compare FedAvg vs FedPer with Local Test Accuracy
# Author: Seth Kinsaul (2025)
# =====================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import random
import copy
import numpy as np

# ----------------- User-tunable hyperparameters -----------------
SEED = 42
BATCH_SIZE = 64           # batch size for training on clients
EVAL_BATCH_SIZE = 64      # batch size for evaluation
LOCAL_EPOCHS = 3          # local epochs per communication round
COMM_ROUNDS = 10           # communication rounds
LR = 0.001
# ----------------------------------------------------------------

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ---------- 1. Import PathMNIST ----------
try:
    import medmnist
    from medmnist import PathMNIST
except ImportError:
    import os
    os.system("pip install medmnist")
    from medmnist import PathMNIST

# ---------- 2. Device Setup ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- 3. Load Dataset ----------
data_transform = transforms.Compose([transforms.ToTensor()])
train_dataset = PathMNIST(split='train', transform=data_transform, download=False)
test_dataset = PathMNIST(split='test', transform=data_transform, download=True)

# ---------- 4. Non-IID Partitioning ----------
hospital_sizes = [5000, 10000, 15000]  # Hospital A, B, C
class_partitions = [
    [0, 1, 2, 3],
    [4, 5, 6],
    [7, 8]
]

hospital_datasets = []
for classes, size in zip(class_partitions, hospital_sizes):
    indices = []
    for idx in range(len(train_dataset)):
        _, label = train_dataset[idx]
        if isinstance(label, (list, tuple, np.ndarray)):
            lab = int(np.asarray(label).squeeze())
        else:
            lab = int(label)
        if lab in classes:
            indices.append(idx)
    random.shuffle(indices)
    subset = Subset(train_dataset, indices[:size])
    hospital_datasets.append(subset)

hospital_loaders = [DataLoader(d, batch_size=BATCH_SIZE, shuffle=True) for d in hospital_datasets]

print("Data partitioned into hospitals (Non-IID):")
for i, (d, classes) in enumerate(zip(hospital_datasets, class_partitions)):
    print(f"  Hospital {chr(65+i)}: {len(d)} samples, classes {classes}")

# ---------- 5. CNN ----------
class FedPerCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(FedPerCNN, self).__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU()
        )
        self.personal_head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.shared(x)
        x = self.shared_fc(x)
        x = self.personal_head(x)
        return x

    def shared_state_dict(self):
        sd = {}
        for k, v in self.shared.state_dict().items():
            sd[f"shared.{k}"] = v
        for k, v in self.shared_fc.state_dict().items():
            sd[f"shared_fc.{k}"] = v
        return sd

    def load_shared_from_state(self, shared_sd):
        s_state = {k.replace("shared.", ""): v for k, v in shared_sd.items() if k.startswith("shared.")}
        sf_state = {k.replace("shared_fc.", ""): v for k, v in shared_sd.items() if k.startswith("shared_fc.")}
        self.shared.load_state_dict(s_state)
        self.shared_fc.load_state_dict(sf_state)

# ---------- 6. Local Training ----------
def process_labels(labels):
    if isinstance(labels, torch.Tensor):
        y = labels
    else:
        y = torch.tensor(labels)
    y = y.squeeze()
    if y.ndim > 1:
        y = torch.argmax(y, dim=1)
    return y.long()

def train_local(model, dataloader, epochs=LOCAL_EPOCHS, lr=LR):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for images, labels in dataloader:
            images = images.to(device)
            labels = process_labels(labels).to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

# ---------- 7. Federated Averaging (equal weighting) ----------
def federated_avg(models_state_dicts, equal_weight=True):
    n_models = len(models_state_dicts)
    avg_state = {}
    keys = models_state_dicts[0].keys()
    for k in keys:
        acc = None
        for i, sd in enumerate(models_state_dicts):
            v = sd[k].float().cpu()
            if acc is None:
                acc = v.clone()
            else:
                acc += v
        if equal_weight:
            acc /= n_models
        avg_state[k] = acc
    return avg_state

# ---------- 8. Evaluation ----------
def evaluate(model, dataloader):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = process_labels(labels).to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0

def predict_proba(model, dataloader):
    model.to(device)
    model.eval()
    probs = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            p = F.softmax(outputs, dim=1).cpu()
            probs.append(p)
    if len(probs) == 0:
        return torch.empty((0, 9))
    return torch.cat(probs, dim=0)

# ---------- 9. Simulate FedAvg ----------
def simulate_fedavg(rounds=COMM_ROUNDS, local_epochs=LOCAL_EPOCHS):
    print("\n--- FedAvg Simulation ---")
    global_model = FedPerCNN().to(device)
    fedavg_global_accs = []
    fedavg_local_accs = []

    for r in range(rounds):
        print(f"\n=== Communication Round {r+1} ===")
        client_state_dicts = []
        for i, loader in enumerate(hospital_loaders):
            client_model = FedPerCNN().to(device)
            client_model.load_state_dict(global_model.state_dict())
            client_model = train_local(client_model, loader, epochs=local_epochs)
            client_state_dicts.append(client_model.state_dict())

        # Average with equal weighting
        avg_state = federated_avg(client_state_dicts, equal_weight=True)
        global_model.load_state_dict(avg_state)

        test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
        gacc = evaluate(global_model, test_loader)
        fedavg_global_accs.append(gacc)

        local_accs = [evaluate(global_model, DataLoader(loader.dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False))
                      for loader in hospital_loaders]
        fedavg_local_accs.append(local_accs)

        print(f"Global Model Accuracy after Round {r+1}: {gacc:.2f}%")
        print(f"Local Test Accuracy per Hospital: {[f'{a:.2f}%' for a in local_accs]}")

    return global_model, fedavg_global_accs, fedavg_local_accs

# ---------- 10. Simulate FedPer ----------
def simulate_fedper(rounds=COMM_ROUNDS, local_epochs=LOCAL_EPOCHS):
    print("\n--- FedPer Simulation ---")
    global_shared_model = FedPerCNN().to(device)
    local_models = [FedPerCNN().to(device) for _ in hospital_loaders]
    fedper_local_accs = []
    fedper_global_accs = []

    for r in range(rounds):
        print(f"\n=== Communication Round {r+1} ===")
        for i in range(len(local_models)):
            shared_sd = global_shared_model.shared_state_dict()
            local_models[i].load_state_dict(global_shared_model.state_dict(), strict=False)
            local_models[i] = train_local(local_models[i], hospital_loaders[i], epochs=local_epochs)

        shared_state_dicts = [m.shared_state_dict() for m in local_models]
        avg_shared = federated_avg(shared_state_dicts, equal_weight=True)

        # update global shared model
        new_global_state = copy.deepcopy(global_shared_model.state_dict())
        for k, v in avg_shared.items():
            for full_k in new_global_state.keys():
                if full_k.endswith(k.split(".", 1)[1]):
                    new_global_state[full_k] = v
        global_shared_model.load_state_dict(new_global_state)

        local_accs = [evaluate(m, DataLoader(loader.dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False))
                      for m, loader in zip(local_models, hospital_loaders)]
        fedper_local_accs.append(local_accs)

        test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
        probs_list = [predict_proba(m, test_loader) for m in local_models]
        if len(probs_list) > 0:
            avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)
            preds = avg_probs.argmax(dim=1)
            y_true = []
            for _, labs in test_loader:
                y_true.append(process_labels(labs))
            y_true = torch.cat(y_true)
            fedper_global_acc = 100.0 * (preds == y_true).sum().item() / y_true.size(0)
        else:
            fedper_global_acc = 0.0
        fedper_global_accs.append(fedper_global_acc)

        print(f"Local Test Accuracy per Hospital after Round {r+1}: {[f'{a:.2f}%' for a in local_accs]}")
        print(f"FedPer Soft-Voting Global Accuracy after Round {r+1}: {fedper_global_acc:.2f}%")

    return local_models, fedper_local_accs, fedper_global_accs

# ---------- 11. Run Simulations ----------
if __name__ == "__main__":
    print("\nStarting simulations (FedAvg then FedPer)...")
    fedavg_model, fedavg_global_accs, fedavg_local_accs = simulate_fedavg()
    fedper_models, fedper_local_accs, fedper_global_accs = simulate_fedper()

    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    final_fedavg_acc = evaluate(fedavg_model, test_loader)
    final_fedper_global = fedper_global_accs[-1] if len(fedper_global_accs) > 0 else 0.0
    final_fedper_local_acc = [evaluate(m, DataLoader(loader.dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False))
                              for m, loader in zip(fedper_models, hospital_loaders)]

    print("\n=== Final Comparison ===")
    print(f"FedAvg Final Global Accuracy: {final_fedavg_acc:.2f}%")
    print(f"FedPer Soft-Voting Global Accuracy: {final_fedper_global:.2f}%")
    for i, acc in enumerate(final_fedper_local_acc):
        print(f"FedPer Final Local Accuracy Hospital {chr(65+i)}: {acc:.2f}%")