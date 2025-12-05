# =====================================================
# Federated Learning Simulation on PathMNIST
# Compare FedAvg vs FedPer with Local Test Accuracy
# Author: Seth Kinsaul (2025) - improved version
# =====================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from matplotlib.ticker import MaxNLocator

# ----------------- User-tunable hyperparameters -----------------
SEED = 42
BATCH_SIZE = 64           # batch size for training on clients
EVAL_BATCH_SIZE = 64      # batch size for evaluation
LOCAL_EPOCHS = 3         # local epochs per communication round
COMM_ROUNDS = 10          # communication rounds
NUM_CLASSES = 9           # PathMNIST has 9 classes
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
train_dataset = PathMNIST(split='train', transform=data_transform, download=True)
val_dataset   = PathMNIST(split='val',   transform=data_transform, download=True)
test_dataset = PathMNIST(split='test', transform=data_transform, download=True)

val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)  # shared validation loader

# ---------- 4. Non-IID Partitioning ----------
hospital_sizes = [15000, 10000, 5000]  # Hospital A, B, C
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

hospital_A_val_idx = [i for i, (_, y) in enumerate(val_dataset) if y in class_partitions[0]]
hospital_B_val_idx = [i for i, (_, y) in enumerate(val_dataset) if y in class_partitions[1]]
hospital_C_val_idx = [i for i, (_, y) in enumerate(val_dataset) if y in class_partitions[2]]

# Create Subsets
hospital_A_val = Subset(val_dataset, hospital_A_val_idx)
hospital_B_val = Subset(val_dataset, hospital_B_val_idx)
hospital_C_val = Subset(val_dataset, hospital_C_val_idx)

# Create DataLoaders
val_loaders = [
    DataLoader(hospital_A_val, batch_size=EVAL_BATCH_SIZE, shuffle=False),
    DataLoader(hospital_B_val, batch_size=EVAL_BATCH_SIZE, shuffle=False),
    DataLoader(hospital_C_val, batch_size=EVAL_BATCH_SIZE, shuffle=False)
]

# ---------- 5. CNN ----------
class FedPerCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # ===== SHARED FEATURE EXTRACTOR =====
        self.shared = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),  # 32x32 → 16x16

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),  # 16x16 → 8x8
        )

        # ===== PERSONAL HEAD =====
        self.personal = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(256),      # Auto-detect input size on first run
            nn.ReLU(),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.shared(x)
        x = self.personal(x)
        return x

    # Return only shared parameters (for FedPer aggregation)
    def shared_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if k.startswith("shared.")}

    # Load JUST the shared part into this model
    def load_shared(self, shared_state):
        current = self.state_dict()
        for k, v in shared_state.items():
            if k in current:
                current[k] = v
        self.load_state_dict(current)

    # Freeze shared layers during local training
    def freeze_shared(self):
        for p in self.shared.parameters():
            p.requires_grad = False

    def unfreeze_shared(self):
        for p in self.shared.parameters():
            p.requires_grad = True


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

def train_local(model, dataloader, epochs=LOCAL_EPOCHS, lr=LR, freeze_shared=False):
    model.to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    criterion = nn.CrossEntropyLoss()
    model.train()

    if freeze_shared:
        model.freeze_shared()
    else:
        model.unfreeze_shared()

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

def compute_metrics(model, dataloader):
    model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = process_labels(labels).to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            y_true.append(labels.cpu())
            y_pred.append(preds.cpu())
    if len(y_true) == 0:
        return 0.0, 0.0, 0.0
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return precision, recall, f1

# ---------- 9. Simulate FedAvg ----------
def simulate_fedavg(rounds=COMM_ROUNDS, local_epochs=LOCAL_EPOCHS):
    print("\n--- FedAvg Simulation ---")
    global_model = FedPerCNN().to(device)
    
    fedavg_global_accs = []
    fedavg_local_accs = []  # local accuracies (global model evaluated on each hospital's validation set)
    fedavg_local_precisions = []
    fedavg_local_recalls = []
    fedavg_local_f1s = []

    for r in range(rounds):
        print(f"\n=== Communication Round {r+1} ===")
        client_state_dicts = []

        # ---------- Local training ----------
        for i, loader in enumerate(hospital_loaders):
            client_model = FedPerCNN().to(device)
            client_model.load_state_dict(global_model.state_dict())
            client_model = train_local(client_model, loader, epochs=local_epochs, freeze_shared=False)
            client_state_dicts.append(client_model.state_dict())

        # ---------- Aggregate to update global model ----------
        avg_state = federated_avg(client_state_dicts, equal_weight=True)
        global_model.load_state_dict(avg_state)

        # ---------- Evaluate global model ----------
        # Global test accuracy
        gacc = evaluate(global_model, DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False))
        fedavg_global_accs.append(gacc)

        # Global model performance per hospital validation set
        local_accs = [evaluate(global_model, val_loader) for val_loader in val_loaders]
        fedavg_local_accs.append(local_accs)

        local_metrics = [compute_metrics(global_model, val_loader) for val_loader in val_loaders]
        precisions = [m[0] * 100 for m in local_metrics]
        recalls = [m[1] * 100 for m in local_metrics]
        f1s = [m[2] * 100 for m in local_metrics]

        fedavg_local_precisions.append(precisions)
        fedavg_local_recalls.append(recalls)
        fedavg_local_f1s.append(f1s)

        print(f"Local Validation Accuracy per Hospital: {[f'{a:.2f}%' for a in local_accs]}")
        print(f"Local Precision per Hospital: {[f'{p:.2f}%' for p in precisions]}")
        print(f"Local Recall per Hospital: {[f'{r:.2f}%' for r in recalls]}")
        print(f"Local F1 per Hospital: {[f'{x:.2f}%' for x in f1s]}")
        print(f"Global Model Accuracy after Round {r+1}: {gacc:.2f}%")

    return (global_model,
            fedavg_global_accs,
            fedavg_local_accs,
            fedavg_local_precisions,
            fedavg_local_recalls,
            fedavg_local_f1s)

# ---------- 10. Simulate FedPer ----------
def simulate_fedper(rounds=COMM_ROUNDS, local_epochs=LOCAL_EPOCHS):
    print("\n--- FedPer Simulation (Improved) ---")

    # Global shared model
    global_model = FedPerCNN().to(device)

    # One personalized model per hospital
    local_models = [FedPerCNN().to(device) for _ in hospital_loaders]

    # Pre-calc dataset sizes for weighted averaging
    local_sizes = [len(loader.dataset) for loader in hospital_loaders]

    # Metric storage
    local_accs_all, local_prec_all, local_rec_all, local_f1_all, soft_accs_all = [], [], [], [], []

    for r in range(rounds):
        print(f"\n=== Communication Round {r+1} ===")

        # 1. Distribute shared layers to each hospital
        shared_state = global_model.shared_state_dict()
        for m in local_models:
            m.load_shared(shared_state)

        # 2. Train personalized layers locally
        for i in range(len(local_models)):
            local_models[i] = train_local(local_models[i], hospital_loaders[i], epochs=local_epochs, freeze_shared=True)

        # 3. Weighted aggregation of shared layers
        shared_states = [m.shared_state_dict() for m in local_models]
        avg_shared = federated_avg_weighted(shared_states, local_sizes)

        # 4. Update global shared model
        global_model.load_shared(avg_shared)

        # 5. Local evaluations
        round_acc, round_prec, round_rec, round_f1 = [], [], [], []

        for m, v_loader in zip(local_models, val_loaders):
            acc = evaluate(m, v_loader)
            p, r, f1 = compute_metrics(m, v_loader)
            round_acc.append(acc)
            round_prec.append(p * 100)
            round_rec.append(r * 100)
            round_f1.append(f1 * 100)
        
        # 5. Soft-voting global accuracy for this round
        soft_acc = fedper_soft_voting(local_models, DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE))
        soft_accs_all.append(soft_acc)

        local_accs_all.append(round_acc)
        local_prec_all.append(round_prec)
        local_rec_all.append(round_rec)
        local_f1_all.append(round_f1)

        print(f"Local Validation Accuracy per Hospital: {[f'{a:.2f}%' for a in round_acc]}")
        print(f"Local Precision per Hospital: {[f'{p:.2f}%' for p in round_prec]}")
        print(f"Local Recall per Hospital: {[f'{r:.2f}%' for r in round_rec]}")
        print(f"Local F1 per Hospital: {[f'{f:.2f}%' for f in round_f1]}")
        print(f"FedPer Soft Voting Global Accuracy: {soft_acc:.2f}%")

    return (local_models,
            local_accs_all,
            local_prec_all,
            local_rec_all,
            local_f1_all,
            soft_accs_all)

def fedper_soft_voting(local_models, dataloader):
    """
    Compute soft-voting predictions across all local models.
    Returns overall accuracy.
    """
    all_probs = []
    for m in local_models:
        m.to(device)
        m.eval()
        probs = []
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                outputs = F.softmax(m(images), dim=1).cpu()
                probs.append(outputs)
        if probs:
            probs = torch.cat(probs, dim=0)
            all_probs.append(probs)
    
    avg_probs = torch.mean(torch.stack(all_probs), dim=0)
    y_pred = torch.argmax(avg_probs, dim=1)

    y_true = []
    with torch.no_grad():
        for _, labels in dataloader:
            labels = process_labels(labels)
            y_true.append(labels)
    y_true = torch.cat(y_true)

    acc = (y_pred == y_true).sum().item() / len(y_true) * 100
    return acc

def federated_avg_weighted(shared_dicts, local_sizes):
    avg = {}

    total_size = sum(local_sizes)

    for key in shared_dicts[0].keys():
        weighted_sum = None

        for state, size in zip(shared_dicts, local_sizes):
            if weighted_sum is None:
                weighted_sum = state[key] * (size / total_size)
            else:
                weighted_sum += state[key] * (size / total_size)

        avg[key] = weighted_sum.clone()

    return avg

# ---------- 11. Run Simulations ----------
if __name__ == "__main__":
    print("\nStarting simulations (FedAvg then FedPer)...")
    # Unpack FedAvg
    (fedavg_model,
    fedavg_global_accs,
    fedavg_local_accs,
    fedavg_local_precisions,
    fedavg_local_recalls,
    fedavg_local_f1s) = simulate_fedavg()
    # Unpack FedPer
    (fedper_models,
    fedper_local_accs,
    fedper_local_precisions,
    fedper_local_recalls,
    fedper_local_f1s,
    soft_accs_all) = simulate_fedper()
    
    # Evaluate FedAvg final global accuracy
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    final_fedavg_acc = evaluate(fedavg_model, test_loader)

    # Compare FedPer mean local accuracy vs FedAvg global accuracy
    fedper_mean_acc = [np.mean(accs) for accs in fedper_local_accs]
    fedavg_final_global = evaluate(fedavg_model, DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE))

    # Final round local accuracies
    final_fedavg_local_acc = fedavg_local_accs[-1]
    final_fedper_local_acc = fedper_local_accs[-1]

    print("\n=== Final Comparison ===")
    print(f"FedAvg Final Global Accuracy: {final_fedavg_acc:.2f}%")

    for i, acc in enumerate(final_fedavg_local_acc):
        print(f"FedAvg Final Local Accuracy Hospital {chr(65+i)}: {acc:.2f}%")
    for i, acc in enumerate(final_fedper_local_acc):
        print(f"FedPer Final Local Accuracy Hospital {chr(65+i)}: {acc:.2f}%")

    print(f"FedPer Final Soft Voting Global Accuracy: {soft_accs_all[-1]:.2f}%")
        
    # Convert to arrays for easy plotting
    fedavg_local_accs_arr = np.array(fedavg_local_accs)
    fedper_local_accs_arr = np.array(fedper_local_accs)

    rounds = np.arange(1, fedavg_local_accs_arr.shape[0] + 1)
    # -----------------------------
    # Plot 1 — FedAvg Local Accuracy
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, fedavg_local_accs_arr[:,0], marker='o', color='green', label='Hospital A')
    plt.plot(rounds, fedavg_local_accs_arr[:,1], marker='o', color='blue', label='Hospital B')
    plt.plot(rounds, fedavg_local_accs_arr[:,2], marker='o', color='red', label='Hospital C')
    plt.title("FedAvg Local Accuracy per Hospital")
    plt.xlabel("Communication Round")
    plt.ylabel("Local Test Accuracy (%)")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Plot 2 — FedPer Local Accuracy
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, fedper_local_accs_arr[:, 0], marker='o', color='green', label='Hospital A')
    plt.plot(rounds, fedper_local_accs_arr[:, 1], marker='o', color='blue', label='Hospital B')
    plt.plot(rounds, fedper_local_accs_arr[:, 2], marker='o', color='red', label='Hospital C')
    plt.title("FedPer Local Accuracy per Hospital")
    plt.xlabel("Communication Round")
    plt.ylabel("Local Test Accuracy (%)")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -----------------------------------------
    # Plot 3 — FedAvg vs FedPer Global Accuracy
    # -----------------------------------------
    fedavg_global_arr = np.array(fedavg_global_accs)
    fedper_soft_arr   = np.array(soft_accs_all)
    rounds = np.arange(1, len(fedavg_global_arr) + 1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, fedavg_global_arr, marker='o', label='FedAvg (Global)', alpha=0.9)
    plt.plot(rounds, fedper_soft_arr, marker='o', label='FedPer Soft Vote (Global)', alpha=0.9)
    plt.xlabel("Communication Round")
    plt.ylabel("Global Test Accuracy (%)")
    plt.title("FedAvg vs FedPer — Global Accuracy per Communication Round")
    plt.xticks(rounds)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------------
    # Plot 4 — FedAvg Local precision
    # -------------------------------
    fedavg_local_precisions_arr = np.array(fedavg_local_precisions)
    fedper_local_precisions_arr = np.array(fedper_local_precisions)

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, fedavg_local_precisions_arr[:,0], marker='o', color='green', label='Hospital A')
    plt.plot(rounds, fedavg_local_precisions_arr[:,1], marker='o', color='blue', label='Hospital B')
    plt.plot(rounds, fedavg_local_precisions_arr[:,2], marker='o', color='red', label='Hospital C')
    plt.title("FedAvg Local Precision per Hospital (%)")
    plt.xlabel("Communication Round")
    plt.ylabel("Precision (%)")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -------------------------------
    # Plot 5 — FedPer Local precision
    # -------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, fedper_local_precisions_arr[:,0], marker='o', color='green', label='Hospital A')
    plt.plot(rounds, fedper_local_precisions_arr[:,1], marker='o', color='blue', label='Hospital B')
    plt.plot(rounds, fedper_local_precisions_arr[:,2], marker='o', color='red', label='Hospital C')
    plt.title("FedPer Local Precision per Hospital (%)")
    plt.xlabel("Communication Round")
    plt.ylabel("Precision (%)")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Plot 6 — FedAvg Local Recall
    # -----------------------------
    fedavg_local_recalls_arr = np.array(fedavg_local_recalls)
    fedper_local_recalls_arr = np.array(fedper_local_recalls)

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, fedavg_local_recalls_arr[:,0], marker='o', color='green', label='Hospital A')
    plt.plot(rounds, fedavg_local_recalls_arr[:,1], marker='o', color='blue', label='Hospital B')
    plt.plot(rounds, fedavg_local_recalls_arr[:,2], marker='o', color='red', label='Hospital C')
    plt.title("FedAvg Local Recall per Hospital (%)")
    plt.xlabel("Communication Round")
    plt.ylabel("Recall (%)")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Plot 7 — FedPer Local Recall
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, fedper_local_recalls_arr[:,0], marker='o', color='green', label='Hospital A')
    plt.plot(rounds, fedper_local_recalls_arr[:,1], marker='o', color='blue', label='Hospital B')
    plt.plot(rounds, fedper_local_recalls_arr[:,2], marker='o', color='red', label='Hospital C')
    plt.title("FedPer Local Recall per Hospital (%)")
    plt.xlabel("Communication Round")
    plt.ylabel("Recall (%)")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ------------------------------
    # Plot 8 — FedAvg Local F1 Score
    # ------------------------------
    fedavg_local_f1s_arr = np.array(fedavg_local_f1s)
    fedper_local_f1s_arr = np.array(fedper_local_f1s)

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, fedavg_local_f1s_arr[:,0], marker='o', color='green', label='Hospital A')
    plt.plot(rounds, fedavg_local_f1s_arr[:,1], marker='o', color='blue', label='Hospital B')
    plt.plot(rounds, fedavg_local_f1s_arr[:,2], marker='o', color='red', label='Hospital C')
    plt.title("FedAvg F1 Score per Hospital (%)")
    plt.xlabel("Communication Round")
    plt.ylabel("F1 Score (%)")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ------------------------------
    # Plot 9 — FedPer Local F1 Score
    # ------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, fedper_local_f1s_arr[:,0], marker='o', color='green', label='Hospital A')
    plt.plot(rounds, fedper_local_f1s_arr[:,1], marker='o', color='blue', label='Hospital B')
    plt.plot(rounds, fedper_local_f1s_arr[:,2], marker='o', color='red', label='Hospital C')
    plt.title("FedPer F1 Score per Hospital (%)")
    plt.xlabel("Communication Round")
    plt.ylabel("F1 Score (%)")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()    
