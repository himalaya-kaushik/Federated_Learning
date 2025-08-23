# final_project.py

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
import flwr as fl
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import List, Tuple

# ==============================================================================
# === 1. Configuration & Setup
# ==============================================================================
# Configuration
NUM_CLIENTS = 4
NUM_ROUNDS = 30
EPOCHS_PER_CLIENT = 1
EPOCHS_CENTRALIZED = 30
BATCH_SIZE = 32
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f" M:S_R[S_R_V] M:S_R[S_R_V] Training on {DEVICE} M:S_R[S_R_V] M:S_R[S_R_V] \n")
torch.manual_seed(42)

# ==============================================================================
# === 2. Model Definition & Data Handling
# ==============================================================================
class Net(nn.Module):
    """A simple CNN for Fashion-MNIST."""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 13 * 13, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        """Defines the forward pass of the model."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(net, trainloader, epochs):
    """Train the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

def test(net, testloader):
    """Validate the model and return loss and accuracy."""
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = loss / len(testloader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy

def load_data():
    """Load Fashion-MNIST with an uneven split for 4 clients."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    full_trainset = FashionMNIST("./data", train=True, download=True, transform=transform)
    testset = FashionMNIST("./data", train=False, download=True, transform=transform)

    # Uneven split (x, y, z, w)
    num_images = len(full_trainset)
    partition_sizes = [int(num_images * 0.4), int(num_images * 0.3), int(num_images * 0.2), int(num_images * 0.1)]
    partition_sizes[-1] += num_images - sum(partition_sizes)
    client_datasets = random_split(full_trainset, partition_sizes, generator=torch.Generator().manual_seed(42))

    trainloaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for ds in client_datasets]
    full_trainloader = DataLoader(full_trainset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloaders, full_trainloader, valloader

# ==============================================================================
# === 3. Serial Learning (Centralized Baseline)
# ==============================================================================
def run_centralized_training(full_trainloader, valloader):
    print(" M:S_R[S_R_V] M:S_R[S_R_V] Starting Serial (Centralized) Training... M:S_R[S_R_V] M:S_R[S_R_V] ")
    model = Net().to(DEVICE)
    history = {"loss": [], "accuracy": []}
    
    for epoch in range(EPOCHS_CENTRALIZED):
        train(model, full_trainloader, epochs=1)
        loss, accuracy = test(model, valloader)
        print(f"  Epoch {epoch+1}/{EPOCHS_CENTRALIZED} -> Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        history["loss"].append(loss)
        history["accuracy"].append(accuracy)
    
    print(" M:S_R[S_R_V] M:S_R[S_R_V] Centralized Training Finished M:S_R[S_R_V] M:S_R[S_R_V] \n")
    return history

# ==============================================================================
# === 4. Cluster Method (Federated Learning with Flower)
# ==============================================================================
# --- Flower Client Definition ---
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.trainloader, epochs=EPOCHS_PER_CLIENT)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

# --- Client Factory ---
def client_fn(cid: str) -> FlowerClient:
    model = Net().to(DEVICE)
    trainloader = client_trainloaders[int(cid)]
    return FlowerClient(model, trainloader, valloader)

# --- Server-side Evaluation Function ---
def get_evaluate_fn(test_loader):
    def evaluate(server_round, parameters, config):
        model = Net().to(DEVICE)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        loss, accuracy = test(model, test_loader)
        print(f"  Round {server_round}: Global Model Loss -> {loss:.4f}, Global Model Accuracy -> {accuracy:.4f}")
        return loss, {"accuracy": accuracy}
    return evaluate

# ==============================================================================
# === 5. Main Execution and Comparison
# ==============================================================================
if __name__ == "__main__":
    # --- Load Data ---
    client_trainloaders, full_trainloader, valloader = load_data()
    
    # --- Run and Time Centralized Baseline ---
    start_time_centralized = time.time()
    centralized_history = run_centralized_training(full_trainloader, valloader)
    end_time_centralized = time.time()
    
    # --- Run and Time Federated Learning ---
    print(" M:S_R[S_R_V] M:S_R[S_R_V] Starting Cluster (Federated) Training... M:S_R[S_R_V] M:S_R[S_R_V] ")
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=get_evaluate_fn(valloader),
    )
    
    start_time_federated = time.time()
    federated_history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.25} if DEVICE.type == "cuda" else {"num_cpus": 1}
    )
    end_time_federated = time.time()
    print(" M:S_R[S_R[S_R_V] M:S_R[S_R_V] M:S_R[S_R_V] Federated Training Finished M:S_R[S_R[S_R_V] M:S_R[S_R_V] M:S_R[S_R_V] \n")
    
    # --- Print Time Comparison ---
    print(" M:S_R[S_R[S_R_V] M:S_R[S_R_V] M:S_R[S_R_V] ----- Training Time Comparison ----- M:S_R[S_R[S_R_V] M:S_R[S_R_V] M:S_R[S_R_V] ")
    print(f"  Total time for Centralized Training: {end_time_centralized - start_time_centralized:.2f} seconds")
    print(f"  Total time for Federated Training:   {end_time_federated - start_time_federated:.2f} seconds")
    print(" M:S_R[S_R[S_R_V] M:S_R[S_R_V] M:S_R[S_R_V] ------------------------------------ M:S_R[S_R[S_R_V] M:S_R[S_R_V] M:S_R[S_R_V] \n")
    
    # --- Visualize the Comparison ---
    print(" M:S_R[S_R[S_R_V] M:S_R[S_R_V] Generating comparison plots... M:S_R[S_R[S_R_V] M:S_R[S_R_V] ")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Plot Accuracy
    fed_rounds = [x for x, y in federated_history.metrics_centralized["accuracy"]]
    fed_accuracy = [y for x, y in federated_history.metrics_centralized["accuracy"]]
    ax1.plot(fed_rounds, fed_accuracy, marker='o', label='Federated Learning')
    ax1.plot(range(1, EPOCHS_CENTRALIZED + 1), centralized_history["accuracy"], marker='x', linestyle='--', label='Centralized Learning')
    
    ax1.set_title("Accuracy Comparison", fontsize=16)
    ax1.set_xlabel("Epoch / Federated Round", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.grid(True)
    ax1.legend()
    ax1.set_xticks(np.arange(0, max(NUM_ROUNDS, EPOCHS_CENTRALIZED) + 1, 5))
    ax1.set_ylim(0.7, 0.95)

    # Plot Loss
    fed_loss = [y for x, y in federated_history.losses_centralized]
    ax2.plot(fed_rounds, fed_loss, marker='o', label='Federated Learning')
    ax2.plot(range(1, EPOCHS_CENTRALIZED + 1), centralized_history["loss"], marker='x', linestyle='--', label='Centralized Learning')

    ax2.set_title("Loss Comparison", fontsize=16)
    ax2.set_xlabel("Epoch / Federated Round", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.grid(True)
    ax2.legend()
    ax2.set_xticks(np.arange(0, max(NUM_ROUNDS, EPOCHS_CENTRALIZED) + 1, 5))
    
    # --- Print Time Comparison ---
    print("\n================= Training Time Comparison =================")
    centralized_time = end_time_centralized - start_time_centralized
    federated_time = end_time_federated - start_time_federated
    print(f"Centralized Training Time: {centralized_time:.2f} seconds")
    print(f"Federated Training Time:   {federated_time:.2f} seconds")
    print("============================================================\n")

    # --- Visualize the Comparison ---
    print("Generating comparison plots...")


    plt.tight_layout()
    plt.show()


