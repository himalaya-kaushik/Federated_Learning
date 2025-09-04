1. Tools used
   tailscale
   pytorch
   flower - ai

server :
terminal 1: flower-superlink --insecure
terminal 2: flwr run . distributed-setup --stream

client:
terminal 1: flower-supernode --insecure --superlink 100.101.155.80:9092 --node-config "partition-id=0 num-partitions=1"
( here ip can change check using tailscale)
FedCluster is a distributed system designed to harness underutilized consumer GPUs for federated learning.  
It eliminates cloud dependency by enabling **peer-to-peer orchestration** with seamless device discovery and zero-configuration deployment.  

---

##  Features
- **Automatic Device Discovery** – peers connect seamlessly over a secure mesh network.  
- **Custom Federated Aggregation** – integrated with [Flower](https://flower.dev/) to enable faster convergence.  
- **3× Faster Convergence** – achieved through optimized aggregation algorithms while preserving data privacy.  
- **Zero-Cloud Deployment** – utilize underutilized compute resources directly, no cloud provider needed.  

---

## 🛠 Technologies Used
- **Programming:** Python, PyTorch  
- **Federated Learning:** Flower, FedAvg, custom aggregation  
- **Networking:** Tailscale, peer-to-peer orchestration, network programming  
- **Systems:** Distributed Systems, GPU compute  

---

## ⚙️ Architecture
1. **Peer Discovery** → Nodes connect automatically over Tailscale mesh.  
2. **Task Orchestration** → Jobs distributed across heterogeneous GPUs.  
3. **Training & Aggregation** → Local models trained with PyTorch, aggregated via Flower + custom strategies.  
4. **Result Convergence** → Faster training while maintaining data locality & privacy.  

---
