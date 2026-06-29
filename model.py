from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# -----------------------------
# Load dataset
# -----------------------------
embeddings = torch.load("embeddings.pt")
labels = torch.load("labels.pt")

# Convert lists to tensors
if isinstance(embeddings, list):
    embeddings = torch.tensor(embeddings, dtype=torch.float)

if isinstance(labels, list):
    labels = torch.tensor(labels, dtype=torch.long)

print("Embeddings shape:", embeddings.shape)
print("Labels shape:", labels.shape)
# Fix embedding shape if it has an extra dimension
if embeddings.dim() == 3 and embeddings.shape[1] == 1:
    embeddings = embeddings.squeeze(1)

print("Fixed Embeddings shape:", embeddings.shape)
# -----------------------------
# Check dataset
# -----------------------------
num_nodes = embeddings.shape[0]

if num_nodes != len(labels):
    raise ValueError("Number of embeddings and labels must be equal!")

# -----------------------------
# Build graph
# -----------------------------
if num_nodes == 1:
    # Self-loop for single node
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
else:
    src = []
    dst = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            src.append(i)
            dst.append(j)

    edge_index = torch.tensor([src, dst], dtype=torch.long)

data = Data(
    x=embeddings,
    edge_index=edge_index,
    y=labels
)

# -----------------------------
# Define GCN
# -----------------------------
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)

# -----------------------------
# Create model
# -----------------------------
print("Number of features:", embeddings.shape[1])

model = GCN(
    num_features=embeddings.shape[1],
    num_classes=2
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# Training
# -----------------------------
model.train()

for epoch in range(20):

    optimizer.zero_grad()

    out = model(data)

    loss = F.nll_loss(out, data.y)

    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch+1} Loss = {loss.item():.4f}")

# -----------------------------
# Evaluation
# -----------------------------
model.eval()

with torch.no_grad():

    pred = model(data).argmax(dim=1)

print("\nAccuracy:", accuracy_score(labels, pred))
print(classification_report(labels, pred))

torch.save(model.state_dict(), "model.pth")

print("\nModel saved successfully.")
