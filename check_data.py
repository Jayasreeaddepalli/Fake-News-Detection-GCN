import torch

embeddings = torch.load("embeddings.pt")
labels = torch.load("labels.pt")

print("Embeddings type:", type(embeddings))
print("Labels type:", type(labels))

if isinstance(embeddings, list):
    print("Number of embeddings:", len(embeddings))
else:
    print("Embeddings shape:", embeddings.shape)

if isinstance(labels, list):
    print("Number of labels:", len(labels))
    print("Labels:", labels)
else:
    print("Labels shape:", labels.shape)
    print("Labels:", labels)
