import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

# -------------------------------
# Step 1: Load CSV files
# -------------------------------
fake_df = pd.read_csv("FakeNewsNet/dataset/politifact_fake.csv")
real_df = pd.read_csv("FakeNewsNet/dataset/politifact_real.csv")

# -------------------------------
# Step 2: Keep only the title column
# -------------------------------
fake_df = fake_df[["title"]].copy()
real_df = real_df[["title"]].copy()

# Add labels
fake_df["label"] = 0
real_df["label"] = 1

# Combine datasets
df = pd.concat([fake_df, real_df], ignore_index=True)

# Remove missing titles
df = df.dropna(subset=["title"])

# Clean text
df["clean_text"] = df["title"].astype(str).str.lower()

print("Total news articles:", len(df))

# -------------------------------
# Step 3: Load BERT
# -------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# -------------------------------
# Step 4: Generate embeddings
# -------------------------------
embeddings = []

for i, text in enumerate(df["clean_text"]):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
    embeddings.append(embedding)

    if (i + 1) % 100 == 0:
        print(f"Processed {i+1}/{len(df)} articles")

# -------------------------------
# Step 5: Save files
# -------------------------------
torch.save(embeddings, "embeddings.pt")
torch.save(torch.tensor(df["label"].values), "labels.pt")

print("\nEmbeddings saved successfully!")
print("Labels saved successfully!")
