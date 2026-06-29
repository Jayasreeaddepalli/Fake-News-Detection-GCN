# 📰 Fake News Detection using BERT + Graph Convolutional Network (GCN)

## 📌 Project Overview

This project implements a Fake News Detection system using **BERT (Bidirectional Encoder Representations from Transformers)** for text embedding generation and **Graph Convolutional Networks (GCN)** for classification. The model is trained on the **FakeNewsNet** dataset to classify news articles as **Real** or **Fake**.

The project demonstrates an end-to-end machine learning pipeline including data preprocessing, feature extraction, graph construction, model training, evaluation, and model saving.

---

## 🚀 Features

- Fake news classification using BERT and GCN
- Text preprocessing and cleaning
- BERT-based feature extraction
- Graph-based learning using PyTorch Geometric
- Model training and evaluation
- GitHub-ready implementation

---

## 🛠️ Technologies Used

- Python
- PyTorch
- PyTorch Geometric
- Hugging Face Transformers (BERT)
- Pandas
- Scikit-learn
- FakeNewsNet Dataset
- Git & GitHub

---

## 📂 Project Structure

```
Fake-News-Detection-GCN/
│
├── FakeNewsNet/
│   └── dataset/
│       ├── politifact_fake.csv
│       ├── politifact_real.csv
│       ├── gossipcop_fake.csv
│       └── gossipcop_real.csv
│
├── preprocess.py
├── model.py
├── test.py
├── embeddings.pt
├── labels.pt
├── model.pth
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

Dataset Used: **FakeNewsNet**

The dataset contains real and fake news articles collected from trusted sources including Politifact and GossipCop.

Dataset files used:

- politifact_fake.csv
- politifact_real.csv
- gossipcop_fake.csv
- gossipcop_real.csv

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Jayasreeaddepalli/Fake-News-Detection-GCN.git

cd Fake-News-Detection-GCN
```

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
```

Activate the environment (Windows):

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Step 1: Generate BERT Embeddings

```bash
python preprocess.py
```

### Step 2: Train the GCN Model

```bash
python model.py
```

### Step 3: Check PyTorch Installation (Optional)

```bash
python test.py
```

---

## 📈 Results

- **Dataset:** FakeNewsNet (Politifact subset)
- **News Articles Processed:** 1056
- **Embedding Model:** BERT Base Uncased
- **Classifier:** Graph Convolutional Network (GCN)
- **Training Epochs:** 20
- **Training Accuracy:** 59.09%

---

## 🔮 Future Improvements

- Improve graph construction using semantic similarity instead of a fully connected graph.
- Introduce train/test split for better model evaluation.
- Perform hyperparameter tuning to improve accuracy.
- Integrate social context and user interaction features.
- Develop a web application for real-time fake news prediction.

---

## 📚 Skills Demonstrated

- Natural Language Processing (NLP)
- Deep Learning
- Graph Neural Networks (GCN)
- Transformer Models (BERT)
- Machine Learning Pipeline Development
- Data Preprocessing
- Python Programming
- Git & GitHub

---

## 👩‍💻 Author

**Jaya Sree Addepalli**

GitHub: https://github.com/Jayasreeaddepalli

---

## 📄 License

This project is developed for educational and academic purposes.
