# Quora Duplicate Question Detection
This repository presents a complete solution for detecting duplicate questions using a combination of traditional ML features, graph-based techniques, and fine-tuned BERT transformer models.

##Project Overview
The main aim is to identify whether a pair of questions from Quora are semantically equivalent. The pipeline includes:

- **Text preprocessing**
- **Feature engineering** (TF-IDF, embeddings, syntactic comparisons)
- **Graph-based features**
- **Classical ML models**
- **BERT fine-tuning**
- **Evaluation & interpretability using SHAP**

---
## 1. Preprocessing

- Lowercasing  
- URL and punctuation removal  
- Stopword removal  
- Lemmatization  
- Preprocessed columns: `q1_prep`, `q2_prep`

---

## 2. Feature Engineering (`FeatureCreation`)

Extracted features include:

- **TF-IDF vector similarity**
- **Bigram overlap**
- **Cosine similarity**
- **Sentence embedding similarity** (using `MiniLM-L6-v2`)
- **Jaccard similarity** (bigrams/trigrams)
- **Word/char count difference & ratio**
- **Euclidean distance**
- **Max common word TF-IDF similarity**

---

## 3. Graph Features (`GraphFeatures`)

Using `networkx`, an undirected graph was constructed:

- Nodes: individual questions  
- Edges: question pairs  
- Features:
  - `q1_degree`, `q2_degree` – number of neighbors in the graph

---

## 4. Models & Pipelines

- **Logistic Regression**
- **Random Forest**
- **XGBoost** (with graph + NLP features)
- **BERT Fine-Tuning** (`bert-base-uncased` with layer unfreezing)

---

## 5. Evaluation Results

### Metrics: F1 Score, Log Loss

#### On Test Data

| Model            | F1 Score | Log Loss |
|------------------|----------|----------|
| Logistic Regression | 0.6992   | 0.5017   |
| Random Forest       | 0.7153   | 0.4824   |
| XGBoost (Graph + NLP) | 0.7861 | 0.3412   |
| **BERT Fine-Tuned**  | **0.8801** | **0.3236** |

---

## 6. SHAP Insights

Key feature importances:

| Feature                | Interpretation |
|------------------------|----------------|
| `semantic_sim`         | Highest semantic similarity boosts duplicate prediction |
| `cos_sim`, `tdif_diff` | High similarity between TF-IDF vectors important |
| `q_char_diff`, `word_ratio` | Duplicate questions have more similar length |
| `same_word`            | Duplicates more often begin with the same word |
| `number_common_words`, `n_bigrams` | Word/bigram overlap contributes positively |
| `q1_degree`, `q2_degree` | Higher degree can slightly reduce duplicate probability |

---

## 7. BERT Fine-Tuning

- Model: `bert-base-uncased`
- Layers unfrozen:
  - Last encoder layer
  - Pooler
  - Classification head
- Training settings:

learning_rate = 2e-5
weight_decay = 0.01
batch_size = 8
num_train_epochs = 3

## 8.Future Improvements
Incorporate additional external metadata
Implement model stacking (e.g., XGBoost + BERT)
API deployment using FastAPI / Streamlit

### License
This project is for educational and research use only.

