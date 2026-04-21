### Sports StackExchange Post Popularity Prediction

This project predicts the popularity of sports-related posts from the StackExchange dataset using machine learning, the goal is to understand which factors influence user engagement in online communities.

### Problem Statement

Given a post, we want to predict its Score based on:

- Interaction metrics
- Content characteristics

### Dataset

I used the publicly available Sports StackExchange dataset.

![alt text](public/image-2.png)

### First Model (Random Forest Regressor)

I used a **Random Forest Regressor** because and also to understand how a model is trained to:

- It handles non-linear relationships
- Works well with tabular data
- Requires minimal preprocessing

### Tech Stack

- Python
- pandas
- scikit-learn
- matplotlib

### Results

- MSE: ~21
- R²: ~0.30

![alt text](public/image.png)

This indicates moderate predictive performance, which is expected due to the noisy nature of user-generated content.

![alt text](public/image-1.png)

### Embeddings and search

So to complete the features of capstone project I implemented a natural language processing pipeline using sentence embeddings.

### Approach
- We used a pre-trained model from the `sentence-transformers` library (`all-MiniLM-L6-v2`)

![alt text](public/image-3.png)

- Each post is transformed into a dense vector representation

![alt text](clusters.png)

### Semantic Search
- Given a query (e.g., "football rules"), the system:

  ![alt text](public/image-4.png)

  1. Converts the query into an embedding
  2. Computes similarity with all posts
  3. Returns the most relevant results

This enables semantic retrieval instead of simple keyword matching.

## How to Run

```bash
pip install -r requirements.txt
python src/preprocess.py
python src/train.py

```