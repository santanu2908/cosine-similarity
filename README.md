# Cosine Similarity

A simple implementation of cosine similarity from scratch using NumPy — built for learning how similarity between vectors works under the hood.

Cosine similarity is widely used in LLMs, RAG pipelines, and machine learning to measure how similar two vectors are, regardless of their magnitude.

## What it covers

- Single vector vs single vector (returns a float)
- Single vector vs a corpus of vectors (returns a list of floats)
- Supports Python lists, NumPy arrays, and PyTorch tensors

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy torch
pip freeze > requirement.txt
```

## Run

```bash
python cosine_similarity.py
```

## Example

```python
from cosine_similarity import cosine_similarity

# Single vector vs single vector
score = cosine_similarity([1, 2, 3], [1, 2, 3])
print(round(score, 4))  # 1.0 (identical vectors)

# Query vs corpus
query = [1, 0, 0]
corpus = [[1, 0, 0], [0, 1, 0], [1, 1, 0]]
scores = cosine_similarity(query, corpus)
print([round(s, 4) for s in scores])  # [1.0, 0.0, 0.7071]
```
