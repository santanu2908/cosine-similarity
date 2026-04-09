import numpy as np
import torch

def cosine_similarity(v1, array_of_vectors):
    """
    Cosine similarity between a vector and either a single vector (1D) or an array of vectors (2D).
    Returns a float for 1D input, or a list of floats for 2D input.
    Safely handles PyTorch tensors (moves to CPU) and NumPy arrays.
    """
    # Handle torch tensors for v1
    if hasattr(v1, "detach"):  # torch tensor
        # detaches gradients and moves to CPU and convert to NumPy array
        v1 = v1.detach().cpu().numpy()
    v1 = np.asarray(v1, dtype=np.float32).ravel()

    # Handle torch tensors for array_of_vectors
    if hasattr(array_of_vectors, "detach"):  # torch tensor
        array_of_vectors = array_of_vectors.detach().cpu().numpy()
    A = np.asarray(array_of_vectors, dtype=np.float32)

    if A.ndim == 1:
        A = A.ravel()
        denom = np.linalg.norm(v1) * np.linalg.norm(A)
        return float(0.0 if denom == 0 else np.dot(v1, A) / denom)

    # 2D case: compute similarities for each row in A
    A = np.atleast_2d(A)
    v1_norm = np.linalg.norm(v1)
    A_norms = np.linalg.norm(A, axis=1)
    print(f"v1_norm: {v1_norm}, A_norms: {A_norms}")
    denom = v1_norm * A_norms
    print(f"Denominator: {denom}")
    with np.errstate(divide='ignore', invalid='ignore'):
        sims = (A @ v1) / np.where(denom == 0, 1.0, denom)
    sims[denom == 0] = 0.0
    return sims.tolist()

# Test cases
# Test with orthogonal vectors
v1 = [1, 0, 0]
v2 = [0, 1, 0]
score = cosine_similarity(v1, v2)
print(round(score, 4))
# score = 0.0  (orthogonal vectors)

# Test with identical vectors
v1 = [1, 2, 3]
v2 = [1, 2, 3]
score = cosine_similarity(v1, v2)
print(round(score, 4))
# score = 1.0  (identical vectors)

# Test with a query and a corpus of vectors
query = [1, 0, 0]
corpus = [
    [1, 0, 0],   # identical
    [0, 1, 0],   # orthogonal
    [1, 1, 0],   # 45 degrees
]
scores = cosine_similarity(query, corpus)
print([round(s, 4) for s in scores])
# scores = [1.0, 0.0, 0.707...]

# Test with PyTorch tensors
v1 = torch.tensor([1.0, 2.0, 3.0])
v2 = torch.tensor([4.0, 5.0, 6.0])
score = cosine_similarity(v1, v2)
print(round(score, 4))
# score = float