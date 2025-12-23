import random
import numpy as np

def color_for_id(global_id: int):
    random.seed(global_id)
    return (
        random.randint(50, 255),
        random.randint(50, 255),
        random.randint(50, 255),
    )

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-6)
    b = b / (np.linalg.norm(b) + 1e-6)
    return float(np.dot(a, b))