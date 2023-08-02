import numpy as np


def normalize_split(split, n=3):
    split = np.array(split[:n], dtype=float)
    split[split < 0] = 0
    split = np.pad(split, (0, n - len(split)), mode="constant")
    split /= np.sum(split)
    return split.tolist()
