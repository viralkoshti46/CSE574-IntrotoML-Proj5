__Course__ = "CSE 574 - Intro To Machine Learning"
__author__ = "Viral Koshti"
__email__ = "viraljag@buffalo.edu"
__UBPersonNumber__ = "50761354"

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from PIL import Image

BLOCK_SIZE = 4
NEURONS = 8
EPOCHS = 35
LEARNING_RATE = 0.5
SEED = 574


def read_image(path):
    """Load the original 8-bit grayscale pixels without rescaling."""
    with Image.open(path) as image:
        if image.mode != "L" or image.size != (512, 512):
            raise ValueError("Expected a 512 by 512 8-bit grayscale image")
        return np.asarray(image, dtype=np.float64)


def make_blocks(image, d):
    """Flatten non-overlapping d-by-d patches, ordered row by row."""
    side = image.shape[0]
    if d <= 0 or side % d:
        raise ValueError("Block size must be a positive divisor of 512")
    return (
        image.reshape(side // d, d, side // d, d)
        .transpose(0, 2, 1, 3)
        .reshape(-1, d * d)
    )


@njit(cache=True)
def train_one_epoch(blocks, weights, order, rate):
    """Online competitive updates: only the closest code vector moves."""
    counts = np.zeros(weights.shape[0], dtype=np.int64)
    for t in range(order.size):
        x = blocks[order[t]]
        best = 0
        best_dist = np.inf
        for k in range(weights.shape[0]):
            dist = 0.0
            for j in range(weights.shape[1]):
                difference = x[j] - weights[k, j]
                dist += difference * difference
            if dist < best_dist:
                best_dist = dist
                best = k
        for j in range(weights.shape[1]):
            weights[best, j] = (1.0 - rate) * weights[best, j] + rate * x[j]
        counts[best] += 1
    return counts


def reconstruct(blocks, weights, d):
    """Assign every patch to its closest neuron, with no weight changes."""
    # ||x-w||^2 = ||x||^2 + ||w||^2 - 2*x.dot(w).
    distances = (
        np.sum(blocks * blocks, axis=1)[:, None]
        + np.sum(weights * weights, axis=1)[None, :]
        - 2.0 * blocks @ weights.T
    )
    labels = np.argmin(distances, axis=1)
    side = 512 // d
    image = (
        weights[labels]
        .reshape(side, side, d, d)
        .transpose(0, 2, 1, 3)
        .reshape(512, 512)
    )
    return image


def run_experiment(
    image, d=BLOCK_SIZE, k=NEURONS, epochs=EPOCHS, seed=SEED, rate=LEARNING_RATE
):
    """Train and return the reconstructed image and per-epoch RMS history."""
    if k < 1 or epochs < 1 or not 0.0 < rate < 1.0:
        raise ValueError("Invalid neuron count, epochs, or learning rate")
    blocks = make_blocks(image, d)
    rng = np.random.default_rng(seed)
    weights = blocks[rng.choice(len(blocks), size=k, replace=k > len(blocks))].copy()
    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(blocks))
        counts = train_one_epoch(blocks, weights, order, rate)
        # Restart neurons that received no training examples this epoch.
        empty = np.flatnonzero(counts == 0)
        if len(empty):
            # Far-away patches are useful fresh candidates for empty neurons.
            errors = np.min(
                (
                    np.sum(blocks * blocks, axis=1)[:, None]
                    + np.sum(weights * weights, axis=1)[None, :]
                    - 2.0 * blocks @ weights.T
                ),
                axis=1,
            )
            worst = np.argsort(errors)[-len(empty) :]
            weights[empty] = blocks[worst]
        rebuilt = reconstruct(blocks, weights, d)
        rms = np.sqrt(np.mean((image - rebuilt) ** 2))
        history.append(float(rms))
    return rebuilt, history


def main():
    image = read_image(Path(__file__).resolve().parent / "barbara.png")
    rebuilt, history = run_experiment(image)
    for rms in history:
        print(f"{rms:.6f}")
    plt.imshow(rebuilt, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
