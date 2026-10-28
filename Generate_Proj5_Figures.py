__Course__ = "CSE 574 - Intro To Machine Learning"
__author__ = "Viral Koshti"
__email__ = "viraljag@buffalo.edu"
__UBPersonNumber__ = "50761354"

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Proj5 import read_image, run_experiment, BLOCK_SIZE, EPOCHS, SEED, LEARNING_RATE


def main():
    folder = Path(__file__).resolve().parent
    original = read_image(folder / "barbara.png")
    results = {}

    for k in (2, 8, 32):
        reconstructed, history = run_experiment(
            original, d=BLOCK_SIZE, k=k, epochs=EPOCHS, seed=SEED, rate=LEARNING_RATE
        )
        results[k] = (reconstructed, history)
        plt.imsave(
            folder / f"Project5_K{k}_reconstruction.png",
            np.clip(reconstructed, 0, 255),
            cmap="gray",
            vmin=0,
            vmax=255,
        )
        print(f"K={k}: final RMS={history[-1]:.6f}")

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original Barbara")
    for ax, k in zip(axes[1:], (2, 8, 32)):
        reconstructed, history = results[k]
        ax.imshow(reconstructed, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"K={k}, RMS={history[-1]:.2f}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(folder / "Project5_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for k in (2, 8, 32):
        history = results[k][1]
        ax.plot(
            np.arange(1, len(history) + 1),
            history,
            marker="o",
            markersize=2.5,
            label=f"K={k}",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMS (pixel intensity)")
    ax.set_title("Competitive learning: epoch vs RMS (d=4)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "Project5_RMS_graph.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved comparison, RMS graph, and three reconstructed images.")


if __name__ == "__main__":
    main()
