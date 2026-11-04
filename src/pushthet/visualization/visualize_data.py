import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pushthet.dataset.push_t import get_dataset
from pushthet.dataset.transform import preprocess_image


def get_episode_bounds(dataset):
    """Episode start/end frame indices from metadata."""
    episodes = dataset.meta.episodes
    return list(zip(episodes["dataset_from_index"], episodes["dataset_to_index"]))


def plot_images(dataset, cfg, episode_bounds, rows=3, cols=8, episode_step=10):
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    for row in range(rows):
        start, end = episode_bounds[min(row * episode_step, len(episode_bounds) - 1)]
        episode_length = end - start
        for col in range(cols):
            offset = min(col * (episode_length // cols), episode_length - 1)
            frame = dataset[start + offset]["observation.image"]
            image = preprocess_image(frame, cfg.image_size).permute(1, 2, 0).numpy()
            axes[row, col].imshow(image)
            axes[row, col].set_title(f"f={offset}", fontsize=8)
            axes[row, col].axis("off")
    fig.suptitle(
        "Expert Push-T Demonstrations from LeRobot (sampled frames)", fontsize=12
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("configs/dataset.yaml")
    dataset = get_dataset(cfg)
    plot_images(dataset, cfg, get_episode_bounds(dataset))
