import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pushthet.dataset.push_t import get_dataset
from pushthet.dataset.transform import make_data
from pushthet.algorithms.dinov2 import DINOEncoder


def get_features(cfg, encoder):
    dataset = get_dataset(cfg)
    images, _, _, ep_bounds= make_data(dataset, cfg)

    features_list = []

    for i in tqdm(range(0, len(images), cfg.encoder_batch_size)):
        batch = (images[i: i+cfg.encoder_batch_size] / 255.0).to(cfg.device)
        feats = encoder(batch)
        features_list.append(feats.cpu())

    features = torch.cat(features_list, dim=0)
    return images, features, ep_bounds


def plot_features(images, features, ep_bounds):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    ep_start, ep_end = ep_bounds[0]
    frame_indices = np.linspace(ep_start, ep_end - 1, 5, dtype=int)

    sel_feats = [features[fi].numpy() for fi in frame_indices]
    pca = PCA(n_components=3).fit(np.concatenate(sel_feats))

    for i, fi in enumerate(frame_indices):
        axes[0, i].imshow(images[fi].permute(1, 2, 0).numpy())
        axes[0, i].set_title(f'Frame {fi - ep_start}', fontsize=9); axes[0, i].axis('off')

        fp = pca.transform(sel_feats[i]).reshape(16, 16, 3)
        fp = (fp - fp.min()) / (fp.max() - fp.min() + 1e-8)
        axes[1, i].imshow(fp); axes[1, i].set_title('DINO PCA', fontsize=9); axes[1, i].axis('off')

    fig.suptitle('DINO features on Push-T expert demonstrations', fontsize=12)
    plt.show()


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/dataset.yaml"),
        OmegaConf.load("configs/algorithm/dino.yaml"),
        OmegaConf.load("configs/experiment/frozen.yaml"),
    )
    dino_encoder = DINOEncoder(cfg).to(cfg.device)
    images, features, ep_bounds = get_features(cfg, dino_encoder)
    plot_features(images, features, ep_bounds)

