import torch
import torch.nn as nn
from omegaconf import OmegaConf


class DINOEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dinov2 = torch.hub.load(cfg.repo, cfg.model)
        if cfg.freeze:
            self.dinov2.eval()
        else:
            self.dinov2.train()
        for p in self.dinov2.parameters():
            p.requires_grad = not cfg.freeze
        self.embed_dim = cfg.embed_dim
        self.num_patches = cfg.num_patches

    def forward(self, x):
        return self.dinov2.forward_features(x)["x_norm_patchtokens"]


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/algorithm/dino.yaml")
    dino_encoder = DINOEncoder(cfg)
    state = "frozen" if cfg.freeze else "trainable"
    print(f"DINO Encoder: {sum(p.numel() for p in dino_encoder.parameters())/1e6:.1f}M params ({state})")
