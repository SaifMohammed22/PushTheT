"""
Test the downloaded data
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pushthet.dataset.push_t import get_dataset


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("configs/dataset.yaml")
    ds = get_dataset(cfg)
    print(ds)
    print(f"num samples: {len(ds)}")
