"""
Download PushT dataset locally into data directory using LeRobotDataset or stream it without
the need to download it locally using StreamingLeRobotDataset
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pushthet.dataset.push_t import (  # noqa: E402
    StreamingLeRobotDataset,
    download_pusht,
)


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("configs/dataset.yaml")
    ds = download_pusht(cfg)
    if isinstance(ds, StreamingLeRobotDataset):
        print(
            f"repo={cfg.repo} episodes={ds.num_episodes} frames={ds.num_frames} fps={ds.fps}")
    else:
        print(ds)
        print(f"num samples: {len(ds)}")
