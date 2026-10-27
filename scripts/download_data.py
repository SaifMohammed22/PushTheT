from lerobot.datasets import LeRobotDataset, StreamingLeRobotDataset


def download_pusht(cfg, stream=False):
    episodes = list(range(cfg.num_episodes))
    if stream:
        return StreamingLeRobotDataset(cfg.repo, episodes=episodes)
    return LeRobotDataset(cfg.repo, root=cfg.data_dir, episodes=episodes)



if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("configs/dataset.yaml")
    ds = download_pusht(cfg)
    if isinstance(ds, StreamingLeRobotDataset):
        print(f"repo={cfg.repo} episodes={ds.num_episodes} frames={ds.num_frames} fps={ds.fps}")
    else:
        print(ds)
        print(f"num samples: {len(ds)}")