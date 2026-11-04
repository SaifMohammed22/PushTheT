from lerobot.datasets import LeRobotDataset, StreamingLeRobotDataset


def download_pusht(cfg, stream=False):
    episodes = list(range(cfg.num_episodes))
    if stream:
        return StreamingLeRobotDataset(cfg.repo, episodes=episodes)
    return LeRobotDataset(cfg.repo, root=cfg.data_dir, episodes=episodes)


def get_dataset(cfg):
    """Return the downloaded PushT dataset."""
    return download_pusht(cfg)
