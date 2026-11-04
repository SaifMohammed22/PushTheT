import torch
from torchvision.transforms import v2
from tqdm.auto import tqdm


def preprocess_image(image, image_size=224, normalize=False, mean=None, std=None):
    image = v2.Resize((image_size, image_size), antialias=True)(image)
    if normalize:
        image = v2.Normalize(mean, std)(image)
    return image


def make_data(dataset, cfg):
    images, actions, states = [], [], []
    episode_bounds = []
    previous_episode, episode_start = None, 0

    for index in tqdm(range(len(dataset)), desc="Extracting and processing data"):
        sample = dataset[index]

        image = preprocess_image(
            sample["observation.image"],
            cfg.image_size,
            cfg.normalize_images,
            cfg.imagenet_mean,
            cfg.imagenet_std,
        )
        images.append(image if cfg.normalize_images else (image * 255).round().to(torch.uint8))
        actions.append(sample["action"])
        states.append(sample["observation.state"])

        episode = sample["episode_index"].item()
        if episode != previous_episode:
            if previous_episode is not None:
                episode_bounds.append((episode_start, index))
            episode_start, previous_episode = index, episode

    episode_bounds.append((episode_start, len(images)))
    return torch.stack(images), torch.stack(actions), torch.stack(states), episode_bounds
