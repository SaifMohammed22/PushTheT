# PushTheT
A World Model/Policy for PushT environment

## Dataset 
I use the [PushT](https://huggingface.co/datasets/lerobot/pusht) dataset with 206 episodes and 25,650 frames. The dataset features are: \
```['observation.image', 'observation.state', 'action', 'episode_index', 'frame_index', 'timestamp', 'next.reward', 'next.done', 'next.success', 'index', 'task_index']```

## Dataset preprocessing
Before dealing with the data, we need to resize and normalize the images/frames.

I resize the frame from (96, 96, 3) to (224, 224, 3) to match the model used. If `cfg.normalize` is true, I normalize using ImageNet mean and std; otherwise, I use unit8 data format to save memory. After that, I stack all the images, actions, states, and episode_bounds and return them.

## Acknowledgement
Following [This notebook](https://colab.research.google.com/github/Vizuara-AI-Lab/Modern-Robot-Learning-V2-Bootcamp/blob/main/lecture-world-models/public/notebooks/DINO_WM_PushT.ipynb#scrollTo=-db0K6_wIKDz) with modifications.
Some parts of this project are adapted from or inspired by [Boyuan Chen](https://github.com/buoyancy99/research-template.git) ML research template under [MIT License](https://github.com/buoyancy99/research-template/blob/main/LICENSE).
```
@article{chen2025diffusion,
  title={Diffusion forcing: Next-token prediction meets full-sequence diffusion},
  author={Chen, Boyuan and Mart{\'\i} Mons{\'o}, Diego and Du, Yilun and Simchowitz, Max and Tedrake, Russ and Sitzmann, Vincent},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={24081--24125},
  year={2025}
}
