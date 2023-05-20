# SNeRL

## Setup Instructions
0. Create a conda environment:
```
conda create -n snerl python=3.9
conda activate snerl
```

1. Install [MuJoCo](https://github.com/deepmind/mujoco) and task environments:
```
cd metaworld
pip install -e .
cd ..
```

2. install [pytorch](https://pytorch.org/get-started/locally/) (use tested on pytorch 1.12.1 with CUDA 11.3)



3. install additional dependencies:
```
pip install scikit-image
pip install tensorboard
pip install termcolor
pip install imageio
pip install imageio-ffmpeg
pip install opencv-python
pip install matplotlib
pip isntall tqdm
pip install timm
pip install configargparse
```





## Usage
Our code does not include the dataset generator for nerf pretraining. Please prepare your dataset for nerf pretraining.

### Pretrain Encoder
```
cd nerf_pretrain
python run_nerf.py --config configs/{env_name}.txt
```

### Train Donstream RL
0. Locate pretained model in './encoder_pretrained/{env_name}/snerl.tar'


1. Use the following commands to train RL agents:


window-open-v2
```
CUDA_VISIBLE_DEVICES=0 python snerl/train.py --env_name window-open-v2 --encoder_type nerf --save_tb --frame_stack 2 --eval_freq 10000 --batch_size 128 --save_video --save_model --image_size 128 --camera_name cam_1_1 cam_7_4 cam_14_2 --multiview 3 --encoder_name 'snerl' --seed 1
```

drawer-open-v2
```
CUDA_VISIBLE_DEVICES=0 python snerl/train.py --env_name drawer-open-v2 --encoder_type nerf --save_tb --frame_stack 2 --eval_freq 10000 --batch_size 128 --save_video --save_model --image_size 128 --camera_name cam_1_1 cam_7_4 cam_14_2 --multiview 3 --encoder_name 'snerl' --seed 1
```

hammer-v2
```
CUDA_VISIBLE_DEVICES=0 python snerl/train.py --env_name hammer-v2 --encoder_type nerf --save_tb --frame_stack 2 --eval_freq 10000 --batch_size 128 --save_video --save_model --image_size 128 --camera_name cam_1_1 cam_7_4 cam_14_2 --multiview 3 --encoder_name 'snerl' --seed 1
```

soccer-v2
```
CUDA_VISIBLE_DEVICES=0 python snerl/train.py --env_name soccer-v2 --encoder_type nerf --save_tb --frame_stack 2 --eval_freq 10000 --batch_size 128 --save_video --save_model --image_size 128 --camera_name cam_1_1 cam_7_4 cam_14_2 --multiview 3 --encoder_name 'snerl' --seed 1
```

Our code sourced and modified from official implementation of [CURL](https://github.com/MishaLaskin/curl).




## Citation
If you use this repo in your research, please consider citing the paper as follows.
```
@inproceedings{shim2023snerl,
  title={SNeRL: Semantic-aware Neural Radiance Fields for Reinforcement Learning},
  author={Shim, Dongseok and Lee, Seungjae and Kim, H Jin},
  booktitle=International Conference on Machine Learning},
  pages={},
  year={2023},
  organization={PMLR}
}
```
