import torch
import numpy as np
import torch.nn as nn
import gym
import re
import os
import cv2
from collections import deque
import random
from torch.utils.data import Dataset, DataLoader
import time
from skimage.util.shape import view_as_windows
import torchvision.transforms as transforms

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device,image_size=84,transform=None, \
        multiview = 3, multicam_contrastive=False, frame_stack=3):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform = transform
        self.multicam_contrastive = multicam_contrastive
        self.multiview = multiview
        self.frame_stack = frame_stack
        self.obs_shape = obs_shape
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        
        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.episode_steps = np.empty((capacity, 1), dtype=np.int16)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False


    

    def add(self, obs, action, reward, next_obs, done, episode_step):
       
        np.copyto(self.obses[self.idx], obs[-self.obs_shape[0]:].cpu())
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.episode_steps[self.idx], episode_step)
        np.copyto(self.next_obses[self.idx], next_obs[-self.obs_shape[0]:].cpu())
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
        
    def get_obs_with_idxs(self, idxs, obses='obses'):
        episode_steps = np.squeeze(self.episode_steps[idxs])
        idxs2 = idxs
        mask = np.zeros_like(idxs)
        mask[episode_steps >= 1] = 1
        idxs1 = idxs2 - mask
        mask = np.zeros_like(idxs)
        mask[episode_steps >= 2] = 1
        idxs0 = idxs1 - mask
        if obses == 'obses':
            obses1 = self.obses[idxs1]
            obses2 = self.obses[idxs2]
            return np.concatenate((obses1, obses2), axis=1)
        elif obses == 'next_obses':
            obses1 = self.next_obses[idxs1]
            obses2 = self.next_obses[idxs2]
            return np.concatenate((obses1, obses2), axis=1)
            

    def sample_proprio(self):
        
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )
        obses = self.get_obs_with_idxs(idxs, obses='obses')
        next_obses = self.get_obs_with_idxs(idxs, obses='next_obses')

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones

    def sample_cpc(self):

        start = time.time()
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )
      
        obses = self.get_obs_with_idxs(idxs, obses='obses')
        next_obses = self.get_obs_with_idxs(idxs, obses='next_obses')
        pos = obses.copy()
        
        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        pos = torch.as_tensor(pos, device=self.device).float()
        
        if self.multicam_contrastive:
            obses = sample_view_from_multiview(obses, self.multiview, rl = True, frame_stack=self.frame_stack)
            next_obses = sample_view_from_multiview(next_obses, self.multiview, rl = True, frame_stack=self.frame_stack)
            pos = sample_view_from_multiview(pos, self.multiview, rl = False,  frame_stack = self.frame_stack)

        obses = random_crop(obses, self.image_size)
        next_obses = random_crop(next_obses, self.image_size)
        pos = random_crop(pos, self.image_size)
    
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        
        cpc_kwargs = dict(obs_anchor=obses, obs_pos=pos,
                          time_anchor=None, time_pos=None)

        return obses, actions, rewards, next_obses, not_dones, cpc_kwargs

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

    def __getitem__(self, idx):
        idx = np.random.randint(
            0, self.capacity if self.full else self.idx, size=1
        )
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity 

class FrameStack(gym.Wrapper):
    def __init__(self, env, k, tsne=False):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self.tsne = tsne
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        if self.tsne:
            obs, state_obs = self.env.reset()
            for _ in range(self._k):
                self._frames.append(obs)
            return self._get_obs(), state_obs
        else:
            obs = self.env.reset()
            for _ in range(self._k):
                self._frames.append(obs)
            return self._get_obs()

    def step(self, action):
        if self.tsne:
            obs, reward, done, info, state_obs = self.env.step(action)
            self._frames.append(obs)
            return self._get_obs(), reward, done, info, state_obs
        else:
            obs, reward, done, info = self.env.step(action)
            self._frames.append(obs)
            return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return torch.cat(list(self._frames), axis=0)


def random_crop(imgs, output_size):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones

    args:
        imgs, batch images with shape (B,C,H,W)
    """
    transform = transforms.RandomCrop((output_size, output_size))
    cropped_imgs = transforms.Lambda(lambda x: torch.stack([transform(x_) for x_ in x]))(imgs)
    
    return cropped_imgs

def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, top:top + new_h, left:left + new_w]
    return image


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


def sample_view_from_multiview(image, multiview = 3, rl = True, frame_stack = 3):

    width = image.shape[-1]
    if image.ndim == 3:
        image = image.view(frame_stack, multiview * 2, 3, width, width)
        if rl:
            image =  image[:, :multiview, :, :, :]
            return image.reshape(frame_stack * multiview * 3, width, width)
        else:
            image =  image[:, multiview:, :, :, :]
            return image.reshape(frame_stack * multiview * 3, width, width)
    elif image.ndim == 4:
        image = image.view(-1, frame_stack, multiview * 2, 3, width, width)
        if rl:
            image =  image[:, :, :multiview, :, :, :]
            return image.reshape(-1, frame_stack * multiview * 3, width, width)
        else:
            image =  image[:, :, multiview:, :, :, :]
            return image.reshape(-1, frame_stack * multiview * 3, width, width)
    else:
        raise NotImplementedError