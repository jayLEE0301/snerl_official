import gym
import metaworld
import numpy as np
import random
import torch

LONGITUDE = 20
LATITUDE = 5

class EnvWrapper(object):
    def __init__(self, env_name, from_pixels, height, width, frame_skip, camera_name, multicam_contrastive, sparse_reward, device, tsne = False):
        assert from_pixels
        ml1 = metaworld.ML1(env_name) # Construct the benchmark, sampling tasks
        self.env = ml1.train_classes[env_name]()  # Create an environment with task `pick_place`
        self.env.set_task(ml1.train_tasks[0])
        self.device = device
        self.tsne = tsne

        if env_name == 'window-open-v2':
            self.env.random_init = False
            # set camera pos
            for i in range(LONGITUDE):
                for j in range(LATITUDE):
                    cam_name_repose="cam_%d_%d" % (i, j)
                    body_ids = self.env.model.camera_name2id(cam_name_repose)
                    self.env.model.cam_pos[body_ids] = self.env.obj_init_pos + self.env.model.cam_pos[body_ids]
            for i in range(LONGITUDE*10):
                for j in range(LATITUDE*10):
                    cam_name_repose="cam2_%d_%d" % (i, j)
                    body_ids = self.env.model.camera_name2id(cam_name_repose)
                    self.env.model.cam_pos[body_ids] = self.env.obj_init_pos + self.env.model.cam_pos[body_ids]
        else:
            self.env.random_init = True
            self.default_campos = np.zeros((3, 3))
            body_ids = self.env.model.camera_name2id("cam_1_1")
            self.default_campos[0] = self.env.model.cam_pos[body_ids]
            body_ids = self.env.model.camera_name2id("cam_7_4")
            self.default_campos[1] = self.env.model.cam_pos[body_ids]
            body_ids = self.env.model.camera_name2id("cam_14_2")
            self.default_campos[2] = self.env.model.cam_pos[body_ids]
        
                
        self.env_name = env_name
        self.from_pixels = from_pixels
        self.height = height
        self.width = width
        self.sparse_reward = sparse_reward
        self._max_episode_steps = self.env.max_path_length
        self.multicam_contrastive = multicam_contrastive
        
        self.camera_name = camera_name
        self.multicam_contrastive = multicam_contrastive


        
    def background_mask(self, single_image, single_depth):
        
        single_image = torch.from_numpy(single_image).to(self.device)
        single_depth = torch.from_numpy(single_depth).to(self.device)
        mask = torch.zeros(single_depth.shape).to(self.device)
        mask[(single_depth) < 0.999] = 1
        mask = torch.unsqueeze(mask, -1)
        single_image = single_image * mask +  255 * (1 - mask)
        
        single_image = torch.permute(single_image, (2,0,1)).type(torch.uint8)
        return single_image
        
        
    def reset(self, *args, **kwargs):
        state_obs = self.env.reset()

        multicam_image = []
        if self.env_name == 'window-open-v2':
            pass
        else:
            body_ids = self.env.model.camera_name2id("cam_1_1")
            self.env.model.cam_pos[body_ids] = self.env._target_pos[:3] + self.default_campos[0]
            body_ids = self.env.model.camera_name2id("cam_7_4")
            self.env.model.cam_pos[body_ids] = self.env._target_pos[:3] + self.default_campos[1]
            body_ids = self.env.model.camera_name2id("cam_14_2")
            self.env.model.cam_pos[body_ids] = self.env._target_pos[:3] + self.default_campos[2]
            
        
        
        camera_name_aug = self.camera_name.copy()
        if self.multicam_contrastive:
            for single_cam in self.camera_name:
                a, b, c = single_cam.split('_')
                while True:
                    perturb_phi = random.randint(-10, 10)
                    perturb_psi = random.randint(-4, 4)
                    if perturb_phi != 0 or perturb_psi != 0:
                        break
                camera_name_aug.append("cam2_%d_%d" % (int(b)+perturb_phi, int(c)+perturb_psi))

            
        for single_cam in camera_name_aug:
            single_image, single_depth = self.sim.render(width = self.width, height = self.height, camera_name = single_cam, depth=True)
            
            single_image = self.background_mask(single_image, single_depth)
            
            multicam_image.append(single_image)
        if self.tsne:
            return torch.cat(multicam_image, dim=0), state_obs
        else:
            return torch.cat(multicam_image, dim=0)
        
    
    def step(self, action):
        state_obs, reward, done, info = self.env.step(action)

        multicam_image = []
        
        
        camera_name_aug = self.camera_name.copy()
        if self.multicam_contrastive:
            for single_cam in self.camera_name:
                a, b, c = single_cam.split('_')
                while True:
                    perturb_phi = random.randint(-10, 10)
                    perturb_psi = random.randint(-4, 4)
                    if perturb_phi != 0 or perturb_psi != 0:
                        break
                camera_name_aug.append("cam2_%d_%d" % (int(b)+perturb_phi, int(c)+perturb_psi))

        for single_cam in camera_name_aug:
            single_image, single_depth = self.sim.render(width = self.width, height = self.height, camera_name = single_cam, depth=True)
            
            single_image = self.background_mask(single_image, single_depth)
            
            multicam_image.append(single_image)
            
        if self.sparse_reward:
            reward = info['success'] - 1.0
            
        
        if self.env.curr_path_length == self._max_episode_steps:
            done = True
        if self.tsne:
            return torch.cat(multicam_image, dim=0), reward, done, info, state_obs
        else:
            return torch.cat(multicam_image, dim=0), reward, done, info
    
    
    def __getattr__(self, attrname):
        return getattr(self.env, attrname)