import imageio
import os
import numpy as np
import torch

class VideoRecorder(object):
    def __init__(self, dir_name, height=256, width=256, fps=30, camera_name=None, multicam_contrastive = False):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.fps = fps
        self.frames = []
        self.camera_name = camera_name
        self.multicam_contrastive = multicam_contrastive

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env, obs):
        if self.multicam_contrastive:
            if self.enabled:
                frame = torch.permute(obs, (1,2,0))
                if len(self.camera_name) == 3:
                    frame1 = torch.cat([frame[:, :, -9: -6], frame[:, :, -6: -3], frame[:, :, -3:]], dim=1)
                    frame2 = torch.cat([frame[:, :, 18: 21], frame[:, :, 21: 24], frame[:, :, 24:27]], dim=1)
                    frame = torch.cat([frame1, frame2], dim=0)
                    self.frames.append(frame.cpu().detach().numpy())
                else:
                    self.frames.append(torch.cat([frame[:, :, -3:], frame[:, :, -3:]], dim=1).cpu().detach().numpy())
        else:
            if self.enabled:
                frame = torch.permute(obs, (1,2,0))
                if len(self.camera_name) == 3:
                    frame = torch.cat([frame[:, :, -9: -6], frame[:, :, -6: -3], frame[:, :, -3:]], dim=1)
                    self.frames.append(frame.cpu().detach().numpy())
                else:
                    self.frames.append(frame[:, :, -3:].cpu().detach().numpy())

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)
