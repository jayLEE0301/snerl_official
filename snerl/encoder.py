import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.functional as F


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w





def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias

# for 100 x 100 inputs
OUT_DIM_100 = {2: 47, 4: 43, 6: 39}
# for 84 x 84 inputs
OUT_DIM_84 = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers=18, pretrained=True, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        self.encoder = models.resnet18(True)

        self.encoder.fc = nn.Linear(512, 256)

    def forward(self, obs, outputs = None):
        x = self.encoder.conv1(obs)
        if outputs is not None:
            outputs['conv1'] = x[0]
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.layer1(self.encoder.maxpool(x))
        if outputs is not None:
            outputs['conv2'] = x[0]
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        
        x = self.encoder.avgpool(x)
        x = self.encoder.fc(x.flatten(1))

        return x
    


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32,output_logits=False, log_encoder=False, multiview = 3, frame_stack = 3):
        super().__init__()

        assert len(obs_shape) == 3
        self.width = obs_shape[-1]
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.frame_stack = frame_stack
        self.multiview = multiview

        self.convs = nn.ModuleList(
            [nn.Conv2d(3 * self.multiview, num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        if obs_shape[-1] == 64:
            out_dim = OUT_DIM_64[num_layers]
        elif obs_shape[-1] == 84:
            out_dim = OUT_DIM_64[num_layers]
        elif obs_shape[-1] == 100:
            out_dim = OUT_DIM_100[num_layers]
        else:
            raise NotImplementedError
        self.fc = nn.Linear(num_filters * out_dim * out_dim, int(self.feature_dim / self.frame_stack))
        self.ln = nn.LayerNorm(int(self.feature_dim / self.frame_stack))

        if log_encoder:
            self.outputs = dict()
        else:
            self.outputs = None
            
        self.mlp1 = nn.Linear((int(self.feature_dim / self.frame_stack)), 63)
        self.mlp2 = nn.Linear(63, 63)
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        # obs = obs / 255.
        if self.outputs is not None:
            self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        if self.outputs is not None:
            self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            if self.outputs is not None:
                self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        batch = len(obs)
        obs = obs/255. # (batch_size, frame_stack, multi_view, channel, height, width)
        obs = obs.view(batch * self.frame_stack, self.multiview * 3, self.width, self.width)
        
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h = self.fc(h)
        h = h.view(batch * self.frame_stack, -1)
        if self.outputs is not None:
            self.outputs['fc'] = h
        
        h = self.mlp1(h)
        h_fc = self.mlp2(h)

        h_norm = self.ln(h_fc)
        h_norm = h_norm.view(batch, -1)
        if self.outputs is not None:
            self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            if self.outputs is not None:
                self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters,*args):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


OUT_DIM_128 = {2: 29, 4: 57, 6: 21}


class MeanEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, multiview, frame_stack, encoder_name, finetune_encoder = False, log_encoder = False, output_logits=False, env_name=None):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        assert feature_dim / frame_stack == 63
        self.feature_dim = feature_dim
        self.num_layers = 4
        self.num_filters = 32
        self.output_logits = output_logits
        self.multiview = multiview
        self.frame_stack = frame_stack
        self.encoder_name = encoder_name
        self.finetune_encoder = finetune_encoder
        self.log_encoder = log_encoder
        self.width = obs_shape[-1]
        self.env_name = env_name
        if self.finetune_encoder or self.log_encoder:
            raise NotImplementedError


        if obs_shape[-1] == 64:
            out_dim = OUT_DIM_64[self.num_layers]     
        elif obs_shape[-1] == 128:
            out_dim = OUT_DIM_128[self.num_layers] 
        else: 
            out_dim = OUT_DIM_100[self.num_layers]


        self.convs = nn.ModuleList(
            [nn.Conv2d(3, self.num_filters, 3, stride=2)]
        )
        for i in range(self.num_layers - 1):
            self.convs.append(nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1))

        self.fc = nn.Linear(self.num_filters * out_dim * out_dim, int(self.feature_dim / self.frame_stack))


        # self.fc = nn.Linear(num_filters * out_dim * out_dim * 3, self.feature_dim)

        self.ln = nn.LayerNorm(int(self.feature_dim / self.frame_stack))

        if log_encoder:
            self.outputs = dict()
        else:
            self.outputs = None

        self.mlp1 = nn.Linear((int(self.feature_dim / self.frame_stack)+16), 63)
        self.mlp2 = nn.Linear(63, 63)

        print("use encoder:", encoder_name, "finetune_encoder:", finetune_encoder, "log_encoder:", log_encoder)

        if self.encoder_name == 'snerl':
            self.load_state_dict(torch.load('./encoder_pretrained/{}/snerl.tar'.format(self.env_name), map_location='cpu'))
            print(self.env_name, 'use snerl encoder final')
        else:
            raise NotImplementedError
        for param in self.parameters():
            param.requires_grad = False
            
        for param in self.fc.parameters():
            param.requires_grad = True

        for param in self.mlp1.parameters():
            param.requires_grad = True

        for param in self.mlp2.parameters():
            param.requires_grad = True

        for param in self.ln.parameters():
            param.requires_grad =True

        self.pose1 = pose_spherical(18-180, -(90-27), 4)
        self.pose2 = pose_spherical(126-180, -(90-81), 4)
        self.pose3 = pose_spherical(252-180, -(90-45), 4)

        self.hard_poses = torch.stack([self.pose1, self.pose2, self.pose3], dim=0).view(3, 16)
        self.hard_poses = self.hard_poses.unsqueeze(dim=0)


    def forward_conv(self, obs):
        #obs = obs / 255.
        if self.outputs is not None:
            self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        if self.outputs is not None:
            self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            if self.outputs is not None:
                self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)

        return h


    def forward(self, obs, obs_pose=None, detach=False):
        
        if obs_pose == None:
            obs_pose = self.hard_poses.cuda()
        batch = len(obs)

        obs = obs/255. # (batch_size, frame_stack, multi_view, channel, height, width)
        
        with torch.no_grad():

            obs = obs.view(batch * self.frame_stack * self.multiview, 3, self.width, self.width)

            obs_pose = obs_pose.expand((batch * self.frame_stack, self.multiview, 16)).cuda()

            h = self.forward_conv(obs)
        
        if detach:
            h = h.detach()
            
            
        h = self.fc(h)
        
        h = h.view(batch * self.frame_stack, self.multiview, -1)

        h = torch.cat([h, obs_pose], dim=2)
        h = self.mlp1(h)
        h = torch.mean(h, dim=1, keepdim=False)
        h_fc = self.mlp2(h)


        h_norm = self.ln(h_fc)

        h_norm = h_norm.view(batch, -1)



        # print(h_norm1)
        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            if self.outputs is not None:
                self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)



_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder, 'nerf' : MeanEncoder}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, output_logits=False, multiview = 3, frame_stack = 3, \
        encoder_name = None, finetune_encoder = False, log_encoder = False, env_name = None
):
    assert encoder_type in _AVAILABLE_ENCODERS
    if encoder_type == 'nerf':
        return MeanEncoder(obs_shape, feature_dim, multiview, frame_stack, encoder_name, finetune_encoder, log_encoder, output_logits, env_name)
    else:
        return _AVAILABLE_ENCODERS[encoder_type](
            obs_shape, feature_dim, num_layers, num_filters, output_logits, log_encoder, multiview, frame_stack
    )