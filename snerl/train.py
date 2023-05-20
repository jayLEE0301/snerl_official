import numpy as np
import torch
import argparse
import os
import math
import gym
import sys
import random
import time
import json
import copy
import datetime

import utils
from logger import Logger
from video import VideoRecorder

from curl_sac import CurlSacAgent
from torchvision import transforms
from env_wrapper import EnvWrapper



def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--env_name', default='window-open-v2')
    parser.add_argument('--pre_transform_image_size', default=128, type=int)
    parser.add_argument('--camera_name', nargs='*', default=None)
    parser.add_argument('--multicam_contrastive', default=False, action='store_true')
    parser.add_argument('--multiview', default=3, type=int)
    parser.add_argument('--sparse', default=False, action='store_true')

    parser.add_argument('--image_size', default=128, type=int)
    parser.add_argument('--action_repeat', default=5, type=int)
    parser.add_argument('--frame_stack', default=2, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    parser.add_argument('--agent', default='curl_sac', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=1000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    parser.add_argument('--stddev_schedule', default='linear(1.0,0.1,500000)', type=str)
    parser.add_argument('--stddev_clip', default=0.3, type=float)
    # encoder
    parser.add_argument('--encoder_type', default='nerf', type=str)
    parser.add_argument('--encoder_feature_dim', default=126, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--curl_latent_dim', default=128, type=int)
    parser.add_argument('--log_encoder', default=False, action='store_true')
    parser.add_argument('--finetune_encoder', default=False, action='store_true')
    parser.add_argument('--encoder_name', default=None, type=str)
    parser.add_argument('--no_cpc', default=False, action='store_true')
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='./exp_local', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--detach_encoder', default=False, action='store_true')
    parser.add_argument('--suffix', default=None, type=str)

    parser.add_argument('--log_interval', default=100, type=int)
    args = parser.parse_args()
    return args


def evaluate(env, agent, video, num_episodes, L, step, args):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        total_success = 0
        for i in range(num_episodes):
            obs = env.reset()
            video.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            while not done:
                # center crop image
                if args.encoder_type == 'pixel':
                    if args.multicam_contrastive:
                        obs = utils.sample_view_from_multiview(obs, args.multiview, rl = True, frame_stack= args.frame_stack)
                    obs = utils.center_crop_image(obs,args.image_size)
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                obs, reward, done, info = env.step(action)
                video.record(env, obs)
                episode_reward += reward
            total_success += info['success']
            video.save('%d.mp4' % step)
            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)
        
        L.log('eval/' + prefix + 'eval_time', time.time()-start_time , step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        L.log('eval/' + prefix + 'success_rate', (total_success / num_episodes), step)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'curl_sac':
        return CurlSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            curl_latent_dim=args.curl_latent_dim,
            multicam_contrastive = args.multicam_contrastive,
            multiview=args.multiview,
            frame_stack=args.frame_stack,
            log_encoder=args.log_encoder,
            finetune_encoder=args.finetune_encoder,
            encoder_name=args.encoder_name,
            no_cpc=args.no_cpc,
            env_name=args.env_name,
        )
    else:
        assert 'agent is not supported: %s' % args.agent

def main():
    args = parse_args()
    if args.env_name == 'drawer-close-v2' or args.env_name == 'hammer-v2':
        args.stddev_schedule = 'linear(1.0,0.1,100000)'
        args.stddev_clip = 0.1
    elif args.env_name == 'soccer-v2' or args.env_name == 'window-open-v2':
        args.stddev_schedule = 'linear(1.0,0.1,500000)'
        args.stddev_clip = 0.3
    
    
    if args.seed == -1: 
        args.__dict__["seed"] = np.random.randint(1,1000000)
    assert args.multiview == len(args.camera_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    utils.set_seed_everywhere(args.seed)
    env = EnvWrapper(
        env_name=args.env_name,
        from_pixels=(args.encoder_type == 'pixel' or args.encoder_type == 'nerf'),
        height=args.pre_transform_image_size,
        width=args.pre_transform_image_size,
        frame_skip=args.action_repeat,
        camera_name = args.camera_name,
        multicam_contrastive = args.multicam_contrastive,
        sparse_reward = args.sparse,
        device= device,
    )
 
    env.seed(args.seed)

    # stack several consecutive frames together
    if (args.encoder_type == 'pixel' or args.encoder_type == 'nerf'):
        env = utils.FrameStack(env, k=args.frame_stack)
    
    # make directory
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
    env_name = args.env_name
    exp_name = env_name + '-' + ts + '-im' + str(args.image_size) +'-b'  \
    + str(args.batch_size) + '-s' + str(args.seed)  + '-' + args.encoder_type \
    + '-multicam_cont-' + str(args.multicam_contrastive) + '-framestack-' + str(args.frame_stack) \
    + '-multiview-' + str(args.multiview) + '-sparseReward-' + str(args.sparse) + '-' + str(args.encoder_name) + '-' + str(args.suffix)
    args.work_dir = args.work_dir + '/'  + exp_name

    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(dir_name = video_dir if args.save_video else None,\
        camera_name = args.camera_name, multicam_contrastive = args.multicam_contrastive)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    

    action_shape = env.action_space.shape

    if args.encoder_type == 'pixel':
        obs_shape = (args.multiview * 3 * args.frame_stack, args.image_size, args.image_size)
        
        if args.multicam_contrastive:
            pre_aug_obs_shape = (2 * args.multiview * 3*args.frame_stack,args.pre_transform_image_size,args.pre_transform_image_size)
            buffer_obs_shape = (2 * args.multiview * 3,args.pre_transform_image_size,args.pre_transform_image_size)
        else:
            pre_aug_obs_shape = (args.multiview * 3*args.frame_stack,args.pre_transform_image_size,args.pre_transform_image_size)
            buffer_obs_shape = (args.multiview * 3,args.pre_transform_image_size,args.pre_transform_image_size)
            
    elif args.encoder_type == 'nerf':
        assert args.pre_transform_image_size == args.image_size
        obs_shape = (args.multiview * 3 * args.frame_stack, args.pre_transform_image_size,args.pre_transform_image_size)
        pre_aug_obs_shape = (args.multiview * 3 * args.frame_stack,args.image_size,args.image_size)
        buffer_obs_shape = (args.multiview * 3,args.image_size,args.image_size)
        
    else:
        raise NotImplementedError

    replay_buffer = utils.ReplayBuffer(
        obs_shape=buffer_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
        multiview=args.multiview,
        multicam_contrastive=args.multicam_contrastive,
        frame_stack=args.frame_stack
    )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    for step in range(args.num_train_steps):
        # evaluate agent periodically

        if step % args.eval_freq == 0:
            L.log('eval/episode', episode, step)
            evaluate(env, agent, video, args.num_eval_episodes, L, step,args)
            if args.save_model and args.encoder_type == 'pixel':
                agent.save_curl(model_dir, step)
            if args.save_buffer:
                replay_buffer.save(buffer_dir)

        if done:
            if step > 0:
                if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    L.log('train/success_rate', info['success'], step)
                    L.dump(step)
                start_time = time.time()
            if step % args.log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)
                if args.sparse:
                    stddev = utils.schedule(args.stddev_schedule, step)
                    action += np.clip(np.random.normal(loc = 0.0, scale = stddev, size=action.shape), \
                        -args.stddev_clip, args.stddev_clip)
                

        # run training update
        if step >= args.init_steps:
            num_updates = 1 
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        next_obs, reward, done, info = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool, episode_step)

        obs = next_obs
        episode_step += 1


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    main()
