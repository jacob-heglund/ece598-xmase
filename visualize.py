import os
import glob
import time
from datetime import datetime
import argparse
from gym.envs.registration import register

import torch
import numpy as np
from PIL import Image

import gym

# import pybullet_envs

from PPO import PPO
import pdb



"""
One frame corresponding to each timestep is saved in a folder :

PPO_gif_images/env_name/000001.jpg
PPO_gif_images/env_name/000002.jpg
PPO_gif_images/env_name/000003.jpg
...
...
...


if this section is run multiple times or for multiple episodes for the same env_name;
then the saved images will be overwritten.

"""

############################# save images for gif ##############################


def save_gif_images(env_name, has_continuous_action_space, max_ep_len, action_std):

    print("============================================================================================")
    if env_name == "multigrid-single-agent-v0":
        n_agents = 1
        obs_size = [7]

    elif env_name == "multigrid-watcher-v0":
        n_agents = 2
        # agent 0 is the watcher with larger visual range
        # agent 1 is the collector with smaller visual range
        obs_size = [7, 3]

    total_test_episodes = 2     # save gif for this many episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003         # learning rate for actor
    lr_critic = 0.001         # learning rate for critic
    obs_dim = 6

    env = gym.make(env_name)

    #  action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # make directory for saving gif images
    gif_images_dir = "PPO_gif_images" + '/'
    if not os.path.exists(gif_images_dir):
        os.makedirs(gif_images_dir)

    # make environment directory for saving gif images
    gif_images_dir = gif_images_dir + '/' + env_name + '/'
    if not os.path.exists(gif_images_dir):
        os.makedirs(gif_images_dir)

    # make directory for gif
    gif_dir = "PPO_gifs" + '/'
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)

    # make environment directory for gif
    gif_dir = gif_dir + '/' + env_name  + '/'
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)

    ppo_agents = []
    for i in range(n_agents):
        ppo_agents.append(PPO(obs_dim, obs_size[i], action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std))

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    directory = "storage" + '/' + env_name + '/'

    for i in range(n_agents):
        checkpoint_path = directory + f"PPO_{args.env}_seed_{random_seed}_run_{run_num_pretrained}_agent_{i}.pth"
        ppo_agents[i].load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")
    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):

        obs = env.reset()
        ep_reward = 0

        for t in range(1, max_ep_len+1):
            # selection action with policy
            actions = []
            for i in range(n_agents):
                actions.append(ppo_agents[i].select_action(obs[i]))
            obs, reward, done, _ = env.step(actions)
            ep_reward += reward

            img = env.render(mode = 'rgb_array')
            img = Image.fromarray(img)
            img.save(gif_images_dir + '/' + str(t).zfill(6) + '.jpg')

            if done:
                break

        # clear buffer
        for i in range(n_agents):
            ppo_agents[i].buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")
    print("total number of frames / timesteps / images saved : ", t)
    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))
    print("============================================================================================")


def save_gif(env_name):

    print("============================================================================================")

    gif_num = 0     #### change this to prevent overwriting gifs in same env_name folder

    # adjust following parameters to get desired duration, size (bytes) and smoothness of gif
    total_timesteps = 300
    step = 1
    frame_duration = 150

    # input images
    gif_images_dir = "PPO_gif_images/" + env_name + '/*.jpg'

    # ouput gif path
    gif_dir = "PPO_gifs"
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)

    gif_dir = gif_dir + '/' + env_name
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)

    gif_path = gif_dir + '/PPO_' + env_name + str(gif_num) + '.gif'

    img_paths = sorted(glob.glob(gif_images_dir))
    img_paths = img_paths[:total_timesteps]
    img_paths = img_paths[::step]

    print("total frames in gif : ", len(img_paths))
    print("total duration of gif : " + str(round(len(img_paths) * frame_duration / 1000, 2)) + " seconds")

    # save gif
    img, *imgs = [Image.open(f) for f in img_paths]
    img.save(fp=gif_path, format='GIF', append_images=imgs, save_all=True, optimize=True, duration=frame_duration, loop=0)

    print("saved gif at : ", gif_path)
    print("============================================================================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="multigrid-single-agent-v0")
    args = parser.parse_args()

    if args.env == "multigrid-single-agent-v0":
        register(
            id="multigrid-single-agent-v0",
            entry_point="gym_multigrid.envs:CollectGameSingleAgent",
        )

    elif args.env == "multigrid-watcher-v0":
        register(
            id="multigrid-watcher-v0",
            entry_point="gym_multigrid.envs:CollectGameWatcher",
            )

    # save .jpg images in PPO_gif_images folder
    save_gif_images(args.env, has_continuous_action_space=False, max_ep_len=1000, action_std=0.6)

    # save .gif in PPO_gifs folder using .jpg images
    save_gif(args.env)
