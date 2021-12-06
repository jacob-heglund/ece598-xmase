import os
import glob
import time
from datetime import datetime
import argparse
from gym.envs.registration import register

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio

import gym
import random

from PPO import PPO
import pdb


def evaluate(has_continuous_action_space=False, max_ep_len=1000, action_std=0.6):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="multigrid-rat-50-v0", choices=["multigrid-rat-0-v0","multigrid-rat-10-v0", "multigrid-rat-50-v0", "multigrid-rat-100-v0"], help="Environment used to evaluate the agent policy")
    parser.add_argument("--policy_env", type=str, default="multigrid-rat-50-v0", choices=["multigrid-rat-0-v0","multigrid-rat-10-v0", "multigrid-rat-50-v0", "multigrid-rat-100-v0"], help="Agent policy loaded for evaluation")
    parser.add_argument("--n_episodes", type=int, default=10, help="Number of evaluation episodes")
    args = parser.parse_args()

    if args.env == "multigrid-rat-0-v0":
        register(
            id="multigrid-rat-0-v0",
            entry_point="gym_multigrid.envs:CollectGameRat_0",
        )

    elif args.env == "multigrid-rat-10-v0":
        register(
            id="multigrid-rat-10-v0",
            entry_point="gym_multigrid.envs:CollectGameRat_10",
        )

    elif args.env == "multigrid-rat-50-v0":
        register(
            id="multigrid-rat-50-v0",
            entry_point="gym_multigrid.envs:CollectGameRat_50",
        )

    elif args.env == "multigrid-rat-100-v0":
        register(
            id="multigrid-rat-100-v0",
            entry_point="gym_multigrid.envs:CollectGameRat_100",
        )

    else:
        raise NotImplementedError
    env = gym.make(args.env)

    n_agents = 1
    obs_size = [3]
    obs_dim = 6

    K_epochs = 80
    eps_clip = 0.2
    gamma = 0.99
    lr_actor = 0.0003
    lr_critic = 0.001

    #  action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ppo_agents = []
    for i in range(n_agents):
        ppo_agents.append(PPO(obs_dim, obs_size[i], action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std))

    # preTrained weights directory
    random_seed = 1
    run_num_pretrained = 0

    print("setting random seed to ", random_seed)
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    directory = "storage" + '/' + args.policy_env + '/'

    for i in range(n_agents):
        checkpoint_path = directory + f"PPO_{args.policy_env}_seed_{random_seed}_run_{run_num_pretrained}_agent_{i}.pth"
        ppo_agents[i].load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")
    test_running_reward = 0
    frame_count = 0
    img_buffer = []
    obs_buffer = []
    reward_buffer = []

    for ep in range(1, args.n_episodes+1):
        obs = env.reset()
        obs_buffer.append(obs)
        ep_reward = 0

        # render first frame of env
        image_render(img_buffer, args, env, ep, t=frame_count, done=False, save=True)

        for t in range(1, max_ep_len+1):
            # select action with policy
            actions = []
            action_probs = []
            for i in range(n_agents):
                action, action_prob = ppo_agents[i].select_action(obs[i])
                actions.append(action)
                action_probs.append(action_prob)

            obs, reward, done, _ = env.step(actions)
            obs_buffer.append(obs)
            ep_reward += reward
            image_render(img_buffer, args, env, ep, frame_count, done, ep_reward, save=True)
            frame_count += 1

            if done:
                # add some frames to make the gif "hang" at the end
                n_end_frames = 10
                for end_frames in range(n_end_frames):
                    frame_count += 1
                    image_render(img_buffer, args, env, ep, frame_count, done, ep_reward, save=True)
                break

        # clear buffer
        for i in range(n_agents):
            ppo_agents[i].buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t Reward: {}'.format(ep, round(ep_reward.item(), 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")
    print("total number of frames / timesteps / images saved : ", frame_count)
    avg_test_reward = test_running_reward / args.n_episodes
    avg_test_reward = round(avg_test_reward.item(), 2)
    print("average test reward : " + str(avg_test_reward))
    print("============================================================================================")

    # save gif to disk
    save_gif(args, img_buffer)


def save_gif(args, img_buffer):

    print("============================================================================================")

    # adjust following parameters to get desired duration, size (bytes) and smoothness of gif
    total_timesteps = 300
    step = 1

    # output path
    gif_dir = "PPO_gifs"
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)

    gif_dir = f"{gif_dir}/{args.env}"
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)

    print("total frames in gif : ", len(img_buffer))

    # save gif
    gif_path = f"{gif_dir}/policy_{args.policy_env}.gif"
    imageio.mimsave(gif_path, img_buffer, duration=0.15)
    print("saved gif at : ", gif_path)
    print("============================================================================================")


def image_render(img_buffer, args, env, ep, t, done, ep_reward=None, save=True):
    img = env.render(mode="rgb_array", tile_size=75)
    img = Image.fromarray(img).transpose(Image.ROTATE_90)
    I1 = ImageDraw.Draw(img)
    font = ImageFont.truetype("Montserrat-Black.otf", 25)
    I1.text((10, 10), f"Goal Position: {env.goal_pos} ({env.signal_color} Signal)", fill=(255, 180, 0), font=font)
    I1.text((10, 40), f"Episode: {ep}", fill=(255, 180, 0), font=font)

    env_str = f"{str(args.env).split('-')[2]}% Left Goal"
    policy_str = f"{str(args.policy_env).split('-')[2]}% Left Goal"
    I1.text((325, 90), f"Environment:\n{env_str}", fill=(255, 180, 0), font=font)
    I1.text((325, 160), f"Agent Policy:\n{policy_str}", fill=(255, 180, 0), font=font)
    if done:
        if ep_reward > 0:
            choice = "Correct"
            choice_color = (0, 255, 0)
        else:
            choice = "Incorrect"
            choice_color = (255, 0, 0)
        I1.text((10, 70), f"{choice}", fill=choice_color, font=font)

    if save:
        img_buffer.append(img)


if __name__ == '__main__':
    evaluate()
