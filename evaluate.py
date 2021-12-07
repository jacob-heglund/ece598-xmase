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


def visualize_agent(args, env):
    #  action space dimension
    if args.has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ppo_agents = []
    for i in range(args.n_agents):
        ppo_agents.append(PPO(args.obs_dim, args.obs_size, action_dim, args.lr_actor, args.lr_critic, args.gamma, args.K_epochs, args.eps_clip, args.has_continuous_action_space, args.action_std))

    directory = "storage" + '/' + args.policy_env + '/'

    for i in range(args.n_agents):
        checkpoint_path = directory + f"PPO_{args.policy_env}_seed_{args.random_seed}_run_{args.run_num_pretrained}_agent_{i}.pth"
        ppo_agents[i].load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")
    test_running_reward = 0
    frame_count = 0
    img_buffer = []
    obs_buffer = []
    reward_buffer = []

    for ep_idx in range(1, args.n_episodes+1):
        obs = env.reset()
        obs_buffer.append(obs)
        ep_reward = 0

        # render first frame of env
        image_render(img_buffer, args, env, ep_idx, t=frame_count, done=False, save=True)

        for t in range(1, args.max_ep_len+1):
            # select action with policy
            actions = []
            action_probs = []
            for i in range(args.n_agents):
                action, action_prob = ppo_agents[i].select_action(obs[i])
                actions.append(action)
                action_probs.append(action_prob)

            obs, reward, done, _ = env.step(actions)
            obs_buffer.append(obs)
            ep_reward += reward
            image_render(img_buffer, args, env, ep_idx, frame_count, done, ep_reward, save=True)
            frame_count += 1

            if done:
                # add some frames to make the gif "hang" at the end
                n_end_frames = 10
                for end_frames in range(n_end_frames):
                    frame_count += 1
                    image_render(img_buffer, args, env, ep_idx, frame_count, done, ep_reward, save=True)
                break

        # clear buffer
        for i in range(args.n_agents):
            ppo_agents[i].buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t Reward: {}'.format(ep_idx, round(ep_reward.item(), 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")
    avg_test_reward = test_running_reward / args.n_episodes
    avg_test_reward = round(avg_test_reward.item(), 2)
    print("average test reward : " + str(avg_test_reward))
    print("============================================================================================")

    # save gif to disk
    save_gif(args, img_buffer)


def save_gif(args, img_buffer):
    # output path
    if not os.path.exists(args.gif_dir):
        os.makedirs(args.gif_dir)
    save_dir = f"{args.gif_dir}/env_{args.env}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    gif_path = f"{save_dir}/policy_{args.policy_env}.gif"
    imageio.mimsave(gif_path, img_buffer, duration=0.15)

    print("============================================================================================")
    print("total frames in gif : ", len(img_buffer))
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="multigrid-rat-50-v0", choices=["multigrid-rat-0-v0","multigrid-rat-10-v0", "multigrid-rat-50-v0", "multigrid-rat-100-v0"], help="Environment used to evaluate the agent policy")
    parser.add_argument("--policy_env", type=str, default="multigrid-rat-50-v0", choices=["multigrid-rat-0-v0","multigrid-rat-10-v0", "multigrid-rat-50-v0", "multigrid-rat-100-v0"], help="Agent policy loaded for evaluation")
    parser.add_argument("--n_episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--n_agents", type=int, default=1, help="Number of agents")
    parser.add_argument("--obs_size", type=int, default=3, help="Agent view width and height")
    parser.add_argument("--obs_dim", type=int, default=6, help="Agent view depth")
    parser.add_argument("--K_epochs", type=int, default=80, help="Number of epochs between PPO update")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="Clipping for PPO update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discounting rate")
    parser.add_argument("--lr_actor", type=float, default=0.0003, help="Actor learning rate")
    parser.add_argument("--lr_critic", type=float, default=0.001, help="Critic learning rate")
    parser.add_argument("--has_continuous_action_space", type=bool, default=False, help="Continuous vs discrete action space")
    parser.add_argument("--max_ep_len", type=int, default=1000, help="Max number of time steps for an episode")
    parser.add_argument("--action_std", type=float, default=0.6, help="Starting std. dev. for action distribution (Multivariate Normal)")
    parser.add_argument("--random_seed", type=int, default=1, help="Random seed")
    parser.add_argument("--run_num_pretrained", type=int, default=0, help="Load a policy from a particular run number")
    parser.add_argument("--gif_dir", type=str, default="PPO_agent", help="Save directory for output gifs")

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

    print("setting random seed to ", args.random_seed)
    torch.manual_seed(args.random_seed)
    env.seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    visualize_agent(args, env)

