from inspect import trace
import os
import glob
import time
from datetime import datetime
import argparse
import gym
from gym.envs.registration import register

import torch
import numpy as np
import random
import shap
from PIL import Image, ImageDraw, ImageFont
import imageio
import matplotlib.pyplot as plt

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

    for ep_idx in range(1, args.n_episodes+ 1):
        obs = env.reset()
        ep_reward = 0

        # render first frame of env
        render_image(img_buffer, args, env, ep_idx, t=frame_count, done=False, save=True)

        for t in range(1, args.max_ep_len+ 1):
            # select action with policy
            actions = []
            for i in range(args.n_agents):
                action = ppo_agents[i].select_action(obs[i])
                actions.append(action)

            obs, reward, done, _ = env.step(actions)
            ep_reward += reward
            render_image(img_buffer, args, env, ep_idx, frame_count, done, ep_reward, save=True)
            frame_count += 1

            if done:
                # add some frames to make the gif "hang" at the end
                for end_frames in range(args.gif_n_end_frames):
                    frame_count += 1
                    render_image(img_buffer, args, env, ep_idx, frame_count, done, ep_reward, save=True)
                break

        # clear agent buffer
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


def feature_importance_SHAP(args, env):
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

    # create a "background" dataset of agent observations
    test_running_reward = 0
    frame_count = 0
    img_buffer = []
    obs = env.reset()
    obs_buffer = []

    for ep_idx in range(1, args.n_episodes_background + 1):
        obs = env.reset()
        obs_buffer.append(np.expand_dims(obs[0], 0))
        ep_reward = 0

        for t in range(1, args.max_ep_len + 1):
            # select action with policy
            actions = []
            for i in range(args.n_agents):
                action = ppo_agents[i].select_action(obs[i])
                actions.append(action)

            obs, reward, done, _ = env.step(actions)
            obs_buffer.append(np.expand_dims(obs[0], 0))
            ep_reward += reward
            frame_count += 1

            if done:
                break

        # clear agent buffer
        for i in range(args.n_agents):
            ppo_agents[i].buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t Reward: {}'.format(ep_idx, round(ep_reward.item(), 2)))
        ep_reward = 0

    # 'background' dataset of images, X.shape = (n_images, img_size, img_size, img_depth)
    X = np.concatenate(obs_buffer, axis=0)
    print("============================================================================================")
    avg_test_reward = test_running_reward / args.n_episodes
    avg_test_reward = round(avg_test_reward.item(), 2)
    print("average reward in background episodes : " + str(avg_test_reward))
    print("============================================================================================")
    env.close()

    # collect frames from test episodes for analysis with SHAP
    # function to get model output from observations
    def f(obs):
        tmp = obs.copy()
        # select only the first agent's observation since we only have 1 agent anyways
        return ppo_agents[0].select_action(tmp, return_action_prob_shap=True, debug=True)
    masker = shap.maskers.Image("blur(2, 2)", X[0].shape)
    explainer = shap.Explainer(f, masker, output_names = env.actions.available, algorithm="partition")
    # processing done on outputs of explanation
    outputs = shap.Explanation.argsort.flip[:]

    img_buffer = []
    frame_count = 0
    test_episode_envs = ["multigrid-rat-100-v0", "multigrid-rat-0-v0"]
    for ep_idx, test_env in enumerate(test_episode_envs):
        obs_buffer = []
        env = gym.make(test_env)
        obs = env.reset()
        obs_buffer.append(np.expand_dims(obs[0], 0))
        render_image(img_buffer, args, env, ep_idx, t=frame_count, done=False, save=True)

        for t in range(1, args.max_ep_len + 1):
            # select action with policy
            actions = []
            for i in range(args.n_agents):
                action = ppo_agents[i].select_action(obs[i])
                actions.append(action)
            obs, reward, done, _ = env.step(actions)
            obs_buffer.append(np.expand_dims(obs[0], 0))
            render_image(img_buffer, args, env, ep_idx, frame_count, done, reward, save=True)
            frame_count += 1
            if done:
                # add some frames to make the gif "hang" at the end
                for end_frames in range(args.gif_n_end_frames):
                    frame_count += 1
                    render_image(img_buffer, args, env, ep_idx, frame_count, done, reward, save=True)
                print(f"test episode reward: {reward}")

                # feature importance analysis for frames in this episode
                input_images = np.concatenate(obs_buffer[:-1], axis=0)
                shap_values = explainer(input_images, max_evals = 5000, batch_size = 50, outputs=outputs)

                # try to replace items in pixel_values with an RGB-encoded version of them
                ## you can get this from the env, World class probably

                # pdb.set_trace()

                shap.image_plot(shap_values)
                plt.savefig(f"shap_{test_env}.png")

                break

        # clear agent buffer
        for i in range(args.n_agents):
            ppo_agents[i].buffer.clear()
        env.close()

    # save gif of test episode to disk
    save_gif(args, img_buffer)


def feature_importance_CIU(args, env):
    pass

# actually run some agent evaluation stuff here instead of just visualizing the results in a gif
## not a focus of the ECE class report, but maybe if we want to extend to a workshop paper or something
def evaluate_agent(args, env):
    pass


def save_gif(args, img_buffer):
    # output path
    if not os.path.exists(args.gif_dir):
        os.makedirs(args.gif_dir)
    save_dir = f"{args.gif_dir}/env_{args.env}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    gif_path = f"{save_dir}/policy_{args.policy_env}.gif"
    imageio.mimsave(gif_path, img_buffer, duration=args.gif_frame_duration)

    print("============================================================================================")
    print("total frames in gif : ", len(img_buffer))
    print("saved gif at : ", gif_path)
    print("============================================================================================")


def render_image(img_buffer, args, env, ep, t, done, ep_reward=None, save=True):
    img = env.render(mode="rgb_array", tile_size=75, highlight=True)
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
    parser.add_argument("--n_episodes_background", type=int, default=100, help="Number of episodes to use for SHAP or CIU as the 'background' dataset")
    parser.add_argument("--n_agents", type=int, default=1, help="Number of agents")
    parser.add_argument("--obs_size", type=int, default=3, help="Agent view width and height")
    parser.add_argument("--obs_dim", type=int, default=3, help="Agent view depth")
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
    parser.add_argument("--gif_frame_duration", type=float, default=0.5, help="Number of seconds per gif frame")
    parser.add_argument("--gif_n_end_frames", type=int, default=5, help="Number of still frames at the end of each episode's gif")
    parser.add_argument("--mode", type=str, default="visualize_agent", choices=["visualize_agent", "shap", "ciu"], help="if visualize_agent, Visualize agent performance with gifs of agent interacting with environment, else if shap or ciu, run those feature importance algorithms and create gifs of the importance")

    args = parser.parse_args()

    register(
        id="multigrid-rat-0-v0",
        entry_point="gym_multigrid.envs:CollectGameRat_0"
    )

    register(
            id="multigrid-rat-10-v0",
            entry_point="gym_multigrid.envs:CollectGameRat_10")

    register(
        id="multigrid-rat-50-v0",
        entry_point="gym_multigrid.envs:CollectGameRat_50"
    )

    register(
        id="multigrid-rat-100-v0",
        entry_point="gym_multigrid.envs:CollectGameRat_100"
    )

    env = gym.make(args.env)

    print("setting random seed to ", args.random_seed)
    torch.manual_seed(args.random_seed)
    env.seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    if args.mode == "visualize_agent":
        visualize_agent(args, env)
    elif args.mode == "shap":
        feature_importance_SHAP(args, env)
    elif args.mode == "ciu":
        feature_importance_CIU(args, env)
