# based on https://github.com/nikhilbarhate99/PPO-PyTorch, extended to multi-agent case

import os
import glob
import time
from datetime import datetime
import argparse
from gym.envs.registration import register

import torch
import numpy as np
import gym
import random

from PPO import PPO
import pdb

################################### Training ###################################

def train(args, env):
    print_freq = args.max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = args.max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    update_timestep = args.max_ep_len * 4

    # action space dimension
    if args.has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ###################### logging ######################
    #### log files for multiple runs are NOT overwritten

    log_dir = "logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + "/" + args.env + "/"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0

    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + "/PPO_" + args.env + "_log_" + str(run_num) + ".txt"

    print("current logging run number for " + args.env + " : ", run_num)
    print("logging at : " + log_f_name)

    #####################################################
    ################### checkpointing ###################

    directory = "storage"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + "/" + args.env + "/"
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_paths = []
    for i in range(args.n_agents):
        checkpoint_path = directory + f"PPO_{args.env}_seed_{args.random_seed}_run_{args.run_num_pretrained}_agent_{i}.pth"
        checkpoint_paths.append(checkpoint_path)
        print("save checkpoint path : " + checkpoint_path)


    # initialize PPO agents
    ppo_agents = []
    for i in range(args.n_agents):
        ppo_agents.append(PPO(args.obs_dim, args.obs_size, action_dim, args.lr_actor, args.lr_critic, args.gamma, args.K_epochs, args.eps_clip, args.has_continuous_action_space, args.action_std))

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("============================================================================================")
    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write("episode,timestep,reward\n")

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0
    print_avg_reward = 0
    log_running_reward = 0
    log_running_episodes = 0
    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= args.max_training_timesteps:

        obs = env.reset()
        current_ep_reward = 0

        for t in range(1, args.max_ep_len+1):
            # select action with policy
            actions = []
            for i in range(args.n_agents):
                action, _ = ppo_agents[i].select_action(obs[i])
                actions.append(action)
            obs, reward, done, _ = env.step(actions)

            # saving reward and is_terminals
            for i in range(args.n_agents):
                ppo_agents[i].buffer.rewards.append(reward)
                ppo_agents[i].buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                for i in range(args.n_agents):
                    ppo_agents[i].update()

            # if continuous action space; then decay action std of ouput action distribution
            if args.has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(args.action_std_decay_rate, args.min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward.item(), 4)

                log_f.write(f"{i_episode},{time_step},{log_avg_reward}\n")
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # average reward until last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward.item(), 4)

                print(f"Episode : {i_episode} \t Timestep : {time_step} \t Average Reward : {print_avg_reward}")

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % args.save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                for i in range(args.n_agents):
                    print("saving model at : " + checkpoint_paths[i])
                    ppo_agents[i].save(checkpoint_paths[i])

                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")
            # break; if the episode is over
            if done:
                break

        if print_avg_reward >= args.stop_training_thres:
            break

        print_running_reward += current_ep_reward
        print_running_episodes += 1
        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="multigrid-rat-50-v0", choices=["multigrid-rat-0-v0","multigrid-rat-10-v0", "multigrid-rat-50-v0", "multigrid-rat-100-v0"], help="Environment used to evaluate the agent policy")
    parser.add_argument("--n_episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--n_agents", type=int, default=1, help="Number of agents")
    parser.add_argument("--obs_size", type=int, default=3, help="Agent view width and height")
    parser.add_argument("--obs_dim", type=int, default=6, help="Agent view depth")
    parser.add_argument("--K_epochs", type=int, default=80, help="Number of epochs between PPO update")
    parser.add_argument("--max_training_timesteps", type=int, default=int(3e6), help="Max number of training time steps")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="Clipping for PPO update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discounting rate")
    parser.add_argument("--lr_actor", type=float, default=0.0003, help="Actor learning rate")
    parser.add_argument("--lr_critic", type=float, default=0.001, help="Critic learning rate")
    parser.add_argument("--has_continuous_action_space", type=bool, default=False, help="Continuous vs discrete action space")
    parser.add_argument("--action_std_decay_rate", type=float, default=0.05, help="Linearly decay action_std, action_std = action_std - action_std_decay_rate")
    parser.add_argument("--min_action_std", type=float, default=0.1, help="Minimum action_std, stop decay after action_std <= min_action_std")
    parser.add_argument("--action_std_decay_freq", type=int, default=int(2.5e5), help="Action_std decay frequency in number of time steps")
    parser.add_argument("--stop_training_thres", type=float, default=0.99, help="Stop training when average reward >= stop_training_thres")
    parser.add_argument("--max_ep_len", type=int, default=1000, help="Max number of time steps for an episode")
    parser.add_argument("--action_std", type=float, default=0.6, help="Starting std. dev. for action distribution (Multivariate Normal)")
    parser.add_argument("--random_seed", type=int, default=1, help="Random seed")
    parser.add_argument("--run_num_pretrained", type=int, default=0, help="Load a policy from a particular run number")
    parser.add_argument("--save_model_freq", type=int, default=10000, help="Number of time steps between model saves")

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

    train(args, env)
