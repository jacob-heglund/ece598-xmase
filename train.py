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

from PPO import PPO
import pdb

################################### Training ###################################

def train():

    print("============================================================================================")


    ####### initialize environment hyperparameters ######
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="multigrid-single-agent-v0", choices=["multigrid-single-agent-v0", "multigrid-watcher-v0"])
    args = parser.parse_args()

    if args.env == "multigrid-single-agent-v0":
        n_agents = 1
        register(
            id="multigrid-single-agent-v0",
            entry_point="gym_multigrid.envs:CollectGameSingleAgent",
        )
        # size of agent's observation
        obs_size = [7]

    elif args.env == "multigrid-watcher-v0":
        n_agents = 2
        # TODO change for actual values of agent observation sizes
        register(
            id="multigrid-watcher-v0",
            entry_point="gym_multigrid.envs:CollectGameWatcher",
        )
        # agent 0 is the watcher with larger visual range
        # agent 1 is the collector with smaller visual range
        obs_size = [7, 3]

    obs_dim = 6

    has_continuous_action_space = False  # continuous action space; else discrete

    max_ep_len = 1000                   # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(10000)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)

    #####################################################
    ## Note : print/log frequencies should be > than max_ep_len
    ################ PPO hyperparameters ################

    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)

    #####################################################
    print("training environment name : " + args.env)
    env = gym.make(args.env)

    # action space dimension
    if has_continuous_action_space:
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

    run_num_pretrained = 0      #### change this to prevent overwriting weights in same args.env folder

    directory = "storage"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + "/" + args.env + "/"
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_paths = []
    for i in range(n_agents):
        checkpoint_path = directory + f"PPO_{args.env}_seed_{random_seed}_run_{run_num_pretrained}_agent_{i}.pth"
        checkpoint_paths.append(checkpoint_path)
        print("save checkpoint path : " + checkpoint_path)

    #####################################################
    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")

    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)

    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")

    print("--------------------------------------------------------------------------------------------")

    print("obs space dimension : ", obs_dim)
    print("action space dimension : ", action_dim)

    print("--------------------------------------------------------------------------------------------")

    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")

    else:
        print("Initializing a discrete action space policy")

    print("--------------------------------------------------------------------------------------------")

    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)

    print("--------------------------------------------------------------------------------------------")

    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)

    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    #####################################################
    print("============================================================================================")
    ################# training procedure ################

    # initialize a PPO agent
    ppo_agents = []
    # TODO pad the smaller agent"s view so it is the same size as larger agent view
    ## or make the CNN flexible
    for i in range(n_agents):
        ppo_agents.append(PPO(obs_dim, obs_size[i], action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std))


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
    log_running_reward = 0
    log_running_episodes = 0
    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        obs = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):
            # select action with policy
            actions = []
            for i in range(n_agents):
                actions.append(ppo_agents[i].select_action(obs[i]))
            obs, reward, done, _ = env.step(actions)

            # saving reward and is_terminals
            for i in range(n_agents):
                ppo_agents[i].buffer.rewards.append(reward)
                ppo_agents[i].buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                for i in range(n_agents):
                    ppo_agents[i].update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write(f"{i_episode},{time_step},{log_avg_reward}\n")
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print(f"Episode : {i_episode} \t\t Timestep : {time_step} \t\t Average Reward : {print_avg_reward}")

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                for i in range(n_agents):
                    print("saving model at : " + checkpoint_paths[i])
                    ppo_agents[i].save(checkpoint_paths[i])

                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")
            #TODO the agents don't seem to ever reach the goal in single agent
            ## haven't seens success in multi-agent yet either
            ## lol there isn't a goal for the agent to reach for some reason
            # break; if the episode is over
            if done:
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
    train()
