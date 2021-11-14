import gym
import time
from gym.envs.registration import register
import argparse

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-e', '--env', default='watcher', type=str)

args = parser.parse_args()

def main():

    if args.env == 'watcher':
        register(
            id='multigrid-watcher-v0',
            entry_point='gym_multigrid.envs:CollectGameWatcher',
        )
        env = gym.make('multigrid-watcher-v0')
    else:
        exit

    _ = env.reset()

    nb_agents = len(env.agents)

    while True:
        env.render(mode='human', highlight=True)
        time.sleep(0.1)

        #ac = [env.action_space.sample() for _ in range(nb_agents)]
        actions = []
        for agent in range(nb_agents):
            if env.agents[agent].watcher:
                actions.append(env.action_space_watcher.sample())
            else:
                actions.append(env.action_space.sample())

        obs, _, done, _ = env.step(actions)

        if done:
            break

if __name__ == "__main__":
    main()