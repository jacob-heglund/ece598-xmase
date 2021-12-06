#!/bin/sh
python evaluate.py --env=multigrid-rat-50-v0 --policy_env=multigrid-rat-50-v0 --n_episodes=10
python evaluate.py --env=multigrid-rat-50-v0 --policy_env=multigrid-rat-10-v0 --n_episodes=10
python evaluate.py --env=multigrid-rat-50-v0 --policy_env=multigrid-rat-0-v0 --n_episodes=10

