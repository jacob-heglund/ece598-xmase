#!/bin/sh
python evaluate.py --env=multigrid-rat-50-v0 --policy_env=multigrid-rat-50-v0
# python evaluate.py --env=multigrid-rat-50-v0 --policy_env=multigrid-rat-10-v0
# python evaluate.py --env=multigrid-rat-50-v0 --policy_env=multigrid-rat-0-v0

# python evaluate.py --mode=shap --n_episodes_background=1 --n_episodes=1 --env=multigrid-rat-50-v0 --policy_env=multigrid-rat-50-v0

# python evaluate.py --mode=shap --n_episodes_background=1 --n_episodes=1 --env=multigrid-rat-50-v0 --policy_env=multigrid-rat-100-v0
