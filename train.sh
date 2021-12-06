#!/bin/sh
python train.py --env=multigrid-rat-0-v0
python train.py --env=multigrid-rat-10-v0
python train.py --env=multigrid-rat-50-v0
python train.py --env=multigrid-rat-100-v0

