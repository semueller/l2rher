#!/bin/sh

CMD="python play_policy.py --logdir ./logs/ --base_path ./ppath/ --policy periodic/1/policy.pkl"

echo "execute $CMD"

$CMD
