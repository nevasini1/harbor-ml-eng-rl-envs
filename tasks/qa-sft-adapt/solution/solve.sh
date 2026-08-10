#!/bin/bash
# Oracle solution: the reference recipe used to set the upper anchor.
#
# It is deliberately a competent-but-ordinary supervised fine-tune rather than a
# maximal one, so that a strong agent can exceed it and earn the full reward.
set -euo pipefail

mkdir -p /logs/agent /app/final_model
python /solution/train_reference.py 2>&1 | tee /logs/agent/train_log.txt
