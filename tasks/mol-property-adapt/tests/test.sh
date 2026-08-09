#!/bin/bash
# Verifier entrypoint. Runs in a separate container with no network; the private
# split, the base model and the grader are baked into the image at build time.
#
# The reward file is written on every path, including grader failure, and this
# script always overwrites whatever the agent may have left behind.

set -u

REWARD_DIR=/logs/verifier
mkdir -p "${REWARD_DIR}"
# The agent must never be able to reach the reward channel.
chmod 700 "${REWARD_DIR}"

# Discard any reward file the agent may have planted.
rm -f "${REWARD_DIR}/reward.json" "${REWARD_DIR}/reward.txt"

timeout 3000 python /tests/grade.py \
    --submission /app/final_model \
    --base /grader/base_model \
    --private /grader/private \
    --anchors /grader/private/anchors.json \
    --public-hashes /grader/public_hashes.json \
    --out "${REWARD_DIR}/reward.json" \
    --metrics-out "${REWARD_DIR}/metrics.json"
rc=$?

if [ ! -s "${REWARD_DIR}/reward.json" ]; then
    echo "grader produced no reward (exit ${rc}); defaulting to 0" >&2
    printf '{"reward": 0.0}' > "${REWARD_DIR}/reward.json"
    printf '{"status": "grader_failed", "exit_code": %s}' "${rc}" \
        > "${REWARD_DIR}/metrics.json"
fi

cat "${REWARD_DIR}/reward.json"
exit 0
