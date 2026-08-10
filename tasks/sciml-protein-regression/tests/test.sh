#!/bin/bash
# Separate-verifier entrypoint. Always (re)writes the reward channel.
set +e

mkdir -p /logs/verifier && chmod 700 /logs/verifier

python /tests/grade.py \
  --submission /app/final_model \
  --base /grader/base_model \
  --test /tests/private_test/test.csv.gz \
  --out /logs/verifier/reward.json
rc=$?

# grade.py traps its own exceptions and writes a 0 reward, so reaching here
# with no reward.json means it was killed outright: OOM, verifier timeout,
# SIGKILL. Harbor's VerifierResult.rewards takes numbers only, so the reason
# string belongs in reward_meta.json -- putting it in reward.json makes the
# parse fail and loses the score, which is the exact failure this net exists
# to prevent.
if [ ! -f /logs/verifier/reward.json ]; then
  printf '{"reward": 0.0, "grader_exit_code": %d}\n' "$rc" > /logs/verifier/reward.json
  printf '{"reason": "grader_killed_before_write"}\n' > /logs/verifier/reward_meta.json
fi

# reward.txt is normally written by grade.py alongside reward.json; backstop it
# separately so a partial write cannot leave the plain-text channel empty.
if [ ! -f /logs/verifier/reward.txt ]; then
  echo 0.0 > /logs/verifier/reward.txt
fi

exit 0
