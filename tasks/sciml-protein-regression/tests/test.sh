#!/bin/bash
# Separate-verifier entrypoint. Always (re)writes the reward channel.
set +e

mkdir -p /logs/verifier && chmod 700 /logs/verifier

# Discard any reward file the agent may have planted. The other three tasks have
# carried this line since they were written; this one did not, and the guard below
# is a *presence* test -- `[ ! -f ... ]` is false for a planted file, so on the
# killed-grader path a leftover reward.json would be the score Harbor reads.
# `/logs/verifier` is not in this task's `artifacts`, so that path is not currently
# open; this is the defence-in-depth the other graders already have, not a live
# hole being closed.
rm -f /logs/verifier/reward.json /logs/verifier/reward.txt /logs/verifier/reward_meta.json

# `timeout` matched to the other three. Without it the only thing that ends a
# runaway grade is the harness SIGKILL, which is precisely the case the net below
# has to catch; with it, the overrun is this script's own catchable exit code.
timeout 3000 python /tests/grade.py \
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
