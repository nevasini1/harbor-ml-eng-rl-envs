#!/bin/bash
# Separate-verifier entrypoint. Always (re)writes the reward channel.
set +e

chmod 700 /logs/verifier 2>/dev/null || mkdir -p /logs/verifier && chmod 700 /logs/verifier

python /tests/grade.py \
  --submission /app/final_model \
  --base /grader/base_model \
  --test /tests/private_test/test.csv.gz \
  --out /logs/verifier/reward.json

if [ ! -f /logs/verifier/reward.json ]; then
  echo '{"reward": 0.0, "reason": "grader_failed"}' > /logs/verifier/reward.json
  echo 0 > /logs/verifier/reward.txt
fi

exit 0
