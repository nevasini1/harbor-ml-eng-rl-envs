#!/bin/bash
# Thin wrapper: the regrade driver is shared by every task in this repo.
#
# This used to be a 130-line script specific to this task. It was the first one
# written, and the other two tasks needed exactly the same thing, so it moved to
# common/regrade.sh -- unchanged in behaviour, verified by regrading
# jobs/mol-oracle-modal through both and getting the same reward.
#
# The two details that made this task's version different are now the shared
# default, because they were never really task-specific:
#
#   * The agent's log is mounted. Every task ships /logs/agent/train_log.txt as
#     an artifact and every grader scans it for held-out data. Regrading without
#     it exercises less of the grader than production does and could silently
#     clear a submission that production rejects.
#
#   * Per-eval-set detail is printed from metrics.json, not just the reward.
#     reward.json is single-key by design, and the reward alone cannot tell you
#     which eval set moved.
set -euo pipefail
TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" \
    exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/common/regrade.sh" "$@"
