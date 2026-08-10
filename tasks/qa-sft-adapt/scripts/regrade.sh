#!/bin/bash
# Thin wrapper: the regrade driver is shared by every task in this repo.
# See common/regrade.sh for what it does and why the agent log is mounted.
set -euo pipefail
TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" \
    exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/common/regrade.sh" "$@"
