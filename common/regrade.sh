#!/bin/bash
# Re-score a finished trial without re-running the agent. Shared by every task.
#
# Harbor 0.20 has no `regrade` command and does not need one: the agent's
# submission is already on disk under <trial>/artifacts/app/final_model. This
# rebuilds the verifier image (only the COPY layer changes when you edit grade.py
# or the grader/ fixtures) and runs it against that frozen submission, exactly as
# the real verifier does -- same entrypoint, no network, same private split.
#
# Called through each task's scripts/regrade.sh, which sets TASK_DIR:
#
#   ./tasks/qa-sft-adapt/scripts/regrade.sh jobs/2026-08-10__.../qa-sft-adapt__abc
#   ./tasks/qa-sft-adapt/scripts/regrade.sh --all
#
# Results go to <trial>/regrade/ so the original verifier/ output is never
# overwritten; old and new are printed side by side.
#
# Two things this does that a naive rerun would not, both load-bearing:
#
#   * The agent's log is mounted too. Every task ships /logs/agent/train_log.txt
#     as an artifact and every grader scans it for held-out data. Regrading
#     without it would exercise less of the grader than production does and could
#     silently clear a submission that production rejects.
#
#   * reward.json is single-key by design, and these tasks score several eval
#     sets, so per-set detail lives in metrics.json. Both are printed: the reward
#     alone cannot tell you which eval set moved.
set -euo pipefail

: "${TASK_DIR:?TASK_DIR must be set by the calling wrapper}"
TASK_DIR="$(cd "$TASK_DIR" && pwd)"
TASK_NAME="$(basename "$TASK_DIR")"
REPO_ROOT="$(cd "$TASK_DIR/../.." && pwd)"
IMAGE="${REGRADE_IMAGE:-${TASK_NAME}-verifier:regrade}"
MEMORY="${REGRADE_MEMORY:-8g}"

usage() {
    echo "usage: $0 <trial_dir> | --all [--no-build]" >&2
    exit 2
}

build_image() {
    echo "==> building verifier image ($IMAGE)"
    # Full tests/ context, same as Harbor: grader/ (private split, anchors,
    # base-model fixture, public hashes) and grade.py all included.
    docker build -q -t "$IMAGE" "$TASK_DIR/tests" >/dev/null
    echo "    ok"
}

reward_of() {
    python3 -c "
import json,sys
try:
    print(f\"{json.load(open(sys.argv[1]))['reward']:.6f}\")
except Exception:
    print('-')
" "$1" 2>/dev/null || echo "-"
}

# One line per eval set. Prints the uncapped recovery beside the capped one: a
# run that clips to 1.0 looks identical to one that just reaches the reference,
# and only the raw value distinguishes them -- which is how a reference anchor
# set too low stays invisible.
eval_sets_of() {
    python3 -c "
import json,sys
try:
    m = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
if m.get('status') not in (None, 'ok'):
    print(f\"    grader: {m.get('status')} -- {str(m.get('reason'))[:100]}\")
for name, e in (m.get('eval_sets') or {}).items():
    st = e.get('status', '-')
    metric = next((k for k in ('auc', 'acc', 'spearman') if k in e), None)
    val = e.get(metric) if metric else None
    rec, raw = e.get('recovery'), e.get('recovery_raw')
    bits = [f\"    {name:<10} {st:<9}\"]
    bits.append(f\"{metric}={val:.4f}\" if isinstance(val, float) else 'metric=-     ')
    bits.append(f\"recovery={rec:.4f}\" if isinstance(rec, float) else 'recovery=-     ')
    bits.append(f\"raw={raw:.4f}\" if isinstance(raw, float) else 'raw=-     ')
    for key, label in (('private_test_overlap', 'overlap'),
                       ('private_shingle_overlap', 'overlap'),
                       ('tensors_compared', 'tensors')):
        if e.get(key) is not None:
            bits.append(f\"{label}={e[key]}\")
    if st != 'ok' and e.get('reason'):
        bits.append(f\"| {str(e['reason'])[:70]}\")
    print(' '.join(bits))
" "$1" 2>/dev/null || true
}

regrade_one() {
    local trial="${1%/}"
    local sub="$trial/artifacts/app/final_model"
    local agent_logs="$trial/artifacts/logs/agent"
    local out="$trial/regrade"
    local name
    name="$(basename "$trial")"

    if [ ! -d "$sub" ]; then
        echo "SKIP $name -- no artifacts/app/final_model (agent produced nothing)"
        return 0
    fi

    rm -rf "$out"
    mkdir -p "$out"

    # --network none matches the production verifier, which is offline. Inputs
    # are read-only so a buggy grader cannot mutate the evidence it is scoring.
    local mounts=(-v "$(cd "$sub" && pwd):/app/final_model:ro")
    if [ -d "$agent_logs" ]; then
        mounts+=(-v "$(cd "$agent_logs" && pwd):/logs/agent:ro")
    fi

    docker run --rm \
        --network none \
        --memory "$MEMORY" \
        "${mounts[@]}" \
        -v "$(cd "$out" && pwd):/logs/verifier" \
        "$IMAGE" \
        bash /tests/test.sh >"$out/test-stdout.txt" 2>&1 || true

    local old="$trial/verifier/reward.json"
    printf '%-40s old=%-10s new=%-10s\n' \
        "$name" "$(reward_of "$old")" "$(reward_of "$out/reward.json")"
    eval_sets_of "$out/metrics.json"

    if [ ! -f "$out/reward.json" ]; then
        echo "  !! no reward.json written -- the always-write guarantee FAILED"
        tail -5 "$out/test-stdout.txt" | sed 's/^/     /'
    fi
}

DO_BUILD=1
TARGETS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --all)
            while IFS= read -r d; do TARGETS+=("$d"); done < <(
                find "$REPO_ROOT/jobs" -maxdepth 2 -type d -name "${TASK_NAME}__*" | sort
            )
            ;;
        --no-build) DO_BUILD=0 ;;
        -h|--help) usage ;;
        *) TARGETS+=("$1") ;;
    esac
    shift
done

[ ${#TARGETS[@]} -gt 0 ] || usage
[ "$DO_BUILD" -eq 1 ] && build_image

echo "==> regrading ${#TARGETS[@]} trial(s)"
for t in "${TARGETS[@]}"; do regrade_one "$t"; done
