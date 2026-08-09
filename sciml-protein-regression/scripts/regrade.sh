#!/bin/bash
# Re-score a finished trial without re-running the agent.
#
# Harbor 0.20 has no `regrade` command, but it does not need one: the agent's
# submission is already on disk under <trial>/artifacts/app/final_model. This
# rebuilds the verifier image (only the COPY layer changes when you edit
# grade.py) and runs it against that frozen submission, exactly as the real
# verifier does -- same entrypoint, no network, same private test set.
#
#   ./scripts/regrade.sh jobs/2026-08-09__14-08-50/sciml-protein-regression__ZnpRu34
#   ./scripts/regrade.sh --all
#
# Results go to <trial>/regrade/ so the original verifier/ output is never
# overwritten; the two are printed side by side.
set -euo pipefail

TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$TASK_DIR/.." && pwd)"
IMAGE="${REGRADE_IMAGE:-sciml-protein-verifier:regrade}"
MEMORY="${REGRADE_MEMORY:-8g}"

usage() {
    echo "usage: $0 <trial_dir> | --all [--no-build]" >&2
    exit 2
}

build_image() {
    echo "==> building verifier image ($IMAGE)"
    # Full tests/ context, same as Harbor: private_test/ and grade.py included.
    docker build -q -t "$IMAGE" "$TASK_DIR/tests" >/dev/null
    echo "    ok"
}

# Reads a numeric field out of a reward.json, or prints "-".
field() {
    python3 -c "
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    v=d.get(sys.argv[2])
    print('-' if v is None else (f'{v:.4f}' if isinstance(v,float) else v))
except Exception:
    print('-')
" "$1" "$2" 2>/dev/null || echo "-"
}

regrade_one() {
    local trial="${1%/}"
    local sub="$trial/artifacts/app/final_model"
    local out="$trial/regrade"
    local name
    name="$(basename "$trial")"

    if [ ! -d "$sub" ]; then
        echo "SKIP $name -- no artifacts/app/final_model (agent produced nothing)"
        return 0
    fi

    rm -rf "$out"
    mkdir -p "$out"

    # --network none matches the production verifier, which is offline. The
    # submission is mounted read-only so a buggy grader cannot mutate the
    # evidence it is scoring.
    docker run --rm \
        --network none \
        --memory "$MEMORY" \
        -v "$(cd "$sub" && pwd):/app/final_model:ro" \
        -v "$(cd "$out" && pwd):/logs/verifier" \
        "$IMAGE" \
        bash /tests/test.sh >"$out/test-stdout.txt" 2>&1 || true

    local old="$trial/verifier/reward.json"
    printf '%-38s old=%-8s new=%-8s | rho old=%-8s new=%-8s\n' \
        "$name" \
        "$(field "$old" reward)" "$(field "$out/reward.json" reward)" \
        "$(field "$old" spearman)" "$(field "$out/reward.json" spearman)"

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
                find "$REPO_ROOT/jobs" -maxdepth 2 -type d -name 'sciml-protein-regression__*' | sort
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
