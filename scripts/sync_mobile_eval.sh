#!/usr/bin/env bash
# Round-trip mobile eval results through cluster judging.
#
# End-to-end pipeline:
#   1. Pull the eval run directory from the phone via adb (raw.json + trace
#      + transcripts).
#   2. Rsync the run dir up to the cluster's results_pending/.
#   3. Submit the judging SLURM job (Gemma4 26B on a100-80).
#   4. (Optional) wait for the job to finish, then pull the judged output back.
#
# Usage:
#   scripts/sync_mobile_eval.sh pull             # adb pull most recent run
#   scripts/sync_mobile_eval.sh push <run_dir>   # rsync to cluster
#   scripts/sync_mobile_eval.sh judge <run_dir>  # submit SLURM job, print job id
#   scripts/sync_mobile_eval.sh fetch <job_id>   # pull judged results when job done
#   scripts/sync_mobile_eval.sh all              # pull → push → judge → fetch (waits)
#
# Env vars:
#   ANDROID_PACKAGE   default: com.pocketpalai
#   CLUSTER_HOST      default: nus-student-cluster
#   CLUSTER_PROJECT   default: ~/projects/mindmate
#   LOCAL_RESULTS_DIR default: $HOME/mobile_eval_results

set -euo pipefail

ANDROID_PACKAGE="${ANDROID_PACKAGE:-com.pocketpalai}"
CLUSTER_HOST="${CLUSTER_HOST:-nus-student-cluster}"
CLUSTER_PROJECT="${CLUSTER_PROJECT:-~/projects/mindmate}"
LOCAL_RESULTS_DIR="${LOCAL_RESULTS_DIR:-$HOME/mobile_eval_results}"

PHONE_BASE="/sdcard/Android/data/${ANDROID_PACKAGE}/files/eval_runs"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

die() { echo "ERROR: $*" >&2; exit 1; }

usage() {
    cat <<EOF
Usage: $0 <command> [args]

Commands:
  pull              List or pull eval runs from connected Android device
  push <run_dir>    Rsync a local run directory to the cluster
  judge <run_dir>   Submit Gemma4 judging job; prints SLURM job id
  fetch <job_id>    Pull judged results back when SLURM job finishes
  all               Full round-trip: pull most recent → push → judge → wait → fetch

Env vars (override defaults):
  ANDROID_PACKAGE   (current: $ANDROID_PACKAGE)
  CLUSTER_HOST      (current: $CLUSTER_HOST)
  CLUSTER_PROJECT   (current: $CLUSTER_PROJECT)
  LOCAL_RESULTS_DIR (current: $LOCAL_RESULTS_DIR)
EOF
}

check_adb() {
    command -v adb >/dev/null || die "adb not found in PATH (install platform-tools)"
    adb devices | grep -q -E "device$" || die "no Android device connected (adb devices)"
}

check_ssh() {
    ssh -o ConnectTimeout=5 -o BatchMode=yes "$CLUSTER_HOST" 'echo ok' >/dev/null 2>&1 \
        || die "ssh $CLUSTER_HOST failed (VPN up?)"
}

# ──────────────────────────────────────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────────────────────────────────────

cmd_pull() {
    check_adb
    mkdir -p "$LOCAL_RESULTS_DIR"
    echo "[pull] Listing eval runs on device…"
    local runs
    runs=$(adb shell "ls -t $PHONE_BASE 2>/dev/null" | tr -d '\r' | grep -v '^$' || true)
    if [ -z "$runs" ]; then
        die "no eval runs found at $PHONE_BASE on device"
    fi
    local newest
    newest=$(echo "$runs" | head -n1)
    echo "[pull] Most recent run: $newest"
    local dest="$LOCAL_RESULTS_DIR/$newest"
    if [ -d "$dest" ]; then
        echo "[pull] Already pulled — overwrite? [y/N]"
        read -r ans
        [[ "${ans:-N}" =~ ^[Yy]$ ]] || die "aborted"
        rm -rf "$dest"
    fi
    mkdir -p "$dest"
    adb pull "$PHONE_BASE/$newest/." "$dest/" >/dev/null
    echo "[pull] ✓ $dest"
    # Sanity check
    [ -f "$dest/raw.json" ] || die "missing raw.json in pulled run — incomplete eval?"
    python3 -c "
import json
d = json.load(open('$dest/raw.json'))
print(f'  scenarios: {len(d[\"scenarios\"])}')
print(f'  model:     {d.get(\"model\")}')
print(f'  device:    {d.get(\"device\")}')
print(f'  timestamp: {d.get(\"timestamp\")}')
" 2>/dev/null || true
    # Print the local path so callers can use it
    echo "$dest"
}

cmd_push() {
    local run_dir="${1:-}"
    [ -n "$run_dir" ] || die "push requires <run_dir>"
    [ -d "$run_dir" ] || die "not a directory: $run_dir"
    check_ssh

    local basename
    basename=$(basename "$run_dir")
    local remote_dir="$CLUSTER_PROJECT/results_pending/$basename"

    echo "[push] Pushing $run_dir → $CLUSTER_HOST:$remote_dir"
    ssh "$CLUSTER_HOST" "mkdir -p $remote_dir"
    rsync -az --progress "$run_dir/" "$CLUSTER_HOST:$remote_dir/"
    echo "[push] ✓"
    echo "$remote_dir"
}

cmd_judge() {
    local run_dir="${1:-}"
    [ -n "$run_dir" ] || die "judge requires <run_dir> (cluster-relative or absolute)"
    check_ssh

    # Allow either local-style basename or full remote path
    local remote_dir
    if [[ "$run_dir" == /* || "$run_dir" == "~"* ]]; then
        remote_dir="$run_dir"
    else
        local basename
        basename=$(basename "$run_dir")
        remote_dir="$CLUSTER_PROJECT/results_pending/$basename"
    fi

    # Mobile eval writes raw.json directly inside the run dir.
    local raw_path="$remote_dir/raw.json"

    echo "[judge] Submitting judging SLURM job for $raw_path…"
    local out
    out=$(ssh "$CLUSTER_HOST" "sbatch \
        --gres=gpu:a100-80:1 \
        --export=ALL,MOBILE_RESULTS=$raw_path \
        $CLUSTER_PROJECT/benchmarks/judge_mobile.slurm")
    echo "[judge] $out"
    # Extract job id (last token of 'Submitted batch job NNNN')
    local job_id
    job_id=$(echo "$out" | grep -oE '[0-9]+$' || true)
    [ -n "$job_id" ] || die "couldn't parse SLURM job id from: $out"
    echo "[judge] ✓ job=$job_id"
    echo "$job_id"
}

cmd_fetch() {
    local job_id="${1:-}"
    [ -n "$job_id" ] || die "fetch requires <job_id>"
    check_ssh

    echo "[fetch] Waiting for job $job_id to finish…"
    while true; do
        local state
        state=$(ssh "$CLUSTER_HOST" "sacct -j $job_id --format=State --noheader -P 2>/dev/null | head -n1" \
            | tr -d ' \r\n' || true)
        if [ -z "$state" ]; then
            echo "[fetch] job $job_id not found yet, waiting…"
        else
            echo "[fetch] state=$state"
            case "$state" in
                COMPLETED) break ;;
                FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)
                    die "job $job_id ended in state $state — check $CLUSTER_HOST:~/logs/" ;;
                PENDING|RUNNING|CONFIGURING|REQUEUED) ;;
            esac
        fi
        sleep 10
    done

    echo "[fetch] Pulling judged results…"
    mkdir -p "$LOCAL_RESULTS_DIR/judged"
    # Find the most recent judged dir matching the job's output pattern.
    # Convention: judge_mobile_results.py writes to <output_dir>/<runtime>_<model>_judged_<ts>/
    local newest_judged
    newest_judged=$(ssh "$CLUSTER_HOST" \
        "ls -td $CLUSTER_PROJECT/benchmarks/results/*_judged_* 2>/dev/null | head -n1" \
        | tr -d '\r')
    [ -n "$newest_judged" ] || die "no judged results directory found"
    echo "[fetch] Newest: $newest_judged"
    rsync -az --progress "$CLUSTER_HOST:$newest_judged/" \
        "$LOCAL_RESULTS_DIR/judged/$(basename "$newest_judged")/"

    local local_judged="$LOCAL_RESULTS_DIR/judged/$(basename "$newest_judged")"
    echo "[fetch] ✓ $local_judged"

    # Quick summary
    if [ -f "$local_judged/judged.json" ]; then
        python3 -c "
import json
d = json.load(open('$local_judged/judged.json'))
s = d.get('summary', {})
print(f'  weighted_pct: {s.get(\"weighted_pct\", 0)*100:.1f}%')
print(f'  passed/failed: {s.get(\"passed_scenarios\")}/{s.get(\"failed_scenarios\")}')
print(f'  judge stats: {d.get(\"judge_stats\")}')
" 2>/dev/null || true
    fi
    echo "$local_judged"
}

cmd_all() {
    echo "═══ Pull ═══"
    local local_dir
    local_dir=$(cmd_pull | tail -n1)
    echo
    echo "═══ Push ═══"
    local remote_dir
    remote_dir=$(cmd_push "$local_dir" | tail -n1)
    echo
    echo "═══ Judge ═══"
    local job_id
    job_id=$(cmd_judge "$remote_dir" | tail -n1)
    echo
    echo "═══ Fetch ═══"
    local final
    final=$(cmd_fetch "$job_id" | tail -n1)
    echo
    echo "═══ Done ═══"
    echo "Judged output: $final"
}

# ──────────────────────────────────────────────────────────────────────────────
# Dispatch
# ──────────────────────────────────────────────────────────────────────────────

cmd="${1:-}"
case "$cmd" in
    pull)  shift; cmd_pull "$@" ;;
    push)  shift; cmd_push "$@" ;;
    judge) shift; cmd_judge "$@" ;;
    fetch) shift; cmd_fetch "$@" ;;
    all)   shift; cmd_all "$@" ;;
    ""|-h|--help|help) usage ;;
    *) usage; exit 1 ;;
esac
