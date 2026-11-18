#!/bin/bash
# Wait for a cluster job submitted by remote_run.sh, then fetch its outputs.
#
#   remote_wait.sh <run_name> <jobid> <local_dir>
#
# Blocks until the job has left the queue (the waiting happens on the server, one cheap check
# every 20 s), prints its final state and the tail of its log, and copies $REMOTE_OUT and the
# log into <local_dir>. Run it in the background and act on its completion; do not poll.

# ---- set this for your own cluster login (the same alias as in remote_run.sh) ----
HOST=your-ssh-alias
# ------------------------------------------------------------------------------------

set -euo pipefail
[ $# -eq 3 ] || { sed -n '2,8p' "$0"; exit 2; }
NAME=$1; JOB=$2; DEST=$3
# a dropped connection is retried: the job itself is unaffected
until ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=20 "$HOST" "while squeue -h -j $JOB 2>/dev/null | grep -q .; do sleep 20; done; echo \"job $JOB: \$(sacct -n -X -j $JOB -o State,Elapsed,ExitCode 2>/dev/null | head -1)\"; echo '--- log tail'; tail -n 30 ~/scalar-runs/logs/$NAME.log" 2>&1 | grep -v -E '^Loading|requirement: slurm'; [ "${PIPESTATUS[0]}" -eq 0 ]; do echo "connection lost, retrying"; sleep 30; done
mkdir -p "$DEST"
rsync -a -e 'ssh -o BatchMode=yes' "$HOST:scalar-runs/out/$NAME/" "$DEST/" 2>&1 | grep -v -E '^Loading|requirement: slurm' || true
rsync -a -e 'ssh -o BatchMode=yes' "$HOST:scalar-runs/logs/$NAME.log" "$DEST/slurm.log" 2>&1 | grep -v -E '^Loading|requirement: slurm' || true
echo "--- fetched into $DEST:"; ls -la "$DEST" | tail -n +2 | awk '{print $5, $9}'
