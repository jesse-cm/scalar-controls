#!/bin/bash
# Submit one simulation command to the WashU engineering cluster (Slurm; set HOST, PARTITION, ACCOUNT below).
#
#   remote_run.sh <worktree> <run_name> <cores 1-8> <hh:mm:ss> <torch 0|1> <command ...>
#
# The worktree must be committed and clean: the commit is pushed to a private bare repo in your
# cluster home and cloned there, so every ledger row carries a real commit hash. Create that repo
# once: ssh <HOST> git init --bare scalar-runs/repo.git
# The command runs from the repo root on a compute node. It must write its outputs under
# $REMOTE_OUT (a directory outside the checkout), e.g.
#   python3 scripts/matrix.py --experiments 6 --seeds 0-39 --workers 8 --ledger "$REMOTE_OUT/rows.csv"
# Prints the Slurm job id. Then: remote_wait.sh <run_name> <jobid> <local_dir>
# Queue limits (QOS academic-job): 8 cores per job, 4 jobs at once per user, 8 h per job.

# ---- set these for your own cluster login ----
HOST=your-ssh-alias          # a Host entry in ~/.ssh/config that logs in to the cluster without a password prompt
PARTITION=your-partition     # Slurm partition (sbatch -p)
ACCOUNT=your-slurm-account   # Slurm account (sbatch -A)
# -----------------------------------------------

set -euo pipefail
[ $# -ge 6 ] || { sed -n '2,13p' "$0"; exit 2; }
WT=$1; NAME=$2; CORES=$3; TIME=$4; TORCH=$5; shift 5
CMD="$*"
case "$NAME" in *[!A-Za-z0-9_.-]*) echo "run_name: letters, digits, _ . - only"; exit 2;; esac
[ "$CORES" -ge 1 ] && [ "$CORES" -le 8 ] || { echo "cores must be 1-8 (queue limit)"; exit 2; }
cd "$WT"
[ -z "$(git status --porcelain)" ] || { echo "worktree has uncommitted changes; commit them first"; git status --short; exit 2; }
SHA=$(git rev-parse HEAD)
git push -q -f "$HOST:scalar-runs/repo.git" "HEAD:refs/heads/run/$NAME" 2>&1 | grep -v -E '^Loading|requirement: slurm' || true
ssh -o BatchMode=yes "$HOST" bash -s -- "$NAME" "$SHA" "$CORES" "$TIME" "$TORCH" "$(printf '%s' "$CMD" | base64 | tr -d '\n')" "$PARTITION" "$ACCOUNT" <<'REMOTE' 2>&1 | grep -v -E '^Loading|requirement: slurm'
set -euo pipefail
NAME=$1; SHA=$2; CORES=$3; TIME=$4; TORCH=$5; CMD=$(echo "$6" | base64 -d); PARTITION=$7; ACCOUNT=$8
R=$HOME/scalar-runs; D=$R/src/$NAME; O=$R/out/$NAME
mkdir -p "$R/src" "$R/jobs" "$R/logs"; rm -rf "$D" "$O"; mkdir -p "$O"
git clone -q -b "run/$NAME" "$R/repo.git" "$D"
[ "$(git -C "$D" rev-parse HEAD)" = "$SHA" ] || { echo "clone is not at $SHA"; exit 3; }
V="/tmp/$USER-scalar-venv"; EXTRA=""
if [ "$TORCH" = 1 ]; then V="$V-torch"; EXTRA='pip -q install torch --index-url https://download.pytorch.org/whl/cpu'; fi
cat > "$R/jobs/$NAME.sh" <<JOB
#!/bin/bash
set -u
V=$V
(
  flock 9
  if [ ! -f "\$V/.ready" ]; then
    rm -rf "\$V"
    export PIP_CACHE_DIR=/tmp/\$USER-pip-cache
    python3 -m venv "\$V" && . "\$V/bin/activate" && pip -q install --upgrade pip \\
      && pip -q install -r "$D/requirements.txt" pytest ${EXTRA:+&& $EXTRA} && touch "\$V/.ready"
  fi
) 9>"\$V.lock"
[ -f "\$V/.ready" ] || { echo "environment build failed"; exit 4; }
. "\$V/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export REMOTE_OUT="$O"
cd "$D"
echo "node \$(hostname) | \${SLURM_CPUS_PER_TASK:-?} cores | \$(python --version) | commit ${SHA:0:7} | start \$(date +%H:%M:%S)"
$CMD
rc=\$?
echo "exit \$rc | end \$(date +%H:%M:%S) | tree \$(git status --porcelain | wc -l) changed files"
exit \$rc
JOB
sbatch --parsable -p "$PARTITION" -A "$ACCOUNT" -c "$CORES" -t "$TIME" -J "$NAME" -o "$R/logs/$NAME.log" "$R/jobs/$NAME.sh"
REMOTE
