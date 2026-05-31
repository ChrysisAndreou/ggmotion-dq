#!/usr/bin/env bash
set -euo pipefail

DQPOSE_CFG="${DQPOSE_CFG:-cfg/cmu_dqpose_mindirect_short.yml}"
DQPOSE_EXP="${DQPOSE_EXP:-exp/dqposemindirect}"
DQPOSE_STABLE_DIR="${DQPOSE_STABLE_DIR:-exp/dqposemindirect_retrained}"
DQPOSE_RESULT_PREFIX="${DQPOSE_RESULT_PREFIX:-dqposemindirect}"

mkdir -p "${DQPOSE_STABLE_DIR}" "experiments/results/${DQPOSE_RESULT_PREFIX}_robustness"

echo "=== DQPose preprocessing verification ==="
python experiments/verify_dqpose_preprocessing.py --act walking --manner 8

read -r DQPOSE_EQ_DECODE_MODE DQPOSE_POSE_ROT_SCALE DQPOSE_POSE_TRANS_SCALE DQPOSE_VEL_ROT_SCALE DQPOSE_VEL_TRANS_SCALE DQPOSE_OUT_ROT_SCALE DQPOSE_OUT_TRANS_SCALE DQPOSE_OUT_ROT_SCALE_START DQPOSE_OUT_ROT_SCALE_END DQPOSE_OUT_TRANS_SCALE_START DQPOSE_OUT_TRANS_SCALE_END DQPOSE_BLOCK_MODE <<<"${DQPOSE_EQ_CONFIG:-$(python - <<PY
import yaml
with open("${DQPOSE_CFG}", "r") as f:
    cfg = yaml.safe_load(f)
out_rot = cfg.get("dqpose_out_rot_scale", 1.0)
out_trans = cfg.get("dqpose_out_trans_scale", 1.0)
print(
    cfg.get("dqpose_decode_mode", "direct"),
    cfg.get("dqpose_pose_rot_scale", 1.0),
    cfg.get("dqpose_pose_trans_scale", 1.0),
    cfg.get("dqpose_vel_rot_scale", 1.0),
    cfg.get("dqpose_vel_trans_scale", 1.0),
    out_rot,
    out_trans,
    cfg.get("dqpose_out_rot_scale_start", out_rot),
    cfg.get("dqpose_out_rot_scale_end", out_rot),
    cfg.get("dqpose_out_trans_scale_start", out_trans),
    cfg.get("dqpose_out_trans_scale_end", out_trans),
    cfg.get("dqpose_block_mode", "joint"),
)
PY
)}"

echo "=== DQPose equivariance verification ==="
python test_equivariance.py \
  --model dqpose \
  --dqpose-decode-mode "${DQPOSE_EQ_DECODE_MODE}" \
  --dqpose-pose-rot-scale "${DQPOSE_POSE_ROT_SCALE}" \
  --dqpose-pose-trans-scale "${DQPOSE_POSE_TRANS_SCALE}" \
  --dqpose-vel-rot-scale "${DQPOSE_VEL_ROT_SCALE}" \
  --dqpose-vel-trans-scale "${DQPOSE_VEL_TRANS_SCALE}" \
  --dqpose-out-rot-scale "${DQPOSE_OUT_ROT_SCALE}" \
  --dqpose-out-trans-scale "${DQPOSE_OUT_TRANS_SCALE}" \
  --dqpose-out-rot-scale-start "${DQPOSE_OUT_ROT_SCALE_START}" \
  --dqpose-out-rot-scale-end "${DQPOSE_OUT_ROT_SCALE_END}" \
  --dqpose-out-trans-scale-start "${DQPOSE_OUT_TRANS_SCALE_START}" \
  --dqpose-out-trans-scale-end "${DQPOSE_OUT_TRANS_SCALE_END}" \
  --dqpose-block-mode "${DQPOSE_BLOCK_MODE}" \
  --n-trials 10 \
  --device cuda

echo "=== DQPose training ==="
python main_cmu.py \
  --cfg "${DQPOSE_CFG}" \
  --model dqpose \
  --exp "${DQPOSE_EXP}" \
  2>&1 | tee "experiments/results/${DQPOSE_RESULT_PREFIX}_train.txt"

if grep -q ">>> early abort" "experiments/results/${DQPOSE_RESULT_PREFIX}_train.txt"; then
  echo "Training hit the failed-trajectory early-abort gate; skipping eval, robustness, and artifact commit."
  exit 20
fi

BEST_CKPT="$(python - <<PY
from pathlib import Path
import re

paths = list(Path("${DQPOSE_EXP}").glob('*/cmu_best_*.pt'))
paths += list(Path("${DQPOSE_EXP}").glob('cmu_best_*.pt'))
if not paths:
    raise SystemExit('No dqpose checkpoints found')

def key(path):
    m = re.search(r'cmu_best_([0-9.]+)_([0-9]+)\.pt$', path.name)
    if not m:
        return (float('inf'), -1)
    return (float(m.group(1)), -int(m.group(2)))

print(min(paths, key=key))
PY
)"

STABLE_CKPT="${DQPOSE_STABLE_DIR}/$(basename "${BEST_CKPT}")"
cp "${BEST_CKPT}" "${STABLE_CKPT}"
echo "Best checkpoint: ${STABLE_CKPT}"

echo "=== DQPose evaluation ==="
python evaluate.py \
  --cfg "${DQPOSE_CFG}" \
  --model dqpose \
  --ckpt "${STABLE_CKPT}" \
  --manner all \
  2>&1 | tee "experiments/results/${DQPOSE_RESULT_PREFIX}_eval.txt"

echo "=== DQPose robustness ==="
python experiments/robustness_exp.py \
  --cfg "${DQPOSE_CFG}" \
  --dq-checkpoint "${STABLE_CKPT}" \
  --xyz-checkpoint exp/xyz_reproduced/cmu_best_20.823_19.pt \
  --manner all \
  --device cuda \
  --out-dir "experiments/results/${DQPOSE_RESULT_PREFIX}_robustness" \
  2>&1 | tee "experiments/results/${DQPOSE_RESULT_PREFIX}_robustness.txt"

if [[ "${DQPOSE_GPU_UPLOAD_WANDB_ARTIFACT:-0}" == "1" ]]; then
  echo "=== Upload GPU artifacts to W&B ==="
  python - <<'PY'
from datetime import datetime
from pathlib import Path
import os

import wandb

run = wandb.init(
    project=os.environ.get("WANDB_PROJECT", "ggmotion"),
    name=os.environ.get("DQPOSE_RESULT_PREFIX", "dqposeres") + "-gpu-artifacts-" + datetime.utcnow().strftime("%m%d-%H%M"),
    job_type="gpu-artifacts",
    tags=["cmu", "dqpose", "gpu-artifacts"],
)
prefix = os.environ.get("DQPOSE_RESULT_PREFIX", "dqposeres")
stable_dir = os.environ.get("DQPOSE_STABLE_DIR", "exp/dqposeres_retrained")
artifact = wandb.Artifact(prefix + "-gpu-results", type="model")
patterns = [
    f"experiments/results/{prefix}_train.txt",
    f"experiments/results/{prefix}_eval.txt",
    f"experiments/results/{prefix}_robustness.txt",
    f"experiments/results/{prefix}_robustness/*.csv",
    f"{stable_dir}/*.pt",
]
added = 0
for pattern in patterns:
    for path in sorted(Path(".").glob(pattern)):
        if path.is_file():
            artifact.add_file(str(path), name=str(path))
            added += 1
if added == 0:
    raise SystemExit("No DQPose GPU artifacts found to upload")
run.log_artifact(artifact)
run.finish()
print(f"Uploaded {added} files to W&B artifact {prefix}-gpu-results")
PY
fi

if [[ "${DQPOSE_GPU_SKIP_GIT:-0}" == "1" ]]; then
  echo "Skipping git commit/push because DQPOSE_GPU_SKIP_GIT=1"
else
  echo "=== Commit GPU artifacts ==="
  git config user.email 'chrysisandreou@outlook.com'
  git config user.name 'Chrysis Andreou'
  git add "experiments/results/${DQPOSE_RESULT_PREFIX}_train.txt" \
          "experiments/results/${DQPOSE_RESULT_PREFIX}_eval.txt" \
          "experiments/results/${DQPOSE_RESULT_PREFIX}_robustness.txt" \
          experiments/results/${DQPOSE_RESULT_PREFIX}_robustness/*.csv
  git add -f "${DQPOSE_STABLE_DIR}"/*.pt
  git commit -m "${DQPOSE_COMMIT_MSG:-Add DQPose GPU results}" || echo "No GPU artifact changes to commit"
  git push origin main
fi

echo "DONE"
