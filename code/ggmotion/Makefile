# GGMotion Makefile
# Run from: ggmotion/code/

PYTHON ?= $(shell p=$(HOME)/anaconda3/envs/nlp_env/bin/python; [ -x "$$p" ] && echo "$$p" || command -v python3 2>/dev/null || command -v python 2>/dev/null)
CFG = cfg/cmu_short.yml

# --------------------------------------------------------------------------
# Training (RunPod GPU)
# --------------------------------------------------------------------------

.PHONY: train-xyz
train-xyz:
	@echo "=== Training GGMotion-XYZ (CMU short) ==="
	$(PYTHON) main_cmu.py --cfg $(CFG) --model xyz

.PHONY: train-dqpose
train-dqpose:
	@echo "=== Training GGMotion-DQPose (CMU short) ==="
	$(PYTHON) main_cmu.py --cfg cfg/cmu_dqpose_mindirect_short.yml --model dqpose --exp exp/dqposemindirect_retrained

# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

.PHONY: eval-xyz
eval-xyz:
	@echo "=== Evaluating GGMotion-XYZ ==="
	$(PYTHON) evaluate.py --cfg $(CFG) --model xyz --ckpt exp/xyz_reproduced/cmu_best_20.823_19.pt

.PHONY: eval-dqpose
eval-dqpose:
	@echo "=== Evaluating GGMotion-DQPose ==="
	$(PYTHON) evaluate.py --cfg cfg/cmu_dqpose_mindirect_short.yml --model dqpose --ckpt exp/dqposemindirect_retrained/cmu_best_24.595_22.pt

# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------

.PHONY: robustness
robustness:
	@echo "=== Running robustness experiment (rotation + translation) ==="
	$(PYTHON) -m experiments.robustness_exp --cfg cfg/cmu_dqpose_mindirect_short.yml --manner all \
		--dq-checkpoint exp/dqposemindirect_retrained/cmu_best_24.595_22.pt \
		--xyz-checkpoint exp/xyz_reproduced/cmu_best_20.823_19.pt

# --------------------------------------------------------------------------
# Equivariance verification
# --------------------------------------------------------------------------

.PHONY: test-equivariance
test-equivariance:
	@echo "=== Testing GGMotion-XYZ E(n) equivariance ==="
	$(PYTHON) test_equivariance.py --model xyz --n-trials 10

.PHONY: test-equivariance-dqpose
test-equivariance-dqpose:
	@echo "=== Testing GGMotion-DQPose SE(3) equivariance ==="
	$(PYTHON) test_equivariance.py --model dqpose --n-trials 10

.PHONY: verify
verify: test-equivariance test-equivariance-dqpose
	@echo ""
	@echo "=== All equivariance tests passed ==="

# --------------------------------------------------------------------------
# Install dependencies
# --------------------------------------------------------------------------

.PHONY: install
install:
	pip install -r requirements.txt
