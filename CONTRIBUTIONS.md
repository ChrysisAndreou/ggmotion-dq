# Contribution Boundary

This repository contains the released artifacts for the GGMotion-DQPose thesis work. It keeps the original GGMotion baseline surface intact where needed for reproducibility, and adds the final GGMotion-DQPose implementation and evidence chain.

## Inherited From `inkcat520/GGMotion`

The following files are inherited verbatim from the upstream GGMotion repository:

- `code/ggmotion/LICENSE`
- `code/ggmotion/utils/viz.py`
- `code/ggmotion/cfg/3dpw_long.yml`
- `code/ggmotion/cfg/3dpw_short.yml`
- `code/ggmotion/cfg/cmu_long.yml`
- `code/ggmotion/cfg/cmu_short.yml`
- `code/ggmotion/cfg/h36m_long.yml`
- `code/ggmotion/cfg/h36m_short.yml`
- `code/ggmotion/img/*`

These files remain under the upstream MIT license in `code/ggmotion/LICENSE`.

## Inherited Files Modified For This Thesis

These files began as GGMotion baseline files and were modified for the controlled XYZ reproduction, DQPose integration, device portability, or thesis-facing documentation:

- `code/ggmotion/.gitignore`
- `code/ggmotion/README.md`
- `code/ggmotion/main_3dpw.py`
- `code/ggmotion/main_h36m.py`
- `code/ggmotion/main_cmu.py`
- `code/ggmotion/module/model.py`
- `code/ggmotion/module/modules.py`
- `code/ggmotion/utils/data_loader.py`
- `code/ggmotion/utils/data_utils.py`
- `code/ggmotion/utils/kinematics.py`
- `code/ggmotion/utils/runtime.py`

The substantive thesis changes are concentrated in `main_cmu.py`, `utils/data_loader.py`, `utils/data_utils.py`, `utils/kinematics.py`, and `utils/runtime.py`. They add the DQPose data path, FK-derived pose dual quaternions, DQPose model selection, DQ-native loss terms, final evaluation hooks, and robustness support. The smaller changes keep the inherited scripts compatible with the updated shared training loop and non-CUDA-only environments.

## New Thesis Contributions

The following files were authored for this thesis release:

- `code/ggmotion/Makefile`
- `code/ggmotion/requirements.txt`
- `code/ggmotion/evaluate.py`
- `code/ggmotion/test_equivariance.py`
- `code/ggmotion/cfg/cmu_dqpose_mindirect_short.yml`
- `code/ggmotion/module/dq_block.py`
- `code/ggmotion/module/dq_ops.py`
- `code/ggmotion/module/model_dqpose.py`
- `code/ggmotion/experiments/_verify_xyz_180.py`
- `code/ggmotion/experiments/eval_per_action_robustness.py`
- `code/ggmotion/experiments/plot_loss_components.py`
- `code/ggmotion/experiments/plot_per_action_robustness.py`
- `code/ggmotion/experiments/plot_robustness.py`
- `code/ggmotion/experiments/plot_rollout_curves.py`
- `code/ggmotion/experiments/plot_training_curves.py`
- `code/ggmotion/experiments/robustness_exp.py`
- `code/ggmotion/experiments/run_dqpose_gpu_job.sh`
- `code/ggmotion/experiments/smoke_dqpose_step.py`
- `code/ggmotion/experiments/verify_dqpose_preprocessing.py`
- `code/ggmotion/experiments/visualize.py`
- `code/ggmotion/exp/xyz_reproduced/cmu_best_20.823_19.pt`
- `code/ggmotion/exp/dqposemindirect_retrained/cmu_best_24.595_22.pt`
- `code/ggmotion/experiments/results/*`
- `thesis/main.pdf`
- `supplementary/README.md`

These files support the final thesis comparison only: GGMotion-XYZ versus GGMotion-DQPose using the final `dqposemindirect` checkpoint.

## Dataset Provenance

The CMU benchmark files under `code/ggmotion/data/cmu/` are benchmark data, not original thesis code. They originate from the CMU Motion Capture Database and are used in the ConvSeq2Seq/GGMotion CMU 8-action benchmark format. They are included so the reported evaluation commands can run from a fresh clone.

## Licenses

- Upstream GGMotion code remains under the MIT license in `code/ggmotion/LICENSE`.
- Original thesis additions and modifications are licensed under the repository root `LICENSE`.
- The CMU Motion Capture Database files are provided for benchmark reproducibility and are not relicensed by either MIT license file.
