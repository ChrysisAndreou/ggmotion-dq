# GGMotion-DQPose Code

This directory contains the code, configs, checkpoints, and evaluation scripts for the final thesis experiments.

## Reported Models

- **GGMotion-XYZ:** Cartesian baseline, checkpoint `exp/xyz_reproduced/cmu_best_20.823_19.pt`.
- **GGMotion-DQPose:** final DQ-native model, checkpoint `exp/dqposemindirect_retrained/cmu_best_24.595_22.pt`.

## Data

This repository includes the CMU 8-action files used by the thesis benchmark. The data originates from the CMU Motion Capture Database and is used through the ConvSeq2Seq/GGMotion benchmark format.

If replacing the data locally, place CMU Mocap data under:

```text
ggmotion/code/data/cmu
```

## Evaluation

```bash
make eval-xyz
make eval-dqpose
```

Expected 400 ms MPJPE:

- XYZ: 35.3 mm
- DQPose: 41.8 mm

## Verification

```bash
make test-equivariance-dqpose
python experiments/verify_dqpose_preprocessing.py --manner 8
make robustness
```

## Figures and Videos

The thesis qualitative figures and MP4 artifacts are created from the final
DQPose and XYZ checkpoints:

```bash
python experiments/visualize.py --actions walking --multi-row --robustness-figure --canonical-overlay
```

Full action-level artifacts are stored under:

```text
ggmotion/code/experiments/results/visualizations
```

The repository root `supplementary/README.md` links directly to the MP4 artifacts.

## Training

The final DQPose training config is:

```bash
python main_cmu.py --cfg cfg/cmu_dqpose_mindirect_short.yml --model dqpose --exp exp/dqposemindirect_retrained
```

The thesis uses the saved DQPose checkpoint listed above.

## Contribution Boundary

The repository root `CONTRIBUTIONS.md` separates inherited GGMotion files, inherited files modified for this thesis, and newly authored DQPose files. The upstream GGMotion MIT license remains in this directory as `LICENSE`; the repository root license covers the thesis additions.
