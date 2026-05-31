# GGMotion-DQPose for Human Motion Prediction

MSc Thesis -- Chrysis Andreou

## Overview

This repository contains the code, checkpoints, evaluation scripts, thesis PDF, and supplementary videos for the MSc thesis:
**"SE(3)-Equivariant Graph Neural Networks with Dual Quaternions for Human Motion Prediction"**

The final thesis compares two models on the CMU Motion Capture 8-action benchmark:

- **GGMotion-XYZ:** a Cartesian reproduction of the GGMotion backbone.
- **GGMotion-DQPose:** the final DQ-native counterpart using FK-derived per-joint pose dual quaternions.

GGMotion-XYZ is stronger on canonical-frame MPJPE. GGMotion-DQPose trails on that benchmark metric, but remains exactly robust under the tested global SE(3) reference-frame changes.

The contribution boundary is documented in [CONTRIBUTIONS.md](CONTRIBUTIONS.md). The final thesis PDF is [thesis/main.pdf](thesis/main.pdf). The defense presentation is available as an editable [PowerPoint deck](presentation/Andreou_Chrysis_MSc_Defense.pptx) and a [PDF preview](presentation/Andreou_Chrysis_MSc_Defense.pdf). Supplementary videos are indexed in [supplementary/README.md](supplementary/README.md).

## Key Results

MPJPE in millimetres on the CMU 8-action benchmark, 25 fps, 10 observed frames, and 10 predicted frames.

| Model | 80 ms | 160 ms | 320 ms | 400 ms | Avg |
|---|---:|---:|---:|---:|---:|
| GGMotion-XYZ | 7.2 | 13.3 | 27.4 | 35.3 | 20.8 |
| GGMotion-DQPose | 8.4 | 15.7 | 32.4 | 41.8 | 24.6 |

At 400 ms, GGMotion-XYZ reaches 35.3 mm and GGMotion-DQPose reaches 41.8 mm. Under the robustness test, GGMotion-DQPose stays constant to numerical precision under rotations up to 180 degrees and translations up to 5.0 m. GGMotion-XYZ rises to 47.8 mm under rotation and 2074.3 mm under 5.0 m translation at the 400 ms horizon.

## Repository Structure

```text
code/
  ggmotion/
    module/
      model.py                  # GGMotion-XYZ baseline
      model_dqpose.py           # Final GGMotion-DQPose model
      dq_block.py               # Shared 6D DQ log-coordinate block
      dq_ops.py                 # Dual quaternion operations
      modules.py                # Shared building blocks
    utils/
      data_loader.py            # CMU loaders, including FK-derived DQPose data
      data_utils.py             # Data preprocessing utilities
      kinematics.py             # Forward kinematics
      runtime.py                # Training loop
    cfg/
      cmu_short.yml
      cmu_dqpose_mindirect_short.yml
    exp/
      xyz_reproduced/
      dqposemindirect_retrained/
    experiments/
      results/
      robustness_exp.py
      visualize.py
    Makefile
thesis/
  main.pdf
presentation/
  Andreou_Chrysis_MSc_Defense.pptx
  Andreou_Chrysis_MSc_Defense.pdf
supplementary/
  README.md
CONTRIBUTIONS.md
LICENSE
```

## Setup

```bash
pip install -r code/ggmotion/requirements.txt
```

Place CMU Motion Capture data under:

```text
code/ggmotion/data/cmu
```

This repository includes the CMU 8-action files used by the thesis benchmark. The data originates from the CMU Motion Capture Database and is used through the ConvSeq2Seq/GGMotion benchmark format; see [CONTRIBUTIONS.md](CONTRIBUTIONS.md) for provenance notes.

## Evaluation

```bash
cd code/ggmotion
make eval-xyz
make eval-dqpose
```

Expected 400 ms MPJPE:

- GGMotion-XYZ: 35.3 mm
- GGMotion-DQPose: 41.8 mm

## Verification

```bash
cd code/ggmotion
make test-equivariance-dqpose
python experiments/verify_dqpose_preprocessing.py --manner 8
make robustness
```

## Training

```bash
cd code/ggmotion

# GGMotion-XYZ
python main_cmu.py --cfg cfg/cmu_short.yml --model xyz --exp exp/xyz_reproduced

# GGMotion-DQPose
python main_cmu.py --cfg cfg/cmu_dqpose_mindirect_short.yml --model dqpose --exp exp/dqposemindirect_retrained
```

The thesis uses these saved checkpoints:

| Model | Checkpoint |
|---|---|
| GGMotion-XYZ | `exp/xyz_reproduced/cmu_best_20.823_19.pt` |
| GGMotion-DQPose | `exp/dqposemindirect_retrained/cmu_best_24.595_22.pt` |

## Figures and Videos

The thesis qualitative figures and MP4 artifacts are generated from the final DQPose and XYZ checkpoints:

```bash
python experiments/visualize.py --actions walking --multi-row --robustness-figure --canonical-overlay
```

Full action-level artifacts are stored under:

```text
code/ggmotion/experiments/results/visualizations
```

For direct links to the MP4 files, see [supplementary/README.md](supplementary/README.md).

## Thesis PDF

The final thesis PDF is included at:

```text
thesis/main.pdf
```

## Defense Presentation

The defense materials are included at:

```text
presentation/Andreou_Chrysis_MSc_Defense.pptx
presentation/Andreou_Chrysis_MSc_Defense.pdf
```

Use the PowerPoint deck for live presentation because it contains embedded video clips. The PDF is a static preview for quick review.

## Contribution Boundary and Licenses

The original GGMotion baseline is inherited from `inkcat520/GGMotion`; its MIT license remains at [code/ggmotion/LICENSE](code/ggmotion/LICENSE). Original additions and modifications in this release are licensed under the repository root [LICENSE](LICENSE). The file [CONTRIBUTIONS.md](CONTRIBUTIONS.md) identifies inherited, modified, and newly authored material.
