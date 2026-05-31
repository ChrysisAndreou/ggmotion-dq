"""
Loss components over training for the final thesis models.

Parses saved logs and plots primary and auxiliary losses. For DQPose, the
logged pose-DQ residual is included when available.

Usage (from ggmotion/code/):
    python experiments/plot_loss_components.py
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
OUT_DIR = _HERE / 'results'

DQPOSE_LOG = ROOT / 'experiments' / 'results' / 'dqposemindirect_train.txt'
XYZ_LOG = ROOT / 'exp' / 'xyz_reproduced' / 'cmu_eval.log'

LOSS_RE = re.compile(
    r'>>>\s+(?:train|eval)\s+Epoch:\s+(\d+).*?Pri_loss:\s+([0-9.]+),?\s+Aux_loss:\s+([0-9.]+)(?:,?\s+Pose_loss:\s+([0-9.]+))?'
)


def load_losses(path):
    epochs, pri_loss, aux_loss, pose_loss = [], [], [], []
    with open(path, errors='ignore') as f:
        for line in f:
            m = LOSS_RE.search(line)
            if not m:
                continue
            epochs.append(int(m.group(1)))
            pri_loss.append(float(m.group(2)))
            aux_loss.append(float(m.group(3)))
            pose_loss.append(float(m.group(4)) if m.group(4) is not None else np.nan)
    return {
        'epochs': np.array(epochs),
        'pri': np.array(pri_loss),
        'aux': np.array(aux_loss),
        'pose': np.array(pose_loss),
    }


def plot(dqpose, xyz):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    ax1.plot(dqpose['epochs'], dqpose['pri'], color='#b2182b',
             linewidth=2.0, label=f"Primary ({dqpose['pri'][-1]:.4f})")
    ax1.plot(dqpose['epochs'], dqpose['aux'], color='#2166ac',
             linewidth=2.0, linestyle='--',
             label=f"Bone auxiliary ({dqpose['aux'][-1]:.4f})")
    pose_mask = np.isfinite(dqpose['pose']) & (dqpose['pose'] > 0)
    if pose_mask.any():
        ax1.plot(dqpose['epochs'][pose_mask], dqpose['pose'][pose_mask],
                 color='#4d9221', linewidth=2.0, linestyle=':',
                 label=f"DQ pose residual ({dqpose['pose'][pose_mask][-1]:.4f})")
    ax1.set_title('GGMotion-DQPose', fontsize=12, fontweight='bold')

    ax2.plot(xyz['epochs'], xyz['pri'], color='#b2182b',
             linewidth=2.0, label=f"Primary ({xyz['pri'][-1]:.4f})")
    ax2.plot(xyz['epochs'], xyz['aux'], color='#2166ac',
             linewidth=2.0, linestyle='--',
             label=f"Bone auxiliary ({xyz['aux'][-1]:.4f})")
    ax2.set_title('GGMotion-XYZ', fontsize=12, fontweight='bold')

    for ax in [ax1, ax2]:
        ax.set_xlabel('Epoch', fontsize=11)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(axis='both', alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yscale('log')

    ax1.set_ylabel('Loss', fontsize=11)
    fig.suptitle('Training Loss Components', fontsize=13, fontweight='bold')
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ['pdf', 'png']:
        out_path = OUT_DIR / f'loss_components.{ext}'
        fig.savefig(str(out_path), bbox_inches='tight', dpi=300)
        print(f'Saved -> {out_path}')
    plt.close(fig)


def main():
    dqpose = load_losses(DQPOSE_LOG)
    xyz = load_losses(XYZ_LOG)
    print(f"GGMotion-DQPose: {len(dqpose['epochs'])} loss points")
    print(f"GGMotion-XYZ: {len(xyz['epochs'])} loss points")
    plot(dqpose, xyz)


if __name__ == '__main__':
    main()
