"""
Training convergence curves for the final thesis models.

Parses the saved training/evaluation logs for GGMotion-DQPose and
GGMotion-XYZ and generates the thesis convergence figure.

Usage (from ggmotion/code/):
    python experiments/plot_training_curves.py

Outputs:
    experiments/results/training_convergence.pdf
    experiments/results/training_convergence.png
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

MODELS = {
    'dqpose': {
        'path': ROOT / 'experiments' / 'results' / 'dqposemindirect_train.txt',
        'label': 'GGMotion-DQPose',
        'color': '#2166ac',
        'max_epoch': 23,
    },
    'xyz': {
        'path': ROOT / 'exp' / 'xyz_reproduced' / 'cmu_eval.log',
        'label': 'GGMotion-XYZ',
        'color': '#b2182b',
        'max_epoch': 21,
    },
}

EPOCH_RE = re.compile(
    r'>>>\s+(?:train|eval)\s+Epoch:\s+(\d+).*?Pri_loss:\s+([0-9.]+),?\s+Aux_loss:\s+([0-9.]+)'
)
AVG_RE = re.compile(
    r'^avg\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)\s+\|'
)


def parse_log(path):
    train_epochs, pri_loss = [], []
    eval_epochs, eval_mean = [], []
    current_epoch = None

    with open(path, errors='ignore') as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                current_epoch = int(m.group(1))
                train_epochs.append(current_epoch)
                pri_loss.append(float(m.group(2)))
                continue

            m = AVG_RE.search(line)
            if m and current_epoch is not None:
                eval_epochs.append(current_epoch)
                eval_mean.append(float(m.group(5)))

    return {
        'train_epochs': np.array(train_epochs),
        'pri_loss': np.array(pri_loss),
        'eval_epochs': np.array(eval_epochs),
        'eval_mean': np.array(eval_mean),
    }


def plot(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for key, info in MODELS.items():
        d = data[key]
        max_epoch = info.get('max_epoch')
        if max_epoch is not None:
            tmask = d['train_epochs'] <= max_epoch
            emask = d['eval_epochs'] <= max_epoch
        else:
            tmask = np.ones_like(d['train_epochs'], dtype=bool)
            emask = np.ones_like(d['eval_epochs'], dtype=bool)
        ax1.plot(d['train_epochs'][tmask], d['pri_loss'][tmask],
                 color=info['color'], label=info['label'], linewidth=1.8)
        ax2.plot(d['eval_epochs'][emask], d['eval_mean'][emask],
                 color=info['color'], label=info['label'], linewidth=1.8)

    for ax, title, ylabel in [
        (ax1, 'Training Loss', 'Primary loss'),
        (ax2, 'Validation MPJPE', 'Mean MPJPE (mm)'),
    ]:
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(axis='both', alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax1.set_yscale('log')
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ['pdf', 'png']:
        out_path = OUT_DIR / f'training_convergence.{ext}'
        fig.savefig(str(out_path), bbox_inches='tight', dpi=300)
        print(f'Saved -> {out_path}')
    plt.close(fig)


def main():
    data = {}
    for key, info in MODELS.items():
        if not info['path'].exists():
            raise FileNotFoundError(info['path'])
        data[key] = parse_log(info['path'])
        print(f"{info['label']}: {len(data[key]['train_epochs'])} loss points, "
              f"{len(data[key]['eval_epochs'])} eval points")
    plot(data)


if __name__ == '__main__':
    main()
