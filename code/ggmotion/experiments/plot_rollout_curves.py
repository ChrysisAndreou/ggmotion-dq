"""
Rollout MPJPE over training epochs at all four horizons.

Parses the saved logs for the final GGMotion-DQPose and GGMotion-XYZ
runs and generates a 2x2 convergence figure.

Usage (from ggmotion/code/):
    python experiments/plot_rollout_curves.py
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

HORIZONS = ['80ms', '160ms', '320ms', '400ms']
EPOCH_RE = re.compile(r'>>>\s+(?:train|eval)\s+Epoch:\s+(\d+)')
AVG_RE = re.compile(
    r'^avg\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)\s+\|'
)


def parse_log(path):
    result = {h: {'epochs': [], 'values': []} for h in HORIZONS}
    current_epoch = None

    with open(path, errors='ignore') as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                current_epoch = int(m.group(1))
                continue

            m = AVG_RE.search(line)
            if m and current_epoch is not None:
                values = [float(m.group(i)) for i in range(1, 5)]
                for h, value in zip(HORIZONS, values):
                    result[h]['epochs'].append(current_epoch)
                    result[h]['values'].append(value)

    return {
        h: (np.array(d['epochs']), np.array(d['values']))
        for h, d in result.items()
    }


def plot(data):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)

    for ax, horizon in zip(axes.flat, HORIZONS):
        for key, info in MODELS.items():
            epochs, values = data[key][horizon]
            max_epoch = info.get('max_epoch')
            if max_epoch is not None:
                mask = epochs <= max_epoch
                epochs, values = epochs[mask], values[mask]
            ax.plot(epochs, values, color=info['color'],
                    label=info['label'], linewidth=1.8,
                    marker='o', markersize=3)

        ax.set_title(f'Rollout {horizon}', fontsize=12, fontweight='bold')
        ax.set_ylabel('MPJPE (mm)', fontsize=10)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(axis='both', alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_xlabel('Epoch', fontsize=11)

    fig.suptitle('Rollout MPJPE Over Training', fontsize=13, fontweight='bold')
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ['pdf', 'png']:
        out_path = OUT_DIR / f'rollout_convergence.{ext}'
        fig.savefig(str(out_path), bbox_inches='tight', dpi=300)
        print(f'Saved -> {out_path}')
    plt.close(fig)


def main():
    data = {}
    for key, info in MODELS.items():
        if not info['path'].exists():
            raise FileNotFoundError(info['path'])
        data[key] = parse_log(info['path'])
        epochs, vals = data[key]['400ms']
        print(f"{info['label']}: {len(epochs)} eval points, final 400ms={vals[-1]:.1f} mm")
    plot(data)


if __name__ == '__main__':
    main()
