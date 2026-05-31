"""
Per-action robustness bar chart.

Plots a grouped bar chart comparing DQPose vs XYZ at 0 deg and a
non-zero anchor angle (default 90 deg, near the macro peak at 100 deg)
per action.

Usage (from ggmotion/code/):
    # Plot from existing CSV
    python experiments/plot_per_action_robustness.py --plot-only

    # Full run: compute + plot
    python experiments/plot_per_action_robustness.py

Outputs:
    experiments/results/robustness_per_action.csv   (from compute step)
    experiments/results/robustness_per_action.pdf
    experiments/results/robustness_per_action.png
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

ACTIONS_8 = [
    'basketball',
    'basketball_signal',
    'directing_traffic',
    'jumping',
    'running',
    'soccer',
    'walking',
    'washwindow',
]


def compute_per_action(args):
    """Run the DQPose/XYZ per-action robustness evaluation, then save CSV."""
    script = Path(__file__).with_name('eval_per_action_robustness.py')
    cmd = [
        sys.executable, str(script),
        '--out-dir', str(args.out_dir),
        '--angles', f'0,{args.anchor_angle}',
    ]
    subprocess.run(cmd, check=True)


def plot_bar_chart(csv_path, out_dir, horizon=400, anchor_angle=90):
    """Generate grouped bar chart from per-action CSV.

    anchor_angle: non-zero rotation angle (deg) to compare against 0°.
        Defaults to 90°, which sits next to the macro peak at 100° and
        therefore captures near-worst-case XYZ degradation per action.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    h = horizon
    a = anchor_angle
    actions = []
    dq_0, dq_a, xyz_0, xyz_a = [], [], [], []

    for action in ACTIONS_8:
        display = action.replace('_', ' ')
        actions.append(display)

        for row in rows:
            if row['action'] == action and int(row['angle_deg']) == 0:
                dq_0.append(float(row[f'dq_{h}']))
                xyz_0.append(float(row[f'xyz_{h}']))
            elif row['action'] == action and int(row['angle_deg']) == a:
                dq_a.append(float(row[f'dq_{h}']))
                xyz_a.append(float(row[f'xyz_{h}']))

    n = len(actions)
    x = np.arange(n)
    w = 0.20

    fig, ax = plt.subplots(figsize=(10, 4.8))

    ax.bar(x - 1.5*w, dq_0,  w, label='GGMotion-DQPose 0°',     color='#2166ac')
    ax.bar(x - 0.5*w, dq_a,  w, label=f'GGMotion-DQPose {a}°',  color='#92c5de')
    ax.bar(x + 0.5*w, xyz_0, w, label='GGMotion-XYZ 0°',        color='#b2182b')
    ax.bar(x + 1.5*w, xyz_a, w, label=f'GGMotion-XYZ {a}°',     color='#f4a582')

    ax.set_ylabel(f'MPJPE at {h}ms (mm)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(actions, fontsize=10, rotation=25, ha='right')
    ax.legend(fontsize=9, ncol=2, loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    for i, action in enumerate(ACTIONS_8):
        if action == 'directing_traffic' and len(xyz_a) > i and len(xyz_0) > i:
            if xyz_a[i] < xyz_0[i]:
                ax.annotate('',
                    xy=(x[i] + 1.5*w, xyz_a[i] - 0.5),
                    xytext=(x[i] + 1.5*w, xyz_0[i] + 0.5),
                    arrowprops=dict(arrowstyle='->', color='#b2182b',
                                   lw=1.5, shrinkA=0, shrinkB=0))

    plt.tight_layout()

    out_dir = Path(out_dir)
    for ext in ['pdf', 'png']:
        out_path = out_dir / f'robustness_per_action.{ext}'
        fig.savefig(str(out_path), bbox_inches='tight', dpi=300)
        print(f'Saved -> {out_path}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Per-action robustness bar chart (0 deg vs anchor angle)')
    parser.add_argument('--out-dir', default='experiments/results')
    parser.add_argument('--horizon', type=int, default=400,
                        choices=[80, 160, 320, 400])
    parser.add_argument('--anchor-angle', type=int, default=90,
                        help='Non-zero rotation angle for comparison (default 90, near macro peak at 100)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip computation, plot from existing CSV')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    csv_path = out_dir / 'robustness_per_action.csv'

    if not args.plot_only:
        compute_per_action(args)
    elif not csv_path.exists():
        print(f'CSV not found at {csv_path}. Run without --plot-only first.')
        return

    plot_bar_chart(csv_path, args.out_dir, args.horizon, args.anchor_angle)


if __name__ == '__main__':
    main()
