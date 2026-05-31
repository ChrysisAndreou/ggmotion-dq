"""
Plot Exp 2 robustness results.

Usage (from ggmotion/code/):
    python experiments/plot_robustness.py
    python experiments/plot_robustness.py --out-dir experiments/results --horizon 400

Outputs:
    experiments/results/robustness_400ms.pdf
    experiments/results/robustness_400ms.png
"""

import argparse
import csv
from pathlib import Path


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_robustness(args):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available; printing tables only.')
        return

    out_dir = Path(args.out_dir)
    rot_csv   = out_dir / 'robustness_rotation.csv'
    trans_csv = out_dir / 'robustness_translation.csv'

    if not rot_csv.exists() or not trans_csv.exists():
        print(f'CSVs not found in {out_dir}. Run robustness_exp.py first.')
        return

    rot_rows   = load_csv(rot_csv)
    trans_rows = load_csv(trans_csv)
    h = args.horizon

    # Rotation plot
    angles = [float(r['angle_deg']) for r in rot_rows]
    dq_rot  = [float(r[f'dq_{h}'])  for r in rot_rows]
    xyz_rot = [float(r[f'xyz_{h}']) for r in rot_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(angles, dq_rot,  'b-o', markersize=4, linewidth=2,
            label='GGMotion-DQPose (SE(3)-equivariant)')
    ax.plot(angles, xyz_rot, 'r-s', markersize=4, linewidth=2,
            label='GGMotion-XYZ (Cartesian baseline)')
    ax.set_xlabel('Global rotation magnitude (°)', fontsize=12)
    ax.set_ylabel(f'MPJPE at {h}ms (mm)',          fontsize=12)
    ax.set_title('Rotation robustness',             fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 180)
    ax.set_xticks(range(0, 181, 30))
    ax.grid(True, alpha=0.3)

    # Translation plot
    mags  = [float(r['translation_m']) for r in trans_rows]
    dq_t  = [float(r[f'dq_{h}'])  for r in trans_rows]
    xyz_t = [float(r[f'xyz_{h}']) for r in trans_rows]

    ax = axes[1]
    ax.plot(mags, dq_t,  'b-o', markersize=4, linewidth=2,
            label='GGMotion-DQPose (SE(3)-equivariant)')
    ax.plot(mags, xyz_t, 'r-s', markersize=4, linewidth=2,
            label='GGMotion-XYZ (Cartesian baseline)')
    ax.set_xlabel('Global translation magnitude (m)', fontsize=12)
    ax.set_ylabel(f'MPJPE at {h}ms (mm)',             fontsize=12)
    ax.set_title('Translation robustness',             fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 5)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'Robustness under SE(3) perturbation - CMU 8-action benchmark',
        fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_pdf = out_dir / f'robustness_{h}ms.pdf'
    plt.savefig(str(out_pdf), bbox_inches='tight', dpi=300)
    print(f'Saved {out_pdf}')

    out_png = out_dir / f'robustness_{h}ms.png'
    plt.savefig(str(out_png), bbox_inches='tight', dpi=300)
    print(f'Saved {out_png}')
    plt.close()


def print_tables(args):
    out_dir = Path(args.out_dir)
    h = args.horizon

    rot_rows   = load_csv(out_dir / 'robustness_rotation.csv')
    trans_rows = load_csv(out_dir / 'robustness_translation.csv')

    print(f'\n=== Rotation robustness ({h}ms horizon) ===')
    print(f'{"Angle":>8} | {"GGMotion-DQPose":>18} | {"GGMotion-XYZ":>12} | {"XYZ/DQ":>8}')
    print('-' * 50)
    for r in rot_rows:
        dq  = float(r[f'dq_{h}'])
        xyz = float(r[f'xyz_{h}'])
        print(f'{r["angle_deg"]:>7}° | {dq:>18.1f} | {xyz:>12.1f} | {xyz/dq:>7.2f}x')

    print(f'\n=== Translation robustness ({h}ms horizon) ===')
    print(f'{"Transl.":>8} | {"GGMotion-DQPose":>18} | {"GGMotion-XYZ":>12} | {"XYZ/DQ":>8}')
    print('-' * 50)
    for r in trans_rows:
        dq  = float(r[f'dq_{h}'])
        xyz = float(r[f'xyz_{h}'])
        print(f'{float(r["translation_m"]):>6.1f} m | {dq:>18.1f} | {xyz:>12.1f} | {xyz/dq:>7.2f}x')


def main():
    parser = argparse.ArgumentParser(description='Plot Exp 2 robustness results')
    parser.add_argument('--out-dir',  default='experiments/results')
    parser.add_argument('--horizon',  type=int, default=400,
                        choices=[80, 160, 320, 400],
                        help='Which time horizon to plot (ms)')
    args = parser.parse_args()

    print_tables(args)
    plot_robustness(args)


if __name__ == '__main__':
    main()
