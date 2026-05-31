"""Per-action robustness eval for GGMotion-DQPose and GGMotion-XYZ.

Reuses the eval pipeline from robustness_exp.py but persists the per-action
breakdown (not just the macro-average) so Chapter 6 §6.3 can quote per-action
numbers under SE(3) frame change.

Output:
    experiments/results/robustness_per_action.csv

Schema (one row per (action, angle_deg)):
    action,angle_deg,dq_80,xyz_80,dq_160,xyz_160,dq_320,xyz_320,dq_400,xyz_400
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from utils.data_loader import CMU_Motion3D, CMU_Motion3DDQPose
from utils.data_utils import setup_seed
from module.model import GGMNet
from module.model_dqpose import GGMNet_DQPose

from experiments.robustness_exp import (
    EVAL_FRAME, axis_angle_to_rotmat, apply_rotation,
)


def eval_per_action(model, loaders, dim_used, scale, device, transform_fn):
    """Per-action MPJPE at each EVAL_FRAME horizon (mm)."""
    model.eval()
    actions = CMU_Motion3D.actions

    joint_to_ignore = np.array([16, 20, 29, 24, 27, 33, 36])
    index_to_ignore = np.concatenate(
        (joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([15, 15, 15, 23, 23, 32, 32])
    index_to_equal = np.concatenate(
        (joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

    per_action = {a: np.zeros(len(EVAL_FRAME)) for a in actions}
    per_action_count = {a: 0 for a in actions}

    with torch.no_grad():
        for act in actions:
            for input_seq, output_seq, all_seq in loaders[act]:
                batch_size, _, _, pred_len = output_seq.shape
                input_seq = input_seq.to(device)
                all_seq = all_seq.to(device)

                input_p, all_p = transform_fn(input_seq, all_seq)

                outputs = model(input_p)
                outputs = outputs.permute(0, 3, 1, 2).flatten(2)

                pred_3d = all_p[:, -pred_len:].clone()
                targ_p3d = all_p[:, -pred_len:].clone()
                pred_3d[:, :, dim_used] = outputs
                pred_3d[:, :, index_to_ignore] = pred_3d[:, :, index_to_equal]
                pred_p3d = pred_3d.contiguous().view(batch_size, pred_len, -1, 3)
                targ_p3d = targ_p3d.contiguous().view(batch_size, pred_len, -1, 3)

                for k in range(len(EVAL_FRAME)):
                    j = EVAL_FRAME[k][0]
                    mpjpe = torch.mean(torch.norm(
                        targ_p3d[:, j, :, :].contiguous().view(-1, 3) -
                        pred_p3d[:, j, :, :].contiguous().view(-1, 3),
                        2, 1)).item() * batch_size
                    per_action[act][k] += mpjpe
                per_action_count[act] += batch_size

    out = {}
    for act in actions:
        if per_action_count[act] > 0:
            out[act] = (per_action[act] * scale / per_action_count[act]).tolist()
    return out


def main():
    parser = argparse.ArgumentParser(description='Per-action robustness eval at selected rotation angles')
    parser.add_argument('--dq-checkpoint',
                        default='exp/dqposemindirect_retrained/cmu_best_24.595_22.pt')
    parser.add_argument('--xyz-checkpoint',
                        default='exp/xyz_reproduced/cmu_best_20.823_19.pt')
    parser.add_argument('--cfg', default='cfg/cmu_dqpose_mindirect_short.yml')
    parser.add_argument('--data', default='./data/cmu')
    parser.add_argument('--out-dir', default='experiments/results')
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--manner', default='all')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--angles', default='0,90',
                        help='Comma-separated list of rotation angles in degrees (default: 0,90)')
    parser.add_argument('--rotation-axis', default=None,
                        help='Override seeded axis sampling. Format: "ax,ay,az" (will be normalised). '
                             'Use the canonical CUDA axis "0.089,0.993,-0.079" when running on CPU '
                             'to keep the perturbation identical to the macro robustness CSVs.')
    args = parser.parse_args()
    angles = [int(a.strip()) for a in args.angles.split(',') if a.strip()]

    cfg_path = _ROOT / args.cfg
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    class Config:
        pass
    config = Config()
    for k, v in cfg.items():
        setattr(config, k, v)
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)
    print(f'Device: {device}', flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_seed(1234567890)

    print(f'\nLoading CMU test set from {args.data} ...')
    loaders_dq = {}
    loaders_xyz = {}
    dataset_ref = None
    dataset_dq_ref = None
    for act in CMU_Motion3D.define_acts('all'):
        dataset_xyz = CMU_Motion3D(
            args.data, actions=act,
            input_n=config.past_length, output_n=config.future_length,
            split=1, scale=config.scale, test_manner=args.manner,
            hop=getattr(config, 'n_hop', -1))
        dataset_dq = CMU_Motion3DDQPose(
            args.data, actions=act,
            input_n=config.past_length, output_n=config.future_length,
            split=1, scale=config.scale, test_manner=args.manner,
            hop=getattr(config, 'n_hop', -1))
        loaders_xyz[act] = DataLoader(
            dataset_xyz, batch_size=config.batch_size,
            shuffle=False, drop_last=False, num_workers=0, pin_memory=True)
        loaders_dq[act] = DataLoader(
            dataset_dq, batch_size=config.batch_size,
            shuffle=False, drop_last=False, num_workers=0, pin_memory=True)
        dataset_ref = dataset_xyz
        dataset_dq_ref = dataset_dq

    dim_used = dataset_ref.dim_used
    scale = config.scale

    print(f'\nLoading GGMotion-DQPose: {args.dq_checkpoint}')
    model_dq = GGMNet_DQPose(config, dataset_dq_ref.group, dataset_dq_ref.edges).to(device)
    ckpt_dq = torch.load(args.dq_checkpoint, map_location=device, weights_only=False)
    model_dq.load_state_dict(ckpt_dq['state_dict'])
    model_dq.eval()

    print(f'Loading GGMotion-XYZ: {args.xyz_checkpoint}')
    model_xyz = GGMNet(config, dataset_ref.group, dataset_ref.edges).to(device)
    ckpt_xyz = torch.load(args.xyz_checkpoint, map_location=device, weights_only=False)
    model_xyz.load_state_dict(ckpt_xyz['state_dict'])
    model_xyz.eval()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.rotation_axis is not None:
        ax_vals = [float(v.strip()) for v in args.rotation_axis.split(',')]
        assert len(ax_vals) == 3, f'--rotation-axis needs 3 comma-separated values, got {ax_vals}'
        axis = torch.tensor(ax_vals, device=device, dtype=torch.float32)
        axis = axis / axis.norm()
        print(f'\nRotation axis (override): [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]')
    else:
        axis = torch.randn(3, device=device)
        axis = axis / axis.norm()
        print(f'\nRotation axis (seed={args.seed}): [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]')

    rows = []
    for deg in angles:
        angle_rad = torch.tensor(deg * np.pi / 180.0, device=device, dtype=torch.float32)
        R = axis_angle_to_rotmat(axis, angle_rad)

        def transform_fn(inp, all_s, _R=R):
            return apply_rotation(_R, inp, all_s)

        print(f'\n=== Angle {deg}° ===')
        print('  GGMotion-DQPose ...')
        per_dq = eval_per_action(model_dq, loaders_dq, dim_used, scale, device, transform_fn)
        print('  GGMotion-XYZ ...')
        per_xyz = eval_per_action(model_xyz, loaders_xyz, dim_used, scale, device, transform_fn)

        for act in CMU_Motion3D.actions:
            d = per_dq[act]; x = per_xyz[act]
            rows.append({
                'action': act, 'angle_deg': deg,
                'dq_80': d[0], 'xyz_80': x[0],
                'dq_160': d[1], 'xyz_160': x[1],
                'dq_320': d[2], 'xyz_320': x[2],
                'dq_400': d[3], 'xyz_400': x[3],
            })
            print(f'  {act:20s} | DQPose {d[3]:6.2f} | XYZ {x[3]:6.2f}')

    out_csv = out_dir / 'robustness_per_action.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f'\nSaved {out_csv}')


if __name__ == '__main__':
    main()
