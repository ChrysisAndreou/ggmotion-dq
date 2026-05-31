#!/usr/bin/env python
"""
Exp 2: Robustness under SE(3) perturbation (GGMotion).

Apply global SE(3) transforms of increasing magnitude to test inputs and GT,
then measure MPJPE for GGMotion-DQPose and GGMotion-XYZ.

Expected:
  GGMotion-DQPose: flat curve (reference-pose-relative DQ input cancels global g)
  GGMotion-XYZ: rising curve (no equivariance; trained on one canonical frame)

Rotation sweep  : 0° to 180° in 10° steps, fixed random axis
Translation sweep: 0m to 5m in 0.5m steps, fixed random direction

Outputs:
  experiments/results/robustness_rotation.csv
  experiments/results/robustness_translation.csv
  (columns: magnitude, dq_80, dq_160, dq_320, dq_400, xyz_80, xyz_160, xyz_320, xyz_400)

Usage (from ggmotion/code/):
    python experiments/robustness_exp.py

    python experiments/robustness_exp.py \
        --cfg cfg/cmu_dqpose_mindirect_short.yml \
        --dq-checkpoint  exp/dqposemindirect_retrained/cmu_best_24.595_22.pt \
        --xyz-checkpoint exp/xyz_reproduced/cmu_best_20.823_19.pt
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_HERE = Path(__file__).resolve().parent        # ggmotion/code/experiments/
_ROOT = _HERE.parent                           # ggmotion/code/
sys.path.insert(0, str(_ROOT))

from utils.data_loader import CMU_Motion3D, CMU_Motion3DDQPose
from utils.data_utils import setup_seed
from torch.utils.data import DataLoader
from module.model import GGMNet
from module.model_dqpose import GGMNet_DQPose
from module.dq_ops import pose_to_dq, pos_to_dq, dq_mul

EVAL_FRAME = [(1, 80), (3, 160), (7, 320), (9, 400)]
HORIZONS_MS = [80, 160, 320, 400]


# SE(3) transform helpers

def axis_angle_to_rotmat(axis, angle_rad):
    """
    Rodrigues: axis (unit, [3]) + angle (scalar) to rotation matrix [3,3].
    """
    K = torch.zeros(3, 3, device=axis.device, dtype=axis.dtype)
    K[0, 1] = -axis[2]; K[0, 2] =  axis[1]
    K[1, 0] =  axis[2]; K[1, 2] = -axis[0]
    K[2, 0] = -axis[1]; K[2, 1] =  axis[0]
    I = torch.eye(3, device=axis.device, dtype=axis.dtype)
    R = I + torch.sin(angle_rad) * K + (1 - torch.cos(angle_rad)) * (K @ K)
    return R


def apply_rotation(R, input_seq, all_seq, local_root=None):
    """
    Apply rotation R to both model input and full skeleton sequence.

    Args:
        R:         [3, 3] rotation matrix
        input_seq: [B, N, 3, T] - model input (25 joints)
        all_seq:   [B, T, D] - full skeleton (D = 38*3 = 114)
    Returns:
        input_rot: [B, N, 3, T]
        all_rot:   [B, T, D]
    """
    if input_seq.shape[2] == 3:
        input_rot = torch.einsum('ij,bnjt->bnit', R, input_seq)
    elif input_seq.shape[2] == 8:
        g = pose_to_dq(R.view(1, 3, 3), torch.zeros(1, 3, device=R.device, dtype=R.dtype)).view(8)
        input_p = input_seq.permute(0, 1, 3, 2)
        if local_root is None:
            input_rot = dq_mul(g.view(1, 1, 1, 8).expand_as(input_p), input_p)
        else:
            input_rot = input_p.clone()
            root_slice = input_rot[:, local_root:local_root + 1, :, :]
            input_rot[:, local_root:local_root + 1, :, :] = dq_mul(
                g.view(1, 1, 1, 8).expand_as(root_slice), root_slice)
        input_rot = input_rot.permute(0, 1, 3, 2)
    else:
        raise ValueError(f"Unsupported input channels for rotation: {input_seq.shape[2]}")

    # all_seq: reshape to (B, T, J, 3), rotate, reshape back
    B, T, D = all_seq.shape
    all_rot = all_seq.view(B, T, -1, 3) @ R.T
    all_rot = all_rot.view(B, T, D)

    return input_rot, all_rot


def apply_translation(t, input_seq, all_seq, local_root=None):
    """
    Apply translation t to both model input and full skeleton sequence.

    Args:
        t:         [3] translation vector (in model units, i.e. meters if scale=1000)
        input_seq: [B, N, 3, T]
        all_seq:   [B, T, D]
    Returns:
        input_tr:  [B, N, 3, T]
        all_tr:    [B, T, D]
    """
    if input_seq.shape[2] == 3:
        input_tr = input_seq + t.view(1, 1, 3, 1)
    elif input_seq.shape[2] == 8:
        g = pos_to_dq(t.view(1, 3)).view(8)
        input_p = input_seq.permute(0, 1, 3, 2)
        if local_root is None:
            input_tr = dq_mul(g.view(1, 1, 1, 8).expand_as(input_p), input_p)
        else:
            input_tr = input_p.clone()
            root_slice = input_tr[:, local_root:local_root + 1, :, :]
            input_tr[:, local_root:local_root + 1, :, :] = dq_mul(
                g.view(1, 1, 1, 8).expand_as(root_slice), root_slice)
        input_tr = input_tr.permute(0, 1, 3, 2)
    else:
        raise ValueError(f"Unsupported input channels for translation: {input_seq.shape[2]}")

    B, T, D = all_seq.shape
    all_tr = all_seq.view(B, T, -1, 3) + t.view(1, 1, 1, 3)
    all_tr = all_tr.view(B, T, D)

    return input_tr, all_tr


# Core: evaluate one model at one perturbation level

@torch.no_grad()
def eval_perturbed(model, loaders, dim_used, scale, device, transform_fn):
    """
    Run GGMotion test_cmu logic with a perturbation applied to inputs and GT.

    Args:
        model:        GGMNet or GGMNet_DQ
        loaders:      dict {action: DataLoader}
        dim_used:     list of joint indices
        scale:        scaling factor (1000 for CMU)
        device:       torch device
        transform_fn: callable(input_seq, all_seq) -> (input_perturbed, all_perturbed)

    Returns:
        dict {horizon_ms: macro_averaged_mpjpe_mm}
    """
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

    for act in actions:
        for batch_data in loaders[act]:
            input_seq, output_seq, all_seq = batch_data[:3]
            batch_size, _, _, pred_len = output_seq.shape
            input_seq = input_seq.to(device)
            all_seq = all_seq.to(device)

            # Apply perturbation
            input_p, all_p = transform_fn(input_seq, all_seq)

            # Forward pass
            outputs = model(input_p)                                 # (B, N, 3, T)
            outputs = outputs.permute(0, 3, 1, 2).flatten(2)        # (B, T, N*3)

            # Reconstruct full skeleton (same as test_cmu in runtime.py)
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

    # Macro-average: mean of per-action means (consistent with official eval)
    result = {}
    for k, (_, ms) in enumerate(EVAL_FRAME):
        action_means = []
        for act in actions:
            if per_action_count[act] > 0:
                action_means.append(per_action[act][k] * scale / per_action_count[act])
        result[ms] = float(np.mean(action_means))
    return result


# Sweep functions

def rotation_sweep(model_dq, model_xyz, loaders_dq, loaders_xyz, dim_used, scale, device,
                   seed=42, angles=None, dq_local_root=None):
    """Sweep global rotation magnitude from 0° to 180° in 10° steps."""
    if angles is None:
        angles = range(0, 181, 10)
    torch.manual_seed(seed)
    np.random.seed(seed)
    axis = torch.randn(3, device=device)
    axis = axis / axis.norm()
    print(f'  Rotation axis: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]')

    rows = []
    for deg in angles:
        angle_rad = torch.tensor(deg * np.pi / 180.0, device=device, dtype=torch.float32)
        R = axis_angle_to_rotmat(axis, angle_rad)

        def transform_fn(inp, all_s, _R=R):
            return apply_rotation(_R, inp, all_s, local_root=dq_local_root)

        dq_e = eval_perturbed(model_dq, loaders_dq, dim_used, scale, device, transform_fn)
        xyz_e = eval_perturbed(model_xyz, loaders_xyz, dim_used, scale, device, transform_fn)

        row = {
            'angle_deg': deg,
            'dq_80': dq_e[80], 'dq_160': dq_e[160],
            'dq_320': dq_e[320], 'dq_400': dq_e[400],
            'xyz_80': xyz_e[80], 'xyz_160': xyz_e[160],
            'xyz_320': xyz_e[320], 'xyz_400': xyz_e[400],
        }
        rows.append(row)
        print(f'  {deg:3d}° | DQ  {dq_e[80]:.1f}/{dq_e[160]:.1f}/{dq_e[320]:.1f}/{dq_e[400]:.1f} mm'
              f'  | XYZ {xyz_e[80]:.1f}/{xyz_e[160]:.1f}/{xyz_e[320]:.1f}/{xyz_e[400]:.1f} mm')
    return rows


def translation_sweep(model_dq, model_xyz, loaders_dq, loaders_xyz, dim_used, scale, device,
                      seed=99, translations=None, dq_local_root=None):
    """Sweep global translation magnitude from 0m to 5m in 0.5m steps."""
    if translations is None:
        translations = [i * 0.5 for i in range(11)]
    torch.manual_seed(seed)
    np.random.seed(seed)
    direction = torch.randn(3, device=device)
    direction = direction / direction.norm()
    print(f'  Translation direction: [{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}]')

    rows = []
    for mag_m in translations:
        # Data is in original_mm / scale. With scale=1000, data is in meters.
        # Translation of mag_m meters = mag_m in data units.
        t = direction * mag_m
        t = t.to(device)

        def transform_fn(inp, all_s, _t=t):
            return apply_translation(_t, inp, all_s, local_root=dq_local_root)

        dq_e = eval_perturbed(model_dq, loaders_dq, dim_used, scale, device, transform_fn)
        xyz_e = eval_perturbed(model_xyz, loaders_xyz, dim_used, scale, device, transform_fn)

        row = {
            'translation_m': mag_m,
            'dq_80': dq_e[80], 'dq_160': dq_e[160],
            'dq_320': dq_e[320], 'dq_400': dq_e[400],
            'xyz_80': xyz_e[80], 'xyz_160': xyz_e[160],
            'xyz_320': xyz_e[320], 'xyz_400': xyz_e[400],
        }
        rows.append(row)
        print(f'  {mag_m:.1f}m | DQ  {dq_e[80]:.1f}/{dq_e[160]:.1f}/{dq_e[320]:.1f}/{dq_e[400]:.1f} mm'
              f'  | XYZ {xyz_e[80]:.1f}/{xyz_e[160]:.1f}/{xyz_e[320]:.1f}/{xyz_e[400]:.1f} mm')
    return rows


# Main

def main():
    parser = argparse.ArgumentParser(
        description='Robustness Exp 2: MPJPE vs SE(3) perturbation magnitude (GGMotion)')
    parser.add_argument('--dq-checkpoint',
                        default='exp/dqposemindirect_retrained/cmu_best_24.595_22.pt')
    parser.add_argument('--xyz-checkpoint',
                        default='exp/xyz_reproduced/cmu_best_20.823_19.pt')
    parser.add_argument('--cfg', default='cfg/cmu_dqpose_mindirect_short.yml',
                        help='Path to config YAML')
    parser.add_argument('--data', default='./data/cmu',
                        help='Path to CMU data directory')
    parser.add_argument('--out-dir', default='experiments/results',
                        help='Output directory for CSV results')
    parser.add_argument('--device', default=None,
                        help='Device (auto-detect if not set)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for rotation axis (translation uses seed+57)')
    parser.add_argument('--manner', default='all',
                        help='Test sampling: all, 256, or 8')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size (default: from config)')
    parser.add_argument('--angles', default=None,
                        help='Comma-separated rotation angles in degrees (default: 0,10,...,180)')
    parser.add_argument('--translations', default=None,
                        help='Comma-separated translations in meters (default: 0,0.5,...,5)')
    args = parser.parse_args()
    angles = [int(x) for x in args.angles.split(',')] if args.angles else None
    translations = [float(x) for x in args.translations.split(',')] if args.translations else None

    # Load config
    cfg_path = _ROOT / args.cfg
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Merge config into a namespace for model construction
    class Config:
        pass
    config = Config()
    for k, v in cfg.items():
        setattr(config, k, v)
    # Override with CLI args where applicable
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)
    print(f'Device: {device}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_seed(1234567890)  # Same seed as training for data consistency

    # Load test data
    print(f'\nLoading CMU test set from {args.data} ...')
    loaders_dq = {}
    loaders_xyz = {}
    len_test = 0
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
        len_test += len(dataset_xyz)
        dataset_ref = dataset_xyz
        dataset_dq_ref = dataset_dq
    print(f'  {len_test} test samples')

    dim_used = dataset_ref.dim_used
    scale = config.scale

    # Load models
    print(f'\nLoading GGMotion-DQPose: {args.dq_checkpoint}')
    model_dq = GGMNet_DQPose(config, dataset_dq_ref.group, dataset_dq_ref.edges).to(device)
    ckpt_dq = torch.load(args.dq_checkpoint, map_location=device, weights_only=False)
    model_dq.load_state_dict(ckpt_dq['state_dict'])
    model_dq.eval()
    n_dq = sum(p.numel() for p in model_dq.parameters())
    print(f'  {n_dq:,} params | epoch {ckpt_dq["epoch"]}')

    print(f'\nLoading GGMotion-XYZ: {args.xyz_checkpoint}')
    class XYZConfig:
        pass
    xyz_config = XYZConfig()
    for k, v in vars(config).items():
        setattr(xyz_config, k, v)
    xyz_config.key_point = dataset_ref.used_seqs.shape[1]
    model_xyz = GGMNet(xyz_config, dataset_ref.group, dataset_ref.edges).to(device)
    ckpt_xyz = torch.load(args.xyz_checkpoint, map_location=device, weights_only=False)
    model_xyz.load_state_dict(ckpt_xyz['state_dict'])
    model_xyz.eval()
    n_xyz = sum(p.numel() for p in model_xyz.parameters())
    print(f'  {n_xyz:,} params | epoch {ckpt_xyz["epoch"]}')

    # Rotation sweep
    print('\n=== Rotation sweep: 0° to 180° in 10° steps ===')
    rot_rows = rotation_sweep(
        model_dq, model_xyz, loaders_dq, loaders_xyz, dim_used, scale, device,
        seed=args.seed, angles=angles, dq_local_root=None)

    rot_csv = out_dir / 'robustness_rotation.csv'
    with open(rot_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rot_rows[0].keys())
        w.writeheader()
        w.writerows(rot_rows)
    print(f'\n  Saved {rot_csv}')

    # Translation sweep
    print('\n=== Translation sweep: 0m to 5m in 0.5m steps ===')
    trans_rows = translation_sweep(
        model_dq, model_xyz, loaders_dq, loaders_xyz, dim_used, scale, device,
        seed=args.seed + 57, translations=translations, dq_local_root=None)

    trans_csv = out_dir / 'robustness_translation.csv'
    with open(trans_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=trans_rows[0].keys())
        w.writeheader()
        w.writerows(trans_rows)
    print(f'\n  Saved {trans_csv}')

    # Summary
    print('\n=== Summary (400ms horizon) ===')
    print(f'{"Rotation":>10} | {"DQ 400ms":>10} | {"XYZ 400ms":>12} | {"Ratio":>8}')
    print('-' * 48)
    for row in rot_rows:
        ratio = row['xyz_400'] / row['dq_400'] if row['dq_400'] > 0 else float('nan')
        print(f'{row["angle_deg"]:>9}° | {row["dq_400"]:>10.1f} | {row["xyz_400"]:>12.1f} | {ratio:>7.2f}x')

    print()
    print(f'{"Translation":>12} | {"DQ 400ms":>10} | {"XYZ 400ms":>12} | {"Ratio":>8}')
    print('-' * 50)
    for row in trans_rows:
        ratio = row['xyz_400'] / row['dq_400'] if row['dq_400'] > 0 else float('nan')
        print(f'{row["translation_m"]:>10.1f}m | {row["dq_400"]:>10.1f} | {row["xyz_400"]:>12.1f} | {ratio:>7.2f}x')

    print(f'\nResults saved to {out_dir}/')


if __name__ == '__main__':
    main()
