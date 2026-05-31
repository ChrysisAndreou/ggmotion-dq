#!/usr/bin/env python
"""
Qualitative validation: 3D skeleton figures for GGMotion thesis.

Renders ground truth overlaid alongside DQPose and XYZ predictions as static
multi-row figures and canonical-frame overlay grids. Supports SE(3)
perturbation to visually confirm robustness claims.

Three figure types:
  1. Multi-row accuracy figure (GT + DQPose + XYZ across time horizons)
  2. Canonical-frame overlay grid (equivariance test: 0° vs 90°/180°)
  3. Per-model robustness figure (0°/90°/180° rows × time columns)

Usage (from ggmotion/code/):
    # Find representative windows (no visualization)
    python experiments/visualize.py --actions basketball soccer running --find-samples

    # Accuracy multi-row figure (GT overlaid, per-column MPJPE)
    python experiments/visualize.py --actions walking --multi-row

    # Canonical-frame overlay grid (primary robustness viz)
    python experiments/visualize.py --actions running --canonical-overlay

    # Per-model robustness figure (0°/90°/180° in rotated frames)
    python experiments/visualize.py --actions running --robustness-figure

    # MP4 animation at 0 degrees
    python experiments/visualize.py --actions basketball --rotation 0

    # Override auto-sample with manual window
    python experiments/visualize.py --actions basketball --sample-idx 5 --multi-row
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

_HERE = Path(__file__).resolve().parent        # ggmotion/code/experiments/
_ROOT = _HERE.parent                           # ggmotion/code/
sys.path.insert(0, str(_ROOT))

from utils.data_loader import CMU_Motion3D, CMU_Motion3DDQPose
from utils.data_utils import setup_seed
from torch.utils.data import DataLoader
from module.model import GGMNet
from module.model_dqpose import GGMNet_DQPose
from experiments.robustness_exp import (
    axis_angle_to_rotmat,
    EVAL_FRAME,
    apply_rotation,
)

FFMPEG_PATH = '/opt/homebrew/bin/ffmpeg'

ACTIONS_8 = CMU_Motion3D.actions


# Skeleton drawing

def get_bone_lines(parent):
    """Return list of (child, parent) index pairs for drawing bones."""
    bones = []
    for row in parent:
        c, p = int(row[0]), int(row[1])
        bones.append((c, p))
    return bones


def draw_skeleton(ax, pos, bones, color='black', alpha=1.0, linewidth=1.5,
                  joint_size=8, label=None):
    """Draw a single skeleton frame on a 3D axis. pos: [N_joints, 3] in mm."""
    artists = []
    for i, (c, p) in enumerate(bones):
        if c >= pos.shape[0] or p >= pos.shape[0]:
            continue
        line, = ax.plot(
            [pos[c, 0], pos[p, 0]],
            [pos[c, 2], pos[p, 2]],  # Z to matplotlib Y (depth)
            [pos[c, 1], pos[p, 1]],  # Y to matplotlib Z (up)
            color=color, alpha=alpha, linewidth=linewidth,
            label=label if i == 0 else None
        )
        artists.append(line)
    scatter = ax.scatter(
        pos[:, 0], pos[:, 2], pos[:, 1],
        c=color, s=joint_size, alpha=alpha, zorder=5
    )
    artists.append(scatter)
    return artists


def draw_skeleton_dashed(ax, pos, bones, color='black', alpha=0.85,
                         linewidth=2.0, joint_size=8, label=None):
    """Draw a skeleton with dashed lines (for predictions in overlay)."""
    artists = []
    for i, (c, p) in enumerate(bones):
        if c >= pos.shape[0] or p >= pos.shape[0]:
            continue
        line, = ax.plot(
            [pos[c, 0], pos[p, 0]],
            [pos[c, 2], pos[p, 2]],
            [pos[c, 1], pos[p, 1]],
            color=color, alpha=alpha, linewidth=linewidth,
            linestyle='--', dashes=(4, 3),
            label=label if i == 0 else None
        )
        artists.append(line)
    scatter = ax.scatter(
        pos[:, 0], pos[:, 2], pos[:, 1],
        c=color, s=joint_size, alpha=alpha, zorder=5,
        marker='x'
    )
    artists.append(scatter)
    return artists


def draw_skeleton_dotted(ax, pos, bones, color='black', alpha=0.85,
                         linewidth=2.0, joint_size=8, label=None):
    """Draw a skeleton with dash-dot lines (for 180°-inv overlay)."""
    artists = []
    for i, (c, p) in enumerate(bones):
        if c >= pos.shape[0] or p >= pos.shape[0]:
            continue
        line, = ax.plot(
            [pos[c, 0], pos[p, 0]],
            [pos[c, 2], pos[p, 2]],
            [pos[c, 1], pos[p, 1]],
            color=color, alpha=alpha, linewidth=linewidth,
            linestyle='--', dashes=(8, 3, 3, 3),
            label=label if i == 0 else None
        )
        artists.append(line)
    scatter = ax.scatter(
        pos[:, 0], pos[:, 2], pos[:, 1],
        c=color, s=joint_size, alpha=alpha, zorder=5,
        marker='+'
    )
    artists.append(scatter)
    return artists


# Model loading (GGMotion)

def load_config(cfg_path):
    """Load YAML config and return a namespace object."""
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    class Config:
        pass
    config = Config()
    for k, v in cfg.items():
        setattr(config, k, v)
    return config


def load_ggmotion_model(ckpt_path, model_type, config, group, edges, device):
    """Load a GGMotion model from checkpoint."""
    if model_type == 'dq':
        model = GGMNet_DQPose(config, group, edges).to(device)
    else:
        model = GGMNet(config, group, edges).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    label = 'DQPose' if model_type == 'dq' else model_type.upper()
    print(f'  {label}: {n_params:,} params | epoch {ckpt["epoch"]}')
    return model


# Prediction (GGMotion single-shot, 25-joint)

def _rotate_position_sequence(seq, rotation, rotation_deg):
    if rotation_deg == 0:
        return seq
    return torch.einsum('ij,bnjt->bnit', rotation, seq)


@torch.no_grad()
def predict_with_rotation(models, input_xyz, input_dq, output_pos, scale,
                          device, rotation_deg=0, seed=42):
    """
    Run forward pass for both models, optionally with rotation perturbation.

    Uses XYZ inputs for the Cartesian baseline and pose-DQ inputs for DQPose.

    Args:
        models:     dict {'dq': DQPose model, 'xyz': XYZ model}
        input_xyz:  [1, N=25, 3, T_in] observed positions
        input_dq:   [1, N=25, 8, T_in] observed pose DQs
        output_pos: [1, N=25, 3, T_out] ground-truth future positions
        scale:      data scale (1000 means convert to mm for display)

    Returns:
        preds:     dict model_name -> [T_out, 25, 3] numpy (mm)
        gt:        [T_out, 25, 3] numpy (mm)
        input_pos: [T_in, 25, 3] numpy (mm)
    """
    if rotation_deg != 0:
        torch.manual_seed(seed)
        axis = torch.randn(3, device=device)
        axis = axis / axis.norm()
        angle_rad = torch.tensor(rotation_deg * np.pi / 180.0, device=device,
                                 dtype=torch.float32)
        R = axis_angle_to_rotmat(axis, angle_rad)
        input_xyz_p = _rotate_position_sequence(input_xyz, R, rotation_deg)
        output_p = _rotate_position_sequence(output_pos, R, rotation_deg)
        flat_output = output_pos.permute(0, 3, 1, 2).flatten(2)
        input_dq_p, _ = apply_rotation(R, input_dq, flat_output)
    else:
        input_xyz_p = input_xyz
        input_dq_p = input_dq
        output_p = output_pos

    preds = {}
    for mname, model in models.items():
        model.eval()
        model_input = input_dq_p if mname == 'dq' else input_xyz_p
        out = model(model_input)  # (B, N, 3, T_out)
        # Permute to (B, T_out, N, 3) and convert to mm
        pred_mm = out[0].permute(2, 0, 1).cpu().numpy() * scale  # [T_out, N, 3]
        preds[mname] = pred_mm

    gt = output_p[0].permute(2, 0, 1).cpu().numpy() * scale       # [T_out, N, 3]
    input_pos = input_xyz_p[0].permute(2, 0, 1).cpu().numpy() * scale  # [T_in, N, 3]

    return preds, gt, input_pos


# Representative sample selection

@torch.no_grad()
def find_representative_window(models, loaders_xyz, loaders_dq, scale, action, device):
    """
    Find the test window whose per-model 400ms MPJPE is closest to
    the per-action mean while minimising the gap between representations.

    Selection criterion:
      |DQ_err - DQ_mean| + |XYZ_err - XYZ_mean| + |DQ_err - XYZ_err|

    Returns:
        best_idx:     index of representative window in the dataset
        best_dq_err:  DQPose 400ms MPJPE at that window (mm)
        best_xyz_err: XYZ 400ms MPJPE at that window (mm)
    """
    frame_400_idx = EVAL_FRAME[-1][0]  # frame index 9

    window_errors = []  # list of [dataset_idx, dq_err, xyz_err]

    loader_xyz = loaders_xyz[action]
    loader_dq = loaders_dq[action]
    sample_idx = 0
    for xyz_batch, dq_batch in zip(loader_xyz, loader_dq):
        input_xyz, output_xyz, _ = xyz_batch
        input_dq, output_pos, _, _ = dq_batch
        batch_size = input_xyz.shape[0]
        input_xyz = input_xyz.float().to(device)
        input_dq = input_dq.float().to(device)
        output_pos = output_pos.float().to(device)

        for mname, model in models.items():
            model.eval()
            model_input = input_dq if mname == 'dq' else input_xyz
            out = model(model_input)  # (B, N, 3, T_out)
            # out and output_pos are both (B, N, 3, T)
            pred_mm = out * scale
            gt_mm = output_pos * scale

            err = torch.norm(
                pred_mm[:, :, :, frame_400_idx] -
                gt_mm[:, :, :, frame_400_idx],
                dim=-1
            ).mean(dim=1).detach().cpu().numpy()

            for bi, err_i in enumerate(err):
                idx = sample_idx + bi
                while len(window_errors) <= idx:
                    window_errors.append([len(window_errors), 0.0, 0.0])
                window_errors[idx][1 if mname == 'dq' else 2] = float(err_i)

        sample_idx += batch_size

    if not window_errors:
        return 0, 0.0, 0.0

    dq_mean = np.mean([w[1] for w in window_errors])
    xyz_mean = np.mean([w[2] for w in window_errors])

    def score(w):
        _, dq_e, xyz_e = w
        return abs(dq_e - dq_mean) + abs(xyz_e - xyz_mean) + abs(dq_e - xyz_e)

    best = min(window_errors, key=score)
    best_idx, best_dq, best_xyz = best

    print(f'    Representative window: idx={best_idx} '
          f'(DQPose={best_dq:.1f}mm [mean={dq_mean:.1f}], '
          f'XYZ={best_xyz:.1f}mm [mean={xyz_mean:.1f}])')
    print(f'    Total windows: {len(window_errors)}')

    return best_idx, best_dq, best_xyz


# Axis & camera helpers

CAMERA_ANGLES = {
    'walking':            (20, -60),
    'basketball':         (20,  30),
    'basketball_signal':  (20,  30),
    'directing_traffic':  (20,  30),
    'running':            (20, -60),
    'jumping':            (20,  30),
    'soccer':             (20,  30),
    'washwindow':         (20,  30),
}
DEFAULT_CAMERA = (20, 30)

ACTION_DISPLAY_NAMES = {
    'washwindow': 'Wash Window',
    'basketball_signal': 'Basketball Signal',
    'directing_traffic': 'Directing Traffic',
}


def _format_action_name(action):
    if action in ACTION_DISPLAY_NAMES:
        return ACTION_DISPLAY_NAMES[action]
    return action.replace('_', ' ').title()


def compute_axis_limits(all_positions):
    all_pos = np.concatenate([p.reshape(-1, 3) for p in all_positions], axis=0)
    center = (all_pos.max(axis=0) + all_pos.min(axis=0)) / 2
    max_range = (all_pos.max(axis=0) - all_pos.min(axis=0)).max() / 2
    margin = max_range * 1.05
    return center, margin


def _setup_clean_3d_axis(ax, center, margin, elev, azim):
    cx, cy, cz = center[0], center[2], center[1]
    ax.set_xlim(cx - margin, cx + margin)
    ax.set_ylim(cy - margin, cy + margin)
    ax.set_zlim(cz - margin, cz + margin)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.xaxis.line.set_color('none')
    ax.yaxis.line.set_color('none')
    ax.zaxis.line.set_color('none')
    ax.view_init(elev=elev, azim=azim)


def _draw_gt_and_pred(ax, gt_pos, pred_pos, bones, pred_color):
    draw_skeleton(ax, gt_pos, bones, color='#2E7D32', alpha=0.55,
                  linewidth=4.0, joint_size=10, label='GT')
    draw_skeleton_dashed(ax, pred_pos, bones, color=pred_color, alpha=0.92,
                         linewidth=2.0, joint_size=8, label=None)


def _compute_optimal_azimuth(pred_0, pred_rot, fallback_azim=30):
    diff = pred_rot - pred_0
    mean_dir = np.array([diff[:, 0].mean(), diff[:, 2].mean()])
    if np.linalg.norm(mean_dir) < 1.0:
        return fallback_azim
    theta_div = np.degrees(np.arctan2(mean_dir[1], mean_dir[0]))
    cand_a, cand_b = theta_div + 90, theta_div - 90
    dist_a = min((cand_a - fallback_azim) % 360, (fallback_azim - cand_a) % 360)
    dist_b = min((cand_b - fallback_azim) % 360, (fallback_azim - cand_b) % 360)
    return cand_a if dist_a <= dist_b else cand_b


def _get_rotation_matrix_np(deg, device, seed=42):
    if deg == 0:
        return np.eye(3)
    torch.manual_seed(seed)
    axis = torch.randn(3, device=device)
    axis = axis / axis.norm()
    angle_rad = torch.tensor(deg * np.pi / 180.0, device=device, dtype=torch.float32)
    R = axis_angle_to_rotmat(axis, angle_rad)
    return R.cpu().numpy()


# Per-frame MPJPE report

def print_mpjpe_report(gt_frames, preds_frames, action, rotation_deg):
    horizons = {1: 80, 3: 160, 7: 320, 9: 400}
    rot_str = f' (rot={rotation_deg}°)' if rotation_deg != 0 else ''
    print(f'\n  {action}{rot_str}:')
    for mname, pred in preds_frames.items():
        label = {'dq': 'DQPose', 'xyz': 'XYZ'}.get(mname, mname)
        errors = []
        for fi, ms in horizons.items():
            if fi < pred.shape[0] and fi < gt_frames.shape[0]:
                err = np.linalg.norm(
                    pred[fi] - gt_frames[fi], axis=-1
                ).mean()
                errors.append(f'{ms}ms={err:.1f}')
        print(f'    {label:4s}: {" | ".join(errors)} mm')


# Multi-row accuracy figure

def export_multi_row_figure(gt_frames, preds_frames, input_frames, bones,
                            action, rotation_deg, out_path, azim=None):
    """
    Export field-standard multi-row accuracy figure.

    Layout:
      Rows:    GGMotion-DQPose (top), GGMotion-XYZ (bottom)
      Columns: last input frame, 80ms, 160ms, 320ms, 400ms
    """
    columns = [
        ('Input', 'input', -1),
        ('80ms',  'pred',   1),
        ('160ms', 'pred',   3),
        ('320ms', 'pred',   7),
        ('400ms', 'pred',   9),
    ]

    model_order = [k for k in ('dq', 'xyz') if k in preds_frames]
    model_labels = {'dq': 'GGMotion-DQPose', 'xyz': 'GGMotion-XYZ'}
    model_colors = {'dq': '#2196F3', 'xyz': '#E65100'}

    n_rows = len(model_order)
    n_cols = len(columns)

    all_pos = [input_frames, gt_frames] + list(preds_frames.values())
    center, margin = compute_axis_limits(all_pos)

    elev, default_azim = CAMERA_ANGLES.get(action, DEFAULT_CAMERA)
    if azim is None:
        azim = default_azim

    fig = plt.figure(figsize=(22, 5.5 * n_rows), dpi=300)
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.14, right=1.0, top=0.80, bottom=0.04,
                        wspace=-0.10, hspace=0.05)

    display_action = _format_action_name(action)
    rot_label = f' | {rotation_deg}° rotation' if rotation_deg != 0 else ''
    fig.suptitle(f'{display_action}{rot_label}', fontsize=42,
                 fontweight='bold', y=0.98)

    for ri, mname in enumerate(model_order):
        color = model_colors[mname]
        label = model_labels[mname]

        pred_400 = preds_frames[mname][9]
        gt_400 = gt_frames[9]
        err_400 = np.linalg.norm(pred_400 - gt_400, axis=-1).mean()

        for ci, (col_label, col_src, col_fidx) in enumerate(columns):
            ax = fig.add_subplot(n_rows, n_cols, ri * n_cols + ci + 1,
                                 projection='3d')
            _setup_clean_3d_axis(ax, center, margin, elev, azim)

            is_input = (col_src == 'input')

            if is_input:
                pos = input_frames[col_fidx]
                draw_skeleton(ax, pos, bones, color='#888888', alpha=0.40,
                              linewidth=3.0, joint_size=10)
            else:
                gt_pos = gt_frames[col_fidx]
                pred_pos = preds_frames[mname][col_fidx]
                _draw_gt_and_pred(ax, gt_pos, pred_pos, bones, color)

                col_err = np.linalg.norm(pred_pos - gt_pos, axis=-1).mean()
                ax.text2D(0.5, 0.02, f'{col_err:.1f}mm',
                          transform=ax.transAxes, fontsize=24,
                          ha='center', va='bottom', color=color)

            if ri == 0:
                col_display = 'Input' if is_input else col_label
                ax.set_title(col_display, fontsize=33, pad=2,
                             color='#666666' if is_input else 'black')

            if ci == 0:
                row_text = f'{label}\n{err_400:.1f}mm at 400 ms'
                ax.text2D(-0.06, 0.5, row_text,
                          transform=ax.transAxes, fontsize=24,
                          fontweight='bold', va='center', ha='right',
                          rotation=0, color=color)

    legend_elements = [
        Line2D([0], [0], color='#2E7D32', linewidth=3.0, label='Ground truth'),
        Line2D([0], [0], color='#888888', linewidth=2.0, label='Input context'),
    ]
    for mname in model_order:
        legend_elements.append(
            Line2D([0], [0], color=model_colors[mname], linewidth=3.0,
                   linestyle='--', label=model_labels[mname] + ' prediction')
        )
    fig.legend(handles=legend_elements, loc='upper center',
               ncol=len(legend_elements), fontsize=22, framealpha=0.9,
               bbox_to_anchor=(0.57, 0.92), handlelength=3.0,
               columnspacing=3.0)

    fig.savefig(str(out_path), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Multi-row figure: {out_path}')


# Canonical-frame overlay figure (PRIMARY robustness visualization)

def export_canonical_overlay_figure(gt_frames, preds_0, preds_rot_inv,
                                    divergences, mpjpe_0, mpjpe_rot,
                                    bones, action, out_path, azim=None):
    """
    Export 2x2 canonical-frame overlay grid figure.

    Layout:  rows = models (DQPose top, XYZ bottom)
             cols = rotation angles (90° left, 180° right)
    """
    model_order = [k for k in ('dq', 'xyz') if k in preds_0]
    model_labels = {'dq': 'GGMotion-DQPose', 'xyz': 'GGMotion-XYZ'}

    rot_cols = sorted(preds_rot_inv.keys())
    rot_colors = {90: '#D32F2F', 180: '#7B1FA2'}
    rot_draw_fn = {90: draw_skeleton_dashed, 180: draw_skeleton_dotted}

    frame_idx = EVAL_FRAME[-1][0]  # 400ms = frame 9

    all_pos_frame = [gt_frames[frame_idx:frame_idx + 1]]
    for v in preds_0.values():
        all_pos_frame.append(v[frame_idx:frame_idx + 1])
    for rot_preds in preds_rot_inv.values():
        for v in rot_preds.values():
            all_pos_frame.append(v[frame_idx:frame_idx + 1])
    center, margin = compute_axis_limits(all_pos_frame)

    elev, default_azim = CAMERA_ANGLES.get(action, DEFAULT_CAMERA)
    if azim is None:
        azim = default_azim

    n_rows = len(model_order)
    n_cols = len(rot_cols)

    fig = plt.figure(figsize=(6 * n_cols, 5.5 * n_rows), dpi=300)
    fig.patch.set_facecolor('white')

    display_action = _format_action_name(action)
    fig.suptitle(
        f'Canonical-frame equivariance test: {display_action} at 400 ms',
        fontsize=28, fontweight='bold', x=0.52, y=0.98
    )

    legend_elements = [
        Line2D([0], [0], color='#2E7D32', linewidth=2.5, label='GT'),
        Line2D([0], [0], color='#2196F3', linewidth=3.0,
               label='0° prediction'),
        Line2D([0], [0], color='#D32F2F', linewidth=2.5, linestyle='--',
               label='90°-inv prediction'),
        Line2D([0], [0], color='#7B1FA2', linewidth=2.5, linestyle='-.',
               label='180°-inv prediction'),
    ]
    fig.legend(handles=legend_elements, loc='upper center',
               ncol=4, fontsize=16, framealpha=0.9,
               bbox_to_anchor=(0.52, 0.94), handlelength=2.5)

    _top, _bot = 0.82, 0.10
    _subplot_h = (_top - _bot) / (n_rows + (n_rows - 1) * 0.22)
    _row_centers = [_top - (ri + 0.5) * _subplot_h - ri * 0.22 * _subplot_h
                    for ri in range(n_rows)]

    for ri, mname in enumerate(model_order):
        fig.text(0.02, _row_centers[ri], model_labels[mname],
                 fontsize=24, fontweight='bold', va='center', ha='left',
                 rotation=0)

        for ci, rot_deg in enumerate(rot_cols):
            panel_idx = ri * n_cols + ci + 1
            ax = fig.add_subplot(n_rows, n_cols, panel_idx, projection='3d')
            _setup_clean_3d_axis(ax, center, margin, elev, azim)

            gt_pos = gt_frames[frame_idx]
            pred_0_pos = preds_0[mname][frame_idx]
            pred_rot_pos = preds_rot_inv[rot_deg][mname][frame_idx]

            draw_skeleton(ax, gt_pos, bones, color='#2E7D32', alpha=0.45,
                          linewidth=3.5, joint_size=14)
            draw_skeleton(ax, pred_0_pos, bones, color='#2196F3', alpha=0.95,
                          linewidth=3.0, joint_size=12)
            rot_draw_fn[rot_deg](
                ax, pred_rot_pos, bones,
                color=rot_colors[rot_deg], alpha=0.90,
                linewidth=2.0, joint_size=10,
            )

            if ri == 0:
                ax.set_title(f'{rot_deg}° rotation overlay',
                             fontsize=24, pad=20, color='black',
                             fontweight='bold')

            mp0_val = mpjpe_0[mname]
            mp_rot_val = mpjpe_rot[rot_deg][mname]
            delta = mp_rot_val - mp0_val
            delta_sign = '+' if delta >= 0 else ''
            div_val = divergences[rot_deg][mname]

            annot_text = (f'MPJPE: {mp0_val:.1f} to {mp_rot_val:.1f}mm'
                          f' (Δ = {delta_sign}{delta:.1f}mm)'
                          f'\nDivergence: {div_val:.1f}mm')

            bbox_props = dict(boxstyle='round,pad=0.4', facecolor='white',
                              edgecolor='#999999', alpha=0.92)
            ax.text2D(
                0.5, -0.10, annot_text,
                transform=ax.transAxes, fontsize=14,
                ha='center', va='top', fontweight='bold',
                color='#333333', bbox=bbox_props,
            )

    fig.subplots_adjust(left=0.14, right=0.98, top=0.82, bottom=0.14,
                        wspace=0.02, hspace=0.30)

    fig.savefig(str(out_path), bbox_inches='tight', pad_inches=0.2,
                facecolor='white')
    plt.close(fig)
    print(f'  Overlay grid figure: {out_path}')


# Per-model robustness figure (0°/90°/180° rows × time columns)

def export_robustness_per_model(gt_by_rot, preds_by_rot, input_by_rot,
                                bones, action, model_name, out_path,
                                azim=None):
    """
    Export per-model robustness figure showing predictions in the rotated frame.
    """
    model_labels = {'dq': 'GGMotion-DQPose', 'xyz': 'GGMotion-XYZ'}
    model_colors = {'dq': '#2196F3', 'xyz': '#E65100'}
    color = model_colors.get(model_name, '#333333')
    label = model_labels.get(model_name, model_name)

    col_defs = [
        ('Input', 'input', -1),
        ('80ms',  'pred',   1),
        ('160ms', 'pred',   3),
        ('320ms', 'pred',   7),
        ('400ms', 'pred',   9),
    ]

    rot_rows = sorted(gt_by_rot.keys())
    n_rows = len(rot_rows)
    n_cols = len(col_defs)

    all_pos = []
    for rot_deg in rot_rows:
        all_pos.extend([input_by_rot[rot_deg], gt_by_rot[rot_deg],
                        preds_by_rot[rot_deg]])
    center, margin = compute_axis_limits(all_pos)

    elev, default_azim = CAMERA_ANGLES.get(action, DEFAULT_CAMERA)
    if azim is None:
        azim = default_azim

    fig = plt.figure(figsize=(22, 5.5 * n_rows), dpi=300)
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.14, right=1.0, top=0.85, bottom=0.04,
                        wspace=-0.10, hspace=0.05)

    display_action = _format_action_name(action)
    fig.suptitle(f'{label} - {display_action}', fontsize=42,
                 fontweight='bold', y=0.98)

    for ri, rot_deg in enumerate(rot_rows):
        gt_f = gt_by_rot[rot_deg]
        preds = preds_by_rot[rot_deg]
        input_f = input_by_rot[rot_deg]

        pred_400 = preds[9]
        gt_400 = gt_f[9]
        err_400 = np.linalg.norm(pred_400 - gt_400, axis=-1).mean()

        for ci, (col_label, col_src, col_fidx) in enumerate(col_defs):
            ax = fig.add_subplot(n_rows, n_cols, ri * n_cols + ci + 1,
                                 projection='3d')
            _setup_clean_3d_axis(ax, center, margin, elev, azim)

            is_input = (col_src == 'input')

            if is_input:
                pos = input_f[col_fidx]
                draw_skeleton(ax, pos, bones, color='#888888', alpha=0.40,
                              linewidth=3.0, joint_size=10)
            else:
                gt_pos = gt_f[col_fidx]
                pred_pos = preds[col_fidx]
                _draw_gt_and_pred(ax, gt_pos, pred_pos, bones, color)

                col_err = np.linalg.norm(pred_pos - gt_pos, axis=-1).mean()
                ax.text2D(0.5, 0.02, f'{col_err:.1f}mm',
                          transform=ax.transAxes, fontsize=24,
                          ha='center', va='bottom', color=color)

            if ri == 0:
                ax.set_title(col_label, fontsize=33, pad=2,
                             color='#666666' if is_input else 'black')

            if ci == 0:
                row_text = f'{rot_deg}° rotation\n{err_400:.1f}mm at 400 ms'
                ax.text2D(-0.06, 0.5, row_text,
                          transform=ax.transAxes, fontsize=24,
                          fontweight='bold', va='center', ha='right',
                          rotation=0, color=color)

    legend_elements = [
        Line2D([0], [0], color='#2E7D32', linewidth=3.0, label='Ground truth'),
        Line2D([0], [0], color='#888888', linewidth=2.0, label='Input context'),
        Line2D([0], [0], color=color, linewidth=3.0,
               linestyle='--', label=f'{label} prediction'),
    ]
    fig.legend(handles=legend_elements, loc='upper center',
               ncol=3, fontsize=22, framealpha=0.9,
               bbox_to_anchor=(0.57, 0.93), handlelength=3.5,
               columnspacing=8.0)

    fig.savefig(str(out_path), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Robustness figure: {out_path}')


# Animation

def create_animation(gt_frames, preds_frames, input_frames, bones, action,
                     horizon_ms, rotation_deg, fps=3, azim=None):
    """Create a side-by-side animation: GT overlaid on each model's prediction."""
    n_input = input_frames.shape[0]
    n_pred = gt_frames.shape[0]

    model_names = list(preds_frames.keys())
    model_labels = {'dq': 'GGMotion-DQPose', 'xyz': 'GGMotion-XYZ'}
    model_colors = {'dq': '#2196F3', 'xyz': '#E65100'}

    schedule = []
    for i in range(n_input):
        schedule.extend([('input', i)] * 3)
    schedule.extend([('pause', -1)] * 3)
    for i in range(n_pred):
        schedule.extend([('pred', i)] * 4)
    schedule.extend([('pred', n_pred - 1)] * 8)

    all_pos = [input_frames, gt_frames]
    for v in preds_frames.values():
        all_pos.append(v)
    center, margin = compute_axis_limits(all_pos)

    fig = plt.figure(figsize=(14, 6), dpi=100)
    fig.patch.set_facecolor('white')

    display_action = _format_action_name(action)
    rot_label = f' | rotation: {rotation_deg}°' if rotation_deg != 0 else ''
    fig.suptitle(f'{display_action}{rot_label}', fontsize=16,
                 fontweight='bold', y=0.98)

    elev, default_azim = CAMERA_ANGLES.get(action, DEFAULT_CAMERA)
    if azim is None:
        azim = default_azim

    n_panels = max(len(model_names), 1)
    axes = []
    cx, cy, cz = center[0], center[2], center[1]
    for i in range(n_panels):
        ax = fig.add_subplot(1, n_panels, i + 1, projection='3d')
        ax.set_xlim(cx - margin, cx + margin)
        ax.set_ylim(cy - margin, cy + margin)
        ax.set_zlim(cz - margin, cz + margin)
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Z', fontsize=8)
        ax.set_zlabel('Y (up)', fontsize=8)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=elev, azim=azim)
        axes.append(ax)

    frame_texts = []
    for ax in axes:
        txt = ax.text2D(0.02, 0.02, '', transform=ax.transAxes, fontsize=10)
        frame_texts.append(txt)

    def update(sched_idx):
        phase, data_idx = schedule[sched_idx]
        for ax in axes:
            while ax.lines:
                ax.lines[0].remove()
            while ax.collections:
                ax.collections[0].remove()
        for pi, mname in enumerate(model_names):
            ax = axes[pi]
            color = model_colors.get(mname, 'red')
            label = model_labels.get(mname, mname)
            if phase == 'input':
                pos = input_frames[data_idx]
                draw_skeleton(ax, pos, bones, color='#2E7D32', alpha=0.45,
                              linewidth=1.5, joint_size=8, label='GT/Input')
                draw_skeleton(ax, pos, bones, color=color, alpha=0.7,
                              linewidth=2.0, joint_size=10, label=label)
                frame_texts[pi].set_text(f'INPUT frame {data_idx+1}/{n_input}')
                ax.set_title(label, fontsize=12, pad=2)
            elif phase == 'pause':
                pos = input_frames[-1]
                draw_skeleton(ax, pos, bones, color='#CCCCCC', alpha=0.3,
                              linewidth=1.0, joint_size=6)
                frame_texts[pi].set_text('--- PREDICTING ---')
                ax.set_title(label, fontsize=12, pad=2)
            elif phase == 'pred':
                gt_pos = gt_frames[data_idx]
                pred_pos = preds_frames[mname][data_idx]
                draw_skeleton(ax, gt_pos, bones, color='#2E7D32', alpha=0.55,
                              linewidth=4.0, joint_size=14, label='GT')
                draw_skeleton_dashed(ax, pred_pos, bones, color=color,
                                     alpha=0.92, linewidth=2.0, joint_size=8,
                                     label=label)
                err = np.linalg.norm(pred_pos - gt_pos, axis=-1).mean()
                time_ms = (data_idx + 1) * 40
                frame_texts[pi].set_text(
                    f'PRED t={time_ms}ms | MPJPE={err:.1f}mm')
                ax.set_title(label, fontsize=12, pad=2)
                ax.legend(loc='upper right', fontsize=9, framealpha=0.7)
        return []

    anim_obj = animation.FuncAnimation(
        fig, update, frames=len(schedule),
        interval=1000 // fps, blit=False
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    return fig, anim_obj


def create_equivariance_animation(gt_frames, preds_0, preds_90_inv,
                                  input_frames, bones, action,
                                  fps=3, azim=None):
    """
    Animated canonical-frame overlay at 90° with two panels (DQPose left, XYZ right).

    Each panel shows 3 overlaid skeletons:
      - GT (green, solid, wide): anchor
      - pred@0° (blue, solid): baseline prediction
      - pred@90°-inv (red, dashed): inverse-rotated 90° prediction

    For equivariant DQPose, all three predictions collapse into one skeleton.
    For non-equivariant (XYZ): red skeleton diverges progressively.

    Uses 90° because it produces the strongest XYZ divergence for GGMotion
    on the primary actions (running 118.7mm, jumping 63.0mm). The specific
    rotation axis (seed=42) interacts with each angle differently.

    Includes input phase + pause (matching accuracy animation pacing) and
    full MPJPE + divergence annotation on every prediction frame.
    """
    n_input = input_frames.shape[0]
    n_pred = gt_frames.shape[0]

    model_order = [k for k in ('dq', 'xyz') if k in preds_0]
    model_labels = {'dq': 'GGMotion-DQPose', 'xyz': 'GGMotion-XYZ'}

    schedule = []
    for i in range(n_input):
        schedule.extend([('input', i)] * 3)
    schedule.extend([('pause', -1)] * 3)
    for i in range(n_pred):
        schedule.extend([('pred', i)] * 4)
    schedule.extend([('pred', n_pred - 1)] * 8)

    all_pos = [gt_frames]
    for mname in model_order:
        all_pos.append(preds_0[mname])
        all_pos.append(preds_90_inv[mname])
    center, margin = compute_axis_limits(all_pos)

    n_panels = len(model_order)
    fig = plt.figure(figsize=(8 * n_panels, 7), dpi=100)
    fig.patch.set_facecolor('white')
    display_action = _format_action_name(action)
    fig.suptitle(f'{display_action} - Canonical-frame overlay (90°)',
                 fontsize=15, fontweight='bold', y=0.98)

    axes = []
    frame_texts = []
    cx, cy, cz = center[0], center[2], center[1]
    for pi, mname in enumerate(model_order):
        ax = fig.add_subplot(1, n_panels, pi + 1, projection='3d')
        ax.set_xlim(cx - margin, cx + margin)
        ax.set_ylim(cy - margin, cy + margin)
        ax.set_zlim(cz - margin, cz + margin)
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Z', fontsize=8)
        ax.set_zlabel('Y (up)', fontsize=8)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=CAMERA_ANGLES.get(action, DEFAULT_CAMERA)[0],
                     azim=azim if azim is not None else CAMERA_ANGLES.get(action, DEFAULT_CAMERA)[1])
        ax.set_title(model_labels[mname], fontsize=13, fontweight='bold')
        ft = ax.text2D(0.02, 0.02, '', transform=ax.transAxes, fontsize=10)
        axes.append(ax)
        frame_texts.append(ft)

    def update(sched_idx):
        phase, data_idx = schedule[sched_idx]

        for ax in axes:
            while ax.lines:
                ax.lines[0].remove()
            while ax.collections:
                ax.collections[0].remove()

        for pi, mname in enumerate(model_order):
            ax = axes[pi]
            ft = frame_texts[pi]

            if phase == 'input':
                pos = input_frames[data_idx]
                draw_skeleton(ax, pos, bones, color='#2E7D32', alpha=0.55,
                              linewidth=3.0, joint_size=10, label='GT')
                ft.set_text(f'INPUT frame {data_idx+1}/{n_input}')

            elif phase == 'pause':
                pos = input_frames[-1]
                draw_skeleton(ax, pos, bones, color='#CCCCCC', alpha=0.3,
                              linewidth=1.0, joint_size=6)
                ft.set_text('--- PREDICTING ---')

            elif phase == 'pred':
                gt_pos = gt_frames[data_idx]
                p0_pos = preds_0[mname][data_idx]
                p90_pos = preds_90_inv[mname][data_idx]

                draw_skeleton(ax, gt_pos, bones, color='#2E7D32', alpha=0.55,
                              linewidth=4.0, joint_size=14, label='GT')
                draw_skeleton(ax, p0_pos, bones, color='#1565C0', alpha=0.85,
                              linewidth=2.5, joint_size=10, label='0° pred')
                draw_skeleton_dashed(ax, p90_pos, bones, color='#D32F2F',
                                     alpha=0.85, linewidth=2.0, joint_size=8,
                                     label='90°-inv pred')

                err_0 = np.linalg.norm(p0_pos - gt_pos, axis=-1).mean()
                err_90 = np.linalg.norm(p90_pos - gt_pos, axis=-1).mean()
                delta = err_90 - err_0
                delta_sign = '+' if delta >= 0 else ''
                div = np.linalg.norm(p0_pos - p90_pos, axis=-1).mean()
                time_ms = (data_idx + 1) * 40
                ft.set_text(
                    f't={time_ms}ms | MPJPE: {err_0:.1f} to '
                    f'{err_90:.1f}mm (Δ{delta_sign}{delta:.1f})  '
                    f'Div: {div:.1f}mm'
                )
                ax.legend(loc='upper right', fontsize=9, framealpha=0.7)

        return []

    anim_obj = animation.FuncAnimation(
        fig, update, frames=len(schedule),
        interval=1000 // fps, blit=False
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.92])
    return fig, anim_obj


# Main

def main():
    parser = argparse.ArgumentParser(
        description='Qualitative validation: 3D skeleton figures (GGMotion)')
    parser.add_argument('--actions', nargs='+', default=['basketball'],
                        choices=ACTIONS_8,
                        help='Actions to visualize')
    parser.add_argument('--horizon', type=int, default=400,
                        choices=[80, 160, 320, 400],
                        help='Target horizon for static frame export')
    parser.add_argument('--rotation', nargs='+', type=float, default=[],
                        help='Rotation angles in degrees (e.g. 0 90 180)')
    parser.add_argument('--model', choices=['dq', 'xyz', 'both'],
                        default='both',
                        help='Which model(s) to visualize')
    parser.add_argument('--dq-checkpoint',
                        default='exp/dqposemindirect_retrained/cmu_best_24.595_22.pt')
    parser.add_argument('--xyz-checkpoint',
                        default='exp/xyz_reproduced/cmu_best_20.823_19.pt')
    parser.add_argument('--cfg', default='cfg/cmu_dqpose_mindirect_short.yml',
                        help='Path to config YAML')
    parser.add_argument('--data', default='./data/cmu',
                        help='Path to CMU data directory')
    parser.add_argument('--out-dir',
                        default='experiments/results/visualizations')
    parser.add_argument('--sample-idx', type=int, default=None,
                        help='Override auto-sample: use this window index')
    parser.add_argument('--no-auto-sample', action='store_true',
                        help='Disable auto sample selection')
    parser.add_argument('--device', default=None)
    parser.add_argument('--fps', type=int, default=3,
                        help='Animation FPS')
    parser.add_argument('--gif', action='store_true',
                        help='Also export GIF')
    parser.add_argument('--no-mp4', action='store_true',
                        help='Skip MP4 animation export')
    parser.add_argument('--multi-row', action='store_true',
                        help='Export accuracy multi-row figure')
    parser.add_argument('--robustness-figure', action='store_true',
                        help='Export per-model robustness figure (0/90/180°)')
    parser.add_argument('--canonical-overlay', action='store_true',
                        help='Export canonical-frame overlay grid')
    parser.add_argument('--find-samples', action='store_true',
                        help='Only find representative windows, no visualization')
    parser.add_argument('--manner', default='all',
                        help='Test sampling: all, 256, or 8')
    parser.add_argument('--sample-batch-size', type=int, default=64,
                        help='Batch size used only for representative-window search')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for rotation axis')
    args = parser.parse_args()

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)
    print(f'Device: {device}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    cfg_path = _ROOT / args.cfg
    config = load_config(cfg_path)

    setup_seed(1234567890)

    # Load test data
    print(f'\nLoading CMU test set from {args.data} ...')
    loaders_xyz = {}
    loaders_dq = {}
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
            dataset_xyz, batch_size=args.sample_batch_size,
            shuffle=False, drop_last=False, num_workers=0, pin_memory=True)
        loaders_dq[act] = DataLoader(
            dataset_dq, batch_size=args.sample_batch_size,
            shuffle=False, drop_last=False, num_workers=0, pin_memory=True)
        len_test += len(dataset_xyz)
        dataset_ref = dataset_xyz
        dataset_dq_ref = dataset_dq
    print(f'  {len_test} test samples across {len(ACTIONS_8)} actions')

    scale = config.scale
    parent = dataset_ref.parent
    bones = get_bone_lines(parent)

    # Load models
    models = {}
    if args.model in ('dq', 'both'):
        print(f'\nLoading GGMotion-DQPose: {args.dq_checkpoint}')
        models['dq'] = load_ggmotion_model(
            args.dq_checkpoint, 'dq', config,
            dataset_dq_ref.group, dataset_dq_ref.edges, device)

    if args.model in ('xyz', 'both'):
        print(f'\nLoading GGMotion-XYZ: {args.xyz_checkpoint}')
        models['xyz'] = load_ggmotion_model(
            args.xyz_checkpoint, 'xyz', config,
            dataset_ref.group, dataset_ref.edges, device)

    print(f'\nLoaded models: {list(models.keys())}')

    use_auto_sample = (args.sample_idx is None) and (not args.no_auto_sample)

    if args.multi_row and not args.rotation:
        args.rotation = [0.0]

    print(f'\nGenerating visualizations...')
    print(f'  Actions: {args.actions}')
    print(f'  Rotations: {args.rotation}')
    print(f'  Horizon: {args.horizon}ms')
    print(f'  Sample selection: {"auto (balanced per-model mean)" if use_auto_sample else "manual"}')

    for action in args.actions:
        print(f'\n  --- {action} ---')

        # Find representative window
        if use_auto_sample:
            best_idx, dq_err, xyz_err = find_representative_window(
                models, loaders_xyz, loaders_dq, scale, action, device)
        else:
            best_idx = args.sample_idx if args.sample_idx is not None else 0
            print(f'    Manual window: idx={best_idx}')

        if args.find_samples:
            continue

        # Get the selected sample
        dataset_xyz = loaders_xyz[action].dataset
        dataset_dq = loaders_dq[action].dataset
        input_xyz_np, output_xyz_np, _ = dataset_xyz[best_idx]
        input_dq_np, output_pos_np, _, _ = dataset_dq[best_idx]
        # Add batch dim.
        input_xyz = torch.from_numpy(input_xyz_np).unsqueeze(0).float().to(device)
        input_dq = torch.from_numpy(input_dq_np).unsqueeze(0).float().to(device)
        output_pos = torch.from_numpy(output_pos_np).unsqueeze(0).float().to(device)

        # Shared auto-azimuth
        elev, fallback_azim = CAMERA_ANGLES.get(action, DEFAULT_CAMERA)
        if 'xyz' in models:
            preds_0_az, gt_0_az, _ = predict_with_rotation(
                models, input_xyz, input_dq, output_pos, scale, device,
                rotation_deg=0, seed=args.seed)
            preds_180_az, _, _ = predict_with_rotation(
                models, input_xyz, input_dq, output_pos, scale, device,
                rotation_deg=180, seed=args.seed)
            R180 = _get_rotation_matrix_np(180, device, seed=args.seed)
            frame_400_idx = EVAL_FRAME[-1][0]
            pred_0_xyz = preds_0_az['xyz'][frame_400_idx]
            pred_180_inv_xyz = (preds_180_az['xyz'] @ R180)[frame_400_idx]
            shared_azim = _compute_optimal_azimuth(
                pred_0_xyz, pred_180_inv_xyz, fallback_azim)
            print(f'  Shared auto-azimuth: {shared_azim:.0f}° '
                  f'(fallback={fallback_azim}°)')
        else:
            shared_azim = fallback_azim

        # Robustness figure (per-model, 0°/90°/180°)
        if args.robustness_figure:
            print(f'\n  === {action} | robustness figures (per-model) ===')

            rob_rotations = [0, 90, 180]
            gt_by_rot = {}
            preds_by_rot = {}
            input_by_rot = {}

            for rot_deg in rob_rotations:
                preds_r, gt_r, inp_r = predict_with_rotation(
                    models, input_xyz, input_dq, output_pos, scale, device,
                    rotation_deg=rot_deg, seed=args.seed)
                gt_by_rot[rot_deg] = gt_r
                input_by_rot[rot_deg] = inp_r
                preds_by_rot[rot_deg] = preds_r
                print_mpjpe_report(gt_r, preds_r, action, rot_deg)

            model_order = [k for k in ('dq', 'xyz') if k in preds_by_rot[0]]
            for mname in model_order:
                rob_path = out_dir / f'rotation_fig_{mname}_{action}.png'
                export_robustness_per_model(
                    gt_by_rot,
                    {r: preds_by_rot[r][mname] for r in rob_rotations},
                    input_by_rot,
                    bones, action, mname, rob_path,
                    azim=shared_azim
                )

        # Canonical-frame overlay grid
        if args.canonical_overlay:
            print(f'\n  === {action} | canonical-frame overlay grid ===')

            frame_400_idx = EVAL_FRAME[-1][0]

            preds_0, gt_0, _ = predict_with_rotation(
                models, input_xyz, input_dq, output_pos, scale, device,
                rotation_deg=0, seed=args.seed)

            mpjpe_0 = {}
            for mname, pred in preds_0.items():
                err = np.linalg.norm(
                    pred[frame_400_idx] - gt_0[frame_400_idx], axis=-1
                ).mean()
                mpjpe_0[mname] = err
                label = {'dq': 'DQPose', 'xyz': 'XYZ'}.get(mname, mname)
                print(f'    {label} MPJPE@0°: {err:.1f}mm')

            preds_rot_inv = {}
            mpjpe_rot = {}
            divergences = {}

            for rot_deg in [90, 180]:
                preds_rot, gt_rot, _ = predict_with_rotation(
                    models, input_xyz, input_dq, output_pos, scale, device,
                    rotation_deg=rot_deg, seed=args.seed)
                R_np = _get_rotation_matrix_np(rot_deg, device, seed=args.seed)

                inv_preds = {}
                mpjpe_rot[rot_deg] = {}
                divergences[rot_deg] = {}

                for mname, pred in preds_rot.items():
                    inv_preds[mname] = pred @ R_np

                    err = np.linalg.norm(
                        pred[frame_400_idx] - gt_rot[frame_400_idx],
                        axis=-1
                    ).mean()
                    mpjpe_rot[rot_deg][mname] = err

                    div = np.linalg.norm(
                        preds_0[mname][frame_400_idx] -
                        inv_preds[mname][frame_400_idx],
                        axis=-1
                    ).mean()
                    divergences[rot_deg][mname] = div

                    label = {'dq': 'DQPose', 'xyz': 'XYZ'}.get(mname, mname)
                    print(f'    {label} @{rot_deg}°: '
                          f'MPJPE={err:.1f}mm  Div={div:.1f}mm')

                preds_rot_inv[rot_deg] = inv_preds

            grid_path = out_dir / f'equivariance_fig_both_{action}.png'
            export_canonical_overlay_figure(
                gt_0, preds_0, preds_rot_inv, divergences,
                mpjpe_0, mpjpe_rot, bones, action, grid_path,
                azim=shared_azim
            )

            # Equivariance video (90° overlay animated over time)
            if not args.no_mp4 and 90 in preds_rot_inv:
                _, _, input_viz_eq = predict_with_rotation(
                    models, input_xyz, input_dq, output_pos, scale, device,
                    rotation_deg=0, seed=args.seed)

                fig_eq, anim_eq = create_equivariance_animation(
                    gt_0, preds_0, preds_rot_inv[90],
                    input_viz_eq, bones, action, fps=args.fps,
                    azim=shared_azim
                )
                eq_mp4 = out_dir / f'equivariance_vid_both_{action}.mp4'
                animation.FFMpegWriter.ffmpeg_path = FFMPEG_PATH
                writer = animation.FFMpegWriter(
                    fps=args.fps,
                    extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p']
                )
                anim_eq.save(str(eq_mp4), writer=writer)
                plt.close(fig_eq)
                print(f'  Equivariance MP4: {eq_mp4}')

        # Per-rotation outputs
        for rot_deg in args.rotation:
            rot_deg_int = int(rot_deg)
            print(f'\n  === {action} | rotation={rot_deg_int}° ===')

            preds, gt_viz, input_viz = predict_with_rotation(
                models, input_xyz, input_dq, output_pos, scale, device,
                rotation_deg=rot_deg_int, seed=args.seed)

            print_mpjpe_report(gt_viz, preds, action, rot_deg_int)

            if args.multi_row:
                mr_path = out_dir / f'accuracy_fig_both_{action}.png'
                export_multi_row_figure(
                    gt_viz, preds, input_viz, bones,
                    action, rot_deg_int, mr_path, azim=shared_azim
                )

            if not args.no_mp4:
                fig, anim_obj = create_animation(
                    gt_viz, preds, input_viz, bones,
                    action, args.horizon, rot_deg_int, fps=args.fps,
                    azim=shared_azim
                )
                rot_suffix = f'_rot{rot_deg_int}' if rot_deg_int != 0 else ''
                mp4_path = out_dir / f'accuracy_vid_both_{action}{rot_suffix}.mp4'
                animation.FFMpegWriter.ffmpeg_path = FFMPEG_PATH
                writer = animation.FFMpegWriter(
                    fps=args.fps,
                    extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p']
                )
                anim_obj.save(str(mp4_path), writer=writer)
                plt.close(fig)
                print(f'  MP4: {mp4_path}')

            if args.gif:
                fig, anim_obj = create_animation(
                    gt_viz, preds, input_viz, bones,
                    action, args.horizon, rot_deg_int, fps=args.fps,
                    azim=shared_azim
                )
                gif_path = out_dir / f'accuracy_gif_both_{action}{rot_suffix}.gif'
                anim_obj.save(str(gif_path), writer='pillow', fps=args.fps)
                plt.close(fig)
                print(f'  GIF: {gif_path}')

    print(f'\nAll outputs saved to {out_dir}/')


if __name__ == '__main__':
    main()
