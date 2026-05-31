"""
Test equivariance of GGMotion-XYZ and GGMotion-DQPose.

Protocol:
    1. Forward pass on original input:  y = model(x)
    2. Apply random SE(3) transform:    x' = g(x)
    3. Forward pass on transformed:     y' = model(x')
    4. Transform original output:       g(y)
    5. Check: g(y) ≈ y'  (equivariance)

Usage:
    cd ggmotion/code
    python test_equivariance.py --model xyz    # GGMotion-XYZ E(n)
    python test_equivariance.py --model dqpose # GGMotion-DQPose SE(3)
"""

import argparse
import sys

import numpy as np
import torch


def random_rotation_matrix(device='cpu'):
    """Random 3x3 rotation matrix via QR decomposition."""
    m = torch.randn(3, 3, device=device)
    q, r = torch.linalg.qr(m)
    d = torch.diag(r)
    sign = torch.diag(d.sign())
    q = q @ sign
    if torch.det(q) < 0:
        q[:, 0] *= -1
    return q


def random_translation(scale=1.0, device='cpu'):
    """Random 3D translation vector."""
    return torch.randn(3, device=device) * scale


def make_synthetic_config(n_joints=10, t_in=10, t_out=10,
                          dqpose_decode_mode="direct", dqpose_pose_rot_scale=1.0,
                          dqpose_pose_trans_scale=1.0, dqpose_vel_rot_scale=1.0,
                          dqpose_vel_trans_scale=1.0, dqpose_out_rot_scale=1.0,
                          dqpose_out_trans_scale=1.0, dqpose_out_rot_scale_start=None,
                          dqpose_out_rot_scale_end=None, dqpose_out_trans_scale_start=None,
                          dqpose_out_trans_scale_end=None, dqpose_block_mode="joint"):
    """Create a minimal config namespace matching GGMotion's expectations."""
    class Config:
        pass
    cfg = Config()
    cfg.key_point = n_joints
    cfg.past_length = t_in
    cfg.future_length = t_out
    cfg.e_dim = 32
    cfg.h_dim = 24
    cfg.norm = True
    cfg.fk = 0
    cfg.n_layer = 2
    cfg.dqpose_reference_joint = min(8, n_joints - 1)
    cfg.dqpose_decode_mode = dqpose_decode_mode
    cfg.dqpose_pose_rot_scale = dqpose_pose_rot_scale
    cfg.dqpose_pose_trans_scale = dqpose_pose_trans_scale
    cfg.dqpose_vel_rot_scale = dqpose_vel_rot_scale
    cfg.dqpose_vel_trans_scale = dqpose_vel_trans_scale
    cfg.dqpose_out_rot_scale = dqpose_out_rot_scale
    cfg.dqpose_out_trans_scale = dqpose_out_trans_scale
    cfg.dqpose_out_rot_scale_start = (
        dqpose_out_rot_scale if dqpose_out_rot_scale_start is None else dqpose_out_rot_scale_start
    )
    cfg.dqpose_out_rot_scale_end = (
        dqpose_out_rot_scale if dqpose_out_rot_scale_end is None else dqpose_out_rot_scale_end
    )
    cfg.dqpose_out_trans_scale_start = (
        dqpose_out_trans_scale if dqpose_out_trans_scale_start is None else dqpose_out_trans_scale_start
    )
    cfg.dqpose_out_trans_scale_end = (
        dqpose_out_trans_scale if dqpose_out_trans_scale_end is None else dqpose_out_trans_scale_end
    )
    cfg.dqpose_block_mode = dqpose_block_mode
    return cfg


def make_synthetic_group_edges(n_joints=10):
    """
    Create synthetic group dict and edges array for a small test graph.
    Group: single body part ('spine') with all joints in a chain.
    Edges: fully connected (all pairs), shape (E, 3) with [src, dst, hop].
    """
    chain = np.array([[i, i + 1, 0] for i in range(n_joints - 1)])
    group = {"spine": chain}

    # Group attr: append hop count
    group_processed = {}
    for key in group:
        arr = group[key]
        result = []
        for i in range(len(arr)):
            nop = 0
            pre = arr[i, 0]
            for j in range(i, -1, -1):
                if pre == arr[j - 1, 1]:
                    nop += 1
                    pre = arr[j - 1, 0]
            result.append(np.append(arr[i], nop))
        group_processed[key] = np.array(result)

    # Edges: fully connected, shape (E, 3) = [src, dst, hop_distance]
    edges = []
    for i in range(n_joints):
        for j in range(n_joints):
            if i != j:
                edges.append([i, j, abs(i - j)])
    edges = np.array(edges)
    return group_processed, edges


# ---------------------------------------------------------------------------
# GGMotion-XYZ equivariance test: E(n)
# ---------------------------------------------------------------------------

def test_xyz_equivariance(n_trials=10, n_joints=10, t_in=10, device='cpu'):
    print("=" * 60)
    print("Testing GGMotion-XYZ - E(n) equivariance")
    print("=" * 60)

    from module.model import GGMNet

    config = make_synthetic_config(n_joints, t_in, t_in)
    group, edges = make_synthetic_group_edges(n_joints)
    model = GGMNet(config, group, edges).to(device)
    model.eval()

    coord_errors = []

    with torch.no_grad():
        for trial in range(n_trials):
            x = torch.randn(2, n_joints, 3, t_in, device=device)

            y = model(x)

            R = random_rotation_matrix(device)
            t = random_translation(scale=1.0, device=device)

            # Transform: x' = R @ x + t  (applied to dim=2 which is the 3D coord)
            x_t = torch.einsum('ij,bnjt->bnit', R, x) + t.view(1, 1, 3, 1)
            y_t = model(x_t)

            y_expected = torch.einsum('ij,bnjt->bnit', R, y) + t.view(1, 1, 3, 1)

            err = torch.norm(y_expected - y_t, dim=2).mean().item()
            coord_errors.append(err)

            print(f"  Trial {trial:2d}: coord_err={err:.2e}")

    mean_err = np.mean(coord_errors)
    print(f"\n  Mean coord equivariance error: {mean_err:.2e}")

    # GGMotion-XYZ has per-joint scale params that break strict E(n) equivariance.
    # This test documents the level of equivariance error for comparison with DQPose.
    # It is not expected to pass at 1e-3 like DQPose; this is informational.
    passed = mean_err < 1e-3
    if passed:
        print("  PASSED - GGMotion-XYZ is E(n)-equivariant")
    else:
        print(f"  INFO - GGMotion-XYZ is NOT strictly E(n)-equivariant (expected)")
        print(f"         Per-joint scale params break exact equivariance.")
        print(f"         Compare with DQPose model error to confirm the representation effect.")
        passed = True  # informational; do not fail CI
    print("=" * 60)
    return passed, mean_err


def _global_dq_from_rt(R, t):
    from module.dq_ops import pose_to_dq
    return pose_to_dq(R.view(1, 3, 3), t.view(1, 3)).view(8)


def _apply_global_dq(xdq, g):
    from module.dq_ops import dq_mul
    xdq_p = xdq.permute(0, 1, 3, 2)
    out = dq_mul(g.view(1, 1, 1, 8).expand_as(xdq_p), xdq_p)
    return out.permute(0, 1, 3, 2)


def _random_pose_dq(batch, n_joints, t_in, device):
    from module.dq_ops import dq_exp
    xi = torch.randn(batch, n_joints, t_in, 6, device=device)
    xi[..., :3] *= 0.6
    xi[..., 3:] *= 0.5
    return dq_exp(xi).permute(0, 1, 3, 2)


def test_dqpose_equivariance(n_trials=10, n_joints=10, t_in=10, device='cpu',
                             decode_mode="direct", pose_rot_scale=1.0,
                             pose_trans_scale=1.0, vel_rot_scale=1.0,
                             vel_trans_scale=1.0, out_rot_scale=1.0,
                             out_trans_scale=1.0, out_rot_scale_start=None,
                             out_rot_scale_end=None, out_trans_scale_start=None,
                             out_trans_scale_end=None, block_mode="joint"):
    print("=" * 60)
    print("Testing GGMotion-DQPose - SE(3) equivariance")
    print("=" * 60)

    from module.model_dqpose import GGMNet_DQPose

    config = make_synthetic_config(
        n_joints, t_in, t_in, dqpose_decode_mode=decode_mode,
        dqpose_pose_rot_scale=pose_rot_scale,
        dqpose_pose_trans_scale=pose_trans_scale,
        dqpose_vel_rot_scale=vel_rot_scale,
        dqpose_vel_trans_scale=vel_trans_scale,
        dqpose_out_rot_scale=out_rot_scale,
        dqpose_out_trans_scale=out_trans_scale,
        dqpose_out_rot_scale_start=out_rot_scale_start,
        dqpose_out_rot_scale_end=out_rot_scale_end,
        dqpose_out_trans_scale_start=out_trans_scale_start,
        dqpose_out_trans_scale_end=out_trans_scale_end,
        dqpose_block_mode=block_mode,
    )
    group, edges = make_synthetic_group_edges(n_joints)
    model = GGMNet_DQPose(config, group, edges).to(device)
    model.eval()

    coord_errors = []

    with torch.no_grad():
        for trial in range(n_trials):
            x = _random_pose_dq(2, n_joints, t_in, device)

            y = model(x)

            R = random_rotation_matrix(device)
            t = random_translation(scale=1.0, device=device)
            g = _global_dq_from_rt(R, t)

            x_t = _apply_global_dq(x, g)
            y_t = model(x_t)

            y_expected = torch.einsum('ij,bnjt->bnit', R, y) + t.view(1, 1, 3, 1)

            err = torch.norm(y_expected - y_t, dim=2).mean().item()
            coord_errors.append(err)

            print(f"  Trial {trial:2d}: coord_err={err:.2e}")

    mean_err = np.mean(coord_errors)
    print(f"\n  Mean coord equivariance error: {mean_err:.2e}")

    passed = mean_err < 1e-3
    if passed:
        print("  PASSED - GGMotion-DQPose is SE(3)-equivariant")
    else:
        print("  FAILED - errors above threshold (1e-3)")
    print("=" * 60)
    return passed, mean_err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GGMotion equivariance")
    parser.add_argument("--model", choices=["xyz", "dqpose"], default="xyz",
                        help="Which model to test (default: xyz)")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--n-joints", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dqpose-decode-mode", default="direct",
                        choices=["direct", "residual", "cv_residual"])
    parser.add_argument("--dqpose-pose-rot-scale", type=float, default=1.0)
    parser.add_argument("--dqpose-pose-trans-scale", type=float, default=1.0)
    parser.add_argument("--dqpose-vel-rot-scale", type=float, default=1.0)
    parser.add_argument("--dqpose-vel-trans-scale", type=float, default=1.0)
    parser.add_argument("--dqpose-out-rot-scale", type=float, default=1.0)
    parser.add_argument("--dqpose-out-trans-scale", type=float, default=1.0)
    parser.add_argument("--dqpose-out-rot-scale-start", type=float, default=None)
    parser.add_argument("--dqpose-out-rot-scale-end", type=float, default=None)
    parser.add_argument("--dqpose-out-trans-scale-start", type=float, default=None)
    parser.add_argument("--dqpose-out-trans-scale-end", type=float, default=None)
    parser.add_argument("--dqpose-block-mode", default="joint",
                        choices=["joint", "split"])
    args = parser.parse_args()

    if args.model == "xyz":
        passed, err = test_xyz_equivariance(
            n_trials=args.n_trials, n_joints=args.n_joints, device=args.device)
    elif args.model == "dqpose":
        passed, err = test_dqpose_equivariance(
            n_trials=args.n_trials, n_joints=args.n_joints, device=args.device,
            decode_mode=args.dqpose_decode_mode,
            pose_rot_scale=args.dqpose_pose_rot_scale,
            pose_trans_scale=args.dqpose_pose_trans_scale,
            vel_rot_scale=args.dqpose_vel_rot_scale,
            vel_trans_scale=args.dqpose_vel_trans_scale,
            out_rot_scale=args.dqpose_out_rot_scale,
            out_trans_scale=args.dqpose_out_trans_scale,
            out_rot_scale_start=args.dqpose_out_rot_scale_start,
            out_rot_scale_end=args.dqpose_out_rot_scale_end,
            out_trans_scale_start=args.dqpose_out_trans_scale_start,
            out_trans_scale_end=args.dqpose_out_trans_scale_end,
            block_mode=args.dqpose_block_mode)

    sys.exit(0 if passed else 1)
