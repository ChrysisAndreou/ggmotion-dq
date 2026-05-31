"""Verify CMU pose-DQ preprocessing invariants.

Checks:
  - pose DQs satisfy the unit-dual-quaternion constraints,
  - rotation channels are nonzero on real CMU data,
  - DQ translations reconstruct the canonical position stream.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module.dq_ops import dq_to_pos  # noqa: E402
from utils.data_loader import CMU_Motion3DDQPose  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data/cmu")
    parser.add_argument("--act", default="walking")
    parser.add_argument("--manner", default="8")
    parser.add_argument("--scale", type=int, default=1000)
    parser.add_argument("--past_length", type=int, default=10)
    parser.add_argument("--future_length", type=int, default=10)
    args = parser.parse_args()

    dataset = CMU_Motion3DDQPose(
        args.data, actions=args.act, input_n=args.past_length,
        output_n=args.future_length, split=1, scale=args.scale,
        test_manner=args.manner,
    )

    input_dq, output_pos, all_pos, output_dq = dataset[0]
    dq = torch.from_numpy(dataset.used_seqs[: min(len(dataset), 8)]).float()
    pos = torch.from_numpy(dataset.pos_seqs[: min(len(dataset), 8)]).float()
    dq_p = dq.permute(0, 1, 3, 2)

    real = dq_p[..., :4]
    dual = dq_p[..., 4:]
    real_norm_err = torch.max(torch.abs(torch.linalg.norm(real, dim=-1) - 1.0)).item()
    orth_err = torch.max(torch.abs((real * dual).sum(dim=-1))).item()
    rot_max = torch.max(torch.abs(real[..., 1:])).item()
    pos_recon = dq_to_pos(dq_p).permute(0, 1, 3, 2)
    pos_err = torch.max(torch.abs(pos_recon - pos)).item()

    print(f"dataset_len={len(dataset)}")
    print(f"input_dq_shape={input_dq.shape} output_pos_shape={output_pos.shape} output_dq_shape={output_dq.shape}")
    print(f"real_norm_err={real_norm_err:.3e}")
    print(f"real_dual_orthogonality_err={orth_err:.3e}")
    print(f"rotation_imag_max={rot_max:.3e}")
    print(f"position_reconstruction_err_scaled={pos_err:.3e}")

    ok = real_norm_err < 1e-4 and orth_err < 1e-4 and rot_max > 1e-4 and pos_err < 1e-5
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
