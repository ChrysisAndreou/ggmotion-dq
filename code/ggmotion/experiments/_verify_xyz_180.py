"""Quick sanity check: run eval_perturbed exactly as robustness_exp.py would,
for XYZ at 180° only, and compare against eval_per_action result."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from utils.data_loader import CMU_Motion3D
from utils.data_utils import setup_seed
from module.model import GGMNet

from experiments.robustness_exp import (
    eval_perturbed, axis_angle_to_rotmat, apply_rotation,
)


def main():
    device = torch.device('cpu')
    cfg_path = _ROOT / 'cfg/cmu_short.yml'
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    class Config: pass
    config = Config()
    for k, v in cfg.items():
        setattr(config, k, v)

    setup_seed(1234567890)

    loaders_test = {}
    dataset_ref = None
    for act in CMU_Motion3D.define_acts('all'):
        ds = CMU_Motion3D(
            './data/cmu', actions=act,
            input_n=config.past_length, output_n=config.future_length,
            split=1, scale=config.scale, test_manner='all',
            hop=getattr(config, 'n_hop', -1))
        loaders_test[act] = DataLoader(
            ds, batch_size=config.batch_size,
            shuffle=False, drop_last=False, num_workers=0, pin_memory=True)
        dataset_ref = ds
    print(f'Loaded test set', flush=True)

    model_xyz = GGMNet(config, dataset_ref.group, dataset_ref.edges).to(device)
    ckpt = torch.load('exp/xyz_reproduced/cmu_best_20.823_19.pt',
                      map_location=device, weights_only=False)
    model_xyz.load_state_dict(ckpt['state_dict'])
    print(f'Loaded XYZ ckpt epoch={ckpt["epoch"]}', flush=True)

    torch.manual_seed(42)
    np.random.seed(42)
    axis = torch.randn(3, device=device)
    axis = axis / axis.norm()
    print(f'Axis: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]', flush=True)

    angle_rad = torch.tensor(180.0 * np.pi / 180.0, device=device, dtype=torch.float32)
    R = axis_angle_to_rotmat(axis, angle_rad)
    def transform_fn(inp, all_s, _R=R):
        return apply_rotation(_R, inp, all_s)

    print('\nRunning eval_perturbed (robustness_exp version) at 180°...', flush=True)
    res = eval_perturbed(model_xyz, loaders_test, dataset_ref.dim_used,
                         config.scale, device, transform_fn)
    print(f'eval_perturbed result @180° XYZ: {res}', flush=True)


if __name__ == '__main__':
    main()
