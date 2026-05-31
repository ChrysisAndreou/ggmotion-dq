"""
Standalone evaluation - MPJPE in mm at standard horizons.

CMU 8-action protocol (Li 2018 / Mao 2019):
    - 8 actions, 25-joint subset, input 10 frames at 25fps
    - Evaluate at 80, 160, 320, 400 ms (frames [1,3,7,9])
    - Metric: MPJPE in mm on 38-joint reconstructed skeleton

Both models share the GGMotion backbone (ESTAG spatial-temporal alternation,
linear decoder). They differ only in geometric representation:
    GGMotion-XYZ - Cartesian R^3 coordinate updates
    GGMotion-DQPose - FK-derived pose DQs with a DQ log/exp boundary

Usage:
    cd ggmotion/code

    # Evaluate GGMotion-DQPose
    python evaluate.py --model dqpose --cfg cfg/cmu_dqpose_mindirect_short.yml \
        --ckpt exp/dqposemindirect_retrained/cmu_best_24.595_22.pt

    # Evaluate GGMotion-XYZ
    python evaluate.py --model xyz --ckpt exp/xyz_reproduced/cmu_best_20.823_19.pt

    # Single action
    python evaluate.py --model dqpose --cfg cfg/cmu_dqpose_mindirect_short.yml \
        --ckpt exp/dqposemindirect_retrained/cmu_best_24.595_22.pt --act walking
"""

import argparse
import torch
import yaml
from torch.utils.data import DataLoader

from utils.data_loader import CMU_Motion3D, CMU_Motion3DDQPose
from utils.data_utils import setup_seed
from utils import runtime
from module.model import GGMNet
from module.model_dqpose import GGMNet_DQPose

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    setup_seed(args.seed)
    if args.model == 'dqpose':
        dataset_cls = CMU_Motion3DDQPose
    else:
        dataset_cls = CMU_Motion3D
    eval_actions = CMU_Motion3D.define_acts(args.act)

    loaders_test = {}
    for act in eval_actions:
        dataset_test = dataset_cls(
            args.data, actions=act, input_n=args.past_length,
            output_n=args.future_length, split=1, scale=args.scale,
            test_manner=args.manner, hop=args.n_hop,
        )
        loaders_test[act] = DataLoader(
            dataset_test, batch_size=args.batch_size, shuffle=False,
            drop_last=False, num_workers=0, pin_memory=True,
        )

    if args.model == 'dqpose':
        model = GGMNet_DQPose(args, dataset_test.group, dataset_test.edges).to(DEVICE)
    else:
        model = GGMNet(args, dataset_test.group, dataset_test.edges).to(DEVICE)

    labels = {'xyz': 'XYZ', 'dqpose': 'DQPose'}
    print(f"Model: GGMotion-{labels[args.model]} "
          f"({sum(p.numel() for p in model.parameters()) / 1e6:.3f}M params)")
    print(f"Checkpoint: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    print(f"Loaded (epoch {ckpt['epoch']}, pri_loss {ckpt['pri_loss']:.4f}, "
          f"aux_loss {ckpt['aux_loss']:.4f})")

    eval_frame = [(1, 80), (3, 160), (7, 320), (9, 400)]
    _, _, eval_msg = runtime.test_cmu(
        model, eval_frame, eval_actions,
        loaders_test, dataset_test.dim_used, args.scale,
    )
    print(eval_msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GGMotion standalone evaluation')
    parser.add_argument('--model', type=str, required=True, choices=['xyz', 'dqpose'],
                        help='model variant: xyz or dqpose')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='path to checkpoint (.pt)')
    parser.add_argument('--cfg', type=str, default='cfg/cmu_short.yml',
                        help='path to config YAML')
    parser.add_argument('--data', type=str, default='./data/cmu',
                        help='path to CMU dataset')
    parser.add_argument('--act', type=str, default='all',
                        help='action to evaluate (default: all 8)')
    parser.add_argument('--manner', type=str, default='all',
                        help='test manner: all, 256, or 8')
    parser.add_argument('--seed', type=int, default=1234567890)

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        yml_arg = yaml.load(f, Loader=yaml.FullLoader)
    parser.set_defaults(**yml_arg)
    args = parser.parse_args()

    main(args)
