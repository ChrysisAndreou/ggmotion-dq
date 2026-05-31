"""One-step smoke test for the DQPose training/evaluation path."""

import argparse
from pathlib import Path
import sys

import torch
import yaml
from torch.utils.data import DataLoader, Subset

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from module.model_dqpose import GGMNet_DQPose  # noqa: E402
from utils import runtime  # noqa: E402
from utils.data_loader import CMU_Motion3DDQPose  # noqa: E402
from utils.data_utils import setup_seed  # noqa: E402
from utils.runtime import Trainer  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    class Config:
        pass

    config = Config()
    for key, value in cfg.items():
        setattr(config, key, value)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="cfg/cmu_dqpose_mindirect_short.yml")
    parser.add_argument("--data", default="./data/cmu")
    parser.add_argument("--act", default="walking")
    parser.add_argument("--manner", default="8")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = load_config(args.cfg)
    config.batch_size = args.batch_size
    setup_seed(1234567890)

    print(f"device={DEVICE}")
    print("loading train subset")
    train_dataset = CMU_Motion3DDQPose(
        args.data, actions=args.act, input_n=config.past_length,
        output_n=config.future_length, split=0, scale=config.scale,
        aug=False, hop=getattr(config, "n_hop", -1),
    )
    smoke_len = min(len(train_dataset), args.batch_size)
    train_loader = DataLoader(
        Subset(train_dataset, list(range(smoke_len))), batch_size=args.batch_size,
        shuffle=False, drop_last=False, num_workers=0,
    )

    print("loading eval subset")
    test_dataset = CMU_Motion3DDQPose(
        args.data, actions=args.act, input_n=config.past_length,
        output_n=config.future_length, split=1, scale=config.scale,
        test_manner=args.manner, hop=getattr(config, "n_hop", -1),
    )
    test_loader = {
        args.act: DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            drop_last=False, num_workers=0,
        )
    }

    model = GGMNet_DQPose(config, train_dataset.group, train_dataset.edges).to(DEVICE)
    trainer = Trainer(config, model.parameters(), train_dataset.parent)
    lr, pri_loss, aux_loss, pose_loss = trainer.epoch(model, 1, train_loader)
    print(
        f"one_step lr={lr:.7f} pri_loss={pri_loss:.7f} "
        f"aux_loss={aux_loss:.7f} pose_loss={pose_loss:.7f}"
    )

    eval_frame = [(1, 80), (3, 160), (7, 320), (9, 400)]
    _, avg_avg_mpjpe, eval_msg = runtime.test_cmu(
        model, eval_frame, [args.act], test_loader, test_dataset.dim_used, config.scale,
    )
    print(eval_msg)
    print(f"smoke_eval_avg={avg_avg_mpjpe:.3f}")


if __name__ == "__main__":
    main()
