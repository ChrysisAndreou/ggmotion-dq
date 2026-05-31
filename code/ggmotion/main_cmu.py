import argparse
import torch
from utils.data_loader import CMU_Motion3D, CMU_Motion3DDQPose
from utils.data_utils import setup_seed
from torch.utils.data import DataLoader
from utils import runtime
from utils.runtime import Trainer
from module.model import GGMNet
from module.model_dqpose import GGMNet_DQPose
from datetime import datetime
import numpy as np
import yaml
import os
import wandb

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run():
    print('>>> set seed:', args.seed)
    setup_seed(args.seed)
    if args.model == 'dqpose':
        dataset_cls = CMU_Motion3DDQPose
    else:
        dataset_cls = CMU_Motion3D
    eval_actions = CMU_Motion3D.define_acts(args.act)
    print('>>> loading dataset_test')
    loaders_test = {}
    len_test = 0
    for act in eval_actions:
        dataset_test = dataset_cls(args.data, actions=act, input_n=args.past_length, output_n=args.future_length,
                                   split=1, scale=args.scale, test_manner=args.manner, hop=args.n_hop,
                                   )
        loaders_test[act] = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                       num_workers=0, pin_memory=True)
        len_test += dataset_test.__len__()
    print('>>> Testing dataset length: {:d}'.format(len_test))

    print('>>> create model ({})'.format(args.model))
    if args.model == 'dqpose':
        model = GGMNet_DQPose(args, dataset_test.group, dataset_test.edges).to(DEVICE)
    else:
        model = GGMNet(args, dataset_test.group, dataset_test.edges).to(DEVICE)
    print(">>> total params: {:.3f} M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    if args.mode == 'train':

        print('>>> loading dataset_train')
        dataset_train = dataset_cls(args.data, actions="all", input_n=args.past_length, output_n=args.future_length,
                                    split=0, scale=args.scale, aug=args.aug, hop=args.n_hop,
                                    )
        loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=0, pin_memory=True)
        print('>>> Training dataset length: {:d}'.format(dataset_train.__len__()))

        train = Trainer(args, model.parameters(), dataset_test.parent)
        model_backup = {
            "xyz": "./module/model.py",
            "dqpose": "./module/model_dqpose.py",
        }[args.model]
        work_dir, log = runtime.exp_create(args.exp, CMU_Motion3D.exp, args.cfg, model_backup)

        model_tag = getattr(args, "variant_name", args.model)
        wandb.init(
            project="ggmotion",
            name=f"ggmotion-{model_tag}-cmu-{datetime.now().strftime('%m%d-%H%M')}",
            config=vars(args),
            tags=["ggmotion", args.model, model_tag, "cmu"],
        )
        wandb.watch(model, log="gradients", log_freq=50)

        best_avg_mpjpe = float("inf")
        bad_eval_count = 0
        early_stop_start = getattr(args, "early_stop_start_epoch", 0)
        early_stop_patience = getattr(args, "early_stop_patience", 0)
        early_stop_min_delta = getattr(args, "early_stop_min_delta", 0.0)
        early_abort_epoch = getattr(args, "early_abort_epoch", 0)
        early_abort_avg = getattr(args, "early_abort_avg", None)
        early_abort_400 = getattr(args, "early_abort_400", None)

        for epoch in range(1, args.epochs + 1):
            lr, pri_loss, aux_loss, pose_loss = train.epoch(model, epoch, loader_train)
            print(f">>> train Epoch: {epoch} Lr: {lr:.7f} Pri_loss: {pri_loss:.7f}, "
                  f"Aux_loss: {aux_loss:.7f}, Pose_loss: {pose_loss:.7f}")
            wandb.log({"epoch": epoch, "lr": lr, "train/pri_loss": pri_loss,
                       "train/aux_loss": aux_loss, "train/pose_loss": pose_loss})

            if epoch % args.eval_interval == 0:
                if args.future_length > 12:
                    eval_frame = [(1, 80), (3, 160), (7, 320), (9, 400), (13, 560), (24, 1000)]
                else:
                    eval_frame = [(1, 80), (3, 160), (7, 320), (9, 400)]

                print(f">>> eval Epoch: {epoch} Lr: {lr:.7f} Pri_loss: {pri_loss:.7f}, Aux_loss: {aux_loss:.7f} Manner: {args.manner}", file=log, flush=True)
                avg_mpjpe, avg_avg_mpjpe, eval_msg = runtime.test_cmu(model, eval_frame, eval_actions,
                                                                      loaders_test, dataset_test.dim_used, args.scale)
                print(eval_msg)
                print(eval_msg, file=log, flush=True)

                eval_log = {"eval/avg_mpjpe_400ms": float(avg_mpjpe[-1]), "eval/avg_mpjpe_mean": float(avg_avg_mpjpe)}
                for i, (_, ms) in enumerate(eval_frame):
                    eval_log[f"eval/avg_mpjpe_{ms}ms"] = float(avg_mpjpe[i])
                wandb.log(eval_log)

                state = {'epoch': epoch,
                         'pri_loss': pri_loss,
                         'aux_loss': aux_loss,
                         'pose_loss': pose_loss,
                         'state_dict': model.state_dict(),
                         'optimizer': train.optimizer.state_dict()}
                saved_path = runtime.save_ckpt(work_dir, CMU_Motion3D.exp, state, epoch, avg_avg_mpjpe, args.keep)
                if saved_path:
                    model_tag = getattr(args, "variant_name", args.model)
                    artifact = wandb.Artifact(f"best-{model_tag}-cmu", type="model")
                    artifact.add_file(saved_path)
                    wandb.log_artifact(artifact)

                if avg_avg_mpjpe < best_avg_mpjpe - early_stop_min_delta:
                    best_avg_mpjpe = float(avg_avg_mpjpe)
                    bad_eval_count = 0
                else:
                    bad_eval_count += 1

                if early_abort_epoch and epoch >= early_abort_epoch:
                    abort_on_avg = early_abort_avg is not None and avg_avg_mpjpe > early_abort_avg
                    abort_on_400 = early_abort_400 is not None and avg_mpjpe[-1] > early_abort_400
                    if abort_on_avg and abort_on_400:
                        print(
                            f">>> early abort at epoch {epoch}: avg {avg_avg_mpjpe:.3f} "
                            f"> {early_abort_avg:.3f} and 400ms {avg_mpjpe[-1]:.3f} "
                            f"> {early_abort_400:.3f}"
                        )
                        break

                if (early_stop_start and early_stop_patience and epoch >= early_stop_start
                        and bad_eval_count >= early_stop_patience):
                    print(
                        f">>> early stop at epoch {epoch}: no avg improvement >= "
                        f"{early_stop_min_delta:.3f} for {bad_eval_count} evals; "
                        f"best avg {best_avg_mpjpe:.3f}"
                    )
                    break

        wandb.finish()
    elif args.mode == 'eval':
        print(">>> loading ckpt from '{}'".format(args.ckpt))
        ckpt = torch.load(args.ckpt, map_location=DEVICE)
        model.load_state_dict(ckpt['state_dict'])
        print(">>> ckpt loaded (epoch: {} | pri_loss: {} | aux_loss: {})".
              format(ckpt['epoch'], ckpt['pri_loss'], ckpt['aux_loss']))

        if args.future_length > 12:
            eval_frame = [(1, 80), (3, 160), (7, 320), (9, 400), (13, 560), (24, 1000)]
        else:
            eval_frame = [(1, 80), (3, 160), (7, 320), (9, 400)]

        _, _, eval_msg = runtime.test_cmu(model, eval_frame, eval_actions, loaders_test, dataset_test.dim_used, args.scale)
        print(eval_msg)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train && eval && infer options')
    parser.add_argument('--mode', type=str, default='train', help='train or eval')
    parser.add_argument('--data', type=str, default='./data/cmu', help='path to H36M dataset')
    parser.add_argument('--seed', type=int, default=1234567890, help='random seed')
    parser.add_argument('--cfg', type=str, default='cfg/cmu_short.yml', help='path to the configuration in .yml')
    parser.add_argument('--ckpt', type=str, default='./exp/test.pt', help='path to ckpt')
    parser.add_argument('--exp', type=str, default='./exp', help='dir to release experiment')
    parser.add_argument('--manner', type=str, default='all', help='all or 256 or 8')
    parser.add_argument('--viz', type=str, default='./demo', help='viz save path')
    parser.add_argument('--act', type=str, default='all', help='eval action')
    parser.add_argument('--seq', type=int, default=1, help='from 1 to manner')
    parser.add_argument('--model', type=str, default='xyz', choices=['xyz', 'dqpose'],
                        help='model variant: xyz or dqpose')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        yml_arg = yaml.load(f, Loader=yaml.FullLoader)

    parser.set_defaults(**yml_arg)
    args = parser.parse_args()
    run()
