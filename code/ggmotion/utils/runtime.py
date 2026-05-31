import os
import re
from datetime import datetime
import numpy as np
import torch
import shutil
from tqdm import tqdm
from torch import nn, optim
from utils import viz
from module.dq_ops import dq_relative, dq_log, dq_project

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, config, model, parent):
        self.loss_type = config.loss_type
        self.aux_loss = config.aux_loss
        self.loss_mode = getattr(config, "loss_mode", "standard")
        self.position_loss_weight = getattr(config, "position_loss_weight", 1.0)
        self.aux_loss_weight = getattr(config, "aux_loss_weight", 1.0)
        self.pose_loss = getattr(config, "pose_loss", False)
        self.pose_loss_weight = getattr(config, "pose_loss_weight", 0.0)
        self.pose_rot_weight = getattr(config, "pose_rot_weight", 1.0)
        self.pose_trans_weight = getattr(config, "pose_trans_weight", 1.0)
        self.pose_velocity_loss_weight = getattr(config, "pose_velocity_loss_weight", 0.0)
        self.pose_loss_start_epoch = getattr(config, "pose_loss_start_epoch", 1)
        self.pose_joint_weighting = getattr(config, "pose_joint_weighting", "none")
        self.pose_joint_weights = build_pose_joint_weights(parent, self.pose_joint_weighting)
        self.parent = parent
        self.optimizer = optim.Adam(model, lr=config.lr, weight_decay=config.weight_decay)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=config.milestone,
                                                           gamma=config.gamma)

    def epoch(self, model, epoch, loader):
        res = {'pri_loss': 0, 'aux_loss': 0, 'pose_loss': 0, 'counter': 0}
        model.train()
        for batch_data in tqdm(loader, desc='train epoch'):
            input_seq, output_seq, all_seq, target_dq = unpack_batch(batch_data)
            batch, _, _ = all_seq.shape
            input_seq = input_seq.to(DEVICE)
            output_seq = output_seq.to(DEVICE)
            target_dq = target_dq.to(DEVICE) if target_dq is not None else None

            pred = model(input_seq)
            target = output_seq.clone()
            pri_loss = custom_loss(self.loss_type[0], pred, target)
            loss = self.position_loss_weight * pri_loss
            aux_loss = torch.zeros_like(pri_loss, device=pri_loss.device)
            if self.aux_loss:
                target_diff = target[:, self.parent[:, 1]] - target[:, self.parent[:, 0]]
                pred_diff = pred[:, self.parent[:, 1]] - pred[:, self.parent[:, 0]]
                aux_loss = custom_loss(self.loss_type[-1], pred_diff, target_diff)
                loss = loss + self.aux_loss_weight * aux_loss

            pose_loss = torch.zeros_like(pri_loss, device=pri_loss.device)
            if (self.pose_loss and epoch >= self.pose_loss_start_epoch
                    and target_dq is not None and hasattr(model, "last_pred_dq")):
                target_dq_p = target_dq.permute(0, 1, 3, 2)
                pose_loss, _, _ = dq_pose_loss(
                    model.last_pred_dq, target_dq_p,
                    rot_weight=self.pose_rot_weight,
                    trans_weight=self.pose_trans_weight,
                    joint_weight=self.pose_joint_weights,
                )
                if self.pose_velocity_loss_weight > 0:
                    prev_dq = input_seq[..., -1:].permute(0, 1, 3, 2)
                    pred_seq = torch.cat([prev_dq, model.last_pred_dq], dim=2)
                    target_seq = torch.cat([prev_dq, target_dq_p], dim=2)
                    pred_vel = dq_project(dq_relative(pred_seq[:, :, :-1, :], pred_seq[:, :, 1:, :]))
                    target_vel = dq_project(dq_relative(target_seq[:, :, :-1, :], target_seq[:, :, 1:, :]))
                    vel_loss, _, _ = dq_pose_loss(
                        pred_vel, target_vel,
                        rot_weight=self.pose_rot_weight,
                        trans_weight=self.pose_trans_weight,
                        joint_weight=self.pose_joint_weights,
                    )
                    pose_loss = pose_loss + self.pose_velocity_loss_weight * vel_loss
                loss = loss + self.pose_loss_weight * pose_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            self.optimizer.step()

            res['pri_loss'] += pri_loss.item() * batch
            res['aux_loss'] += aux_loss.item() * batch
            res['pose_loss'] += pose_loss.item() * batch
            res['counter'] += batch

        self.lr_scheduler.step()

        return (self.optimizer.param_groups[0]['lr'],
                res['pri_loss'] / res['counter'],
                res['aux_loss'] / res['counter'],
                res['pose_loss'] / res['counter'])


def unpack_batch(batch_data):
    if len(batch_data) == 3:
        input_seq, output_seq, all_seq = batch_data
        return input_seq, output_seq, all_seq, None
    if len(batch_data) == 4:
        input_seq, output_seq, all_seq, target_dq = batch_data
        return input_seq, output_seq, all_seq, target_dq
    raise ValueError(f"Unexpected batch length: {len(batch_data)}")


def build_pose_joint_weights(parent_edges, mode="none"):
    if mode != "subtree":
        return None
    parent_edges = np.asarray(parent_edges, dtype=np.int64)
    if parent_edges.ndim != 2 or parent_edges.shape[1] < 2:
        return None
    parent_edges = parent_edges[:, :2]
    n_nodes = int(parent_edges.max()) + 1
    children = [[] for _ in range(n_nodes)]
    for parent, child in parent_edges:
        children[int(parent)].append(int(child))

    counts = np.ones(n_nodes, dtype=np.float32)

    def visit(node):
        total = 1.0
        for child in children[node]:
            total += visit(child)
        counts[node] = total
        return total

    child_set = set(int(child) for _, child in parent_edges)
    roots = [node for node in range(n_nodes) if node not in child_set]
    for root in roots:
        visit(root)
    counts = counts / counts.mean()
    return torch.from_numpy(counts).to(DEVICE)


def dq_pose_loss(pred_dq, target_dq, rot_weight=1.0, trans_weight=1.0, joint_weight=None):
    err = dq_log(dq_project(dq_relative(pred_dq, target_dq)))
    rot_err = torch.norm(err[..., :3], 2, -1)
    trans_err = torch.norm(err[..., 3:], 2, -1)
    if joint_weight is not None and joint_weight.numel() == err.shape[1]:
        weight = joint_weight.to(device=err.device, dtype=err.dtype).view(1, -1, 1)
        rot_err = rot_err * weight
        trans_err = trans_err * weight
    rot_loss = torch.mean(rot_err)
    trans_loss = torch.mean(trans_err)
    loss = rot_weight * rot_loss + trans_weight * trans_loss
    return loss, rot_loss, trans_loss


def custom_loss(loss_type, pred, target):  # (B,N,3,T)
    if loss_type == "L1":
        loss = torch.mean(torch.norm(target - pred, 1, 2))
    elif loss_type == "L2":
        loss = torch.mean(torch.norm(target - pred, 2, 2))
    else:
        raise NotImplementedError(loss_type)

    return loss


def test(model, eval_frame, action, loader, dim_used, scale):
    eval_msg = ""
    header = f"{'milliseconds':<18} | " + " | ".join(f"{ms[1]:^7d}" for ms in eval_frame) + f" | {'avg':^7} |"
    eval_msg = eval_msg + header
    avg_mpjpe = np.zeros(len(eval_frame))
    model.eval()
    for act in action:
        res = {'loss': 0, 'counter': 0}
        act_mpjpe = np.zeros(len(eval_frame))
        with torch.no_grad():
            for batch_data in loader[act]:
                input_seq, output_seq, all_seq, _ = unpack_batch(batch_data)
                batch_size, _, _, pred_len = output_seq.shape
                all_seq = all_seq.to(DEVICE)
                input_seq = input_seq.to(DEVICE)

                outputs = model(input_seq)  # (B,N,3,T)
                outputs = outputs.permute(0, 3, 1, 2).flatten(2)  # (B,T,N*3)

                joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
                index_to_ignore = np.concatenate(
                    (joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
                joint_equal = np.array([13, 19, 22, 13, 27, 30])
                index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

                pred_3d = all_seq[:, -pred_len:].clone()
                targ_p3d = all_seq[:, -pred_len:].clone()
                pred_3d[:, :, dim_used] = outputs
                pred_3d[:, :, index_to_ignore] = pred_3d[:, :, index_to_equal]
                pred_p3d = pred_3d.contiguous().view(batch_size, pred_len, -1, 3)
                targ_p3d = targ_p3d.contiguous().view(batch_size, pred_len, -1, 3)

                for k in np.arange(0, len(eval_frame)):
                    j = eval_frame[k][0]
                    act_mpjpe[k] += torch.mean(torch.norm(
                        targ_p3d[:, j, :, :].contiguous().view(-1, 3) - pred_p3d[:, j, :, :].contiguous().view(-1, 3),
                        2,
                        1)).item() * batch_size

                res['counter'] += batch_size

        act_mpjpe = act_mpjpe * scale / res['counter']
        avg_mpjpe += act_mpjpe

        value = f"{act:<18} | " + " | ".join(
            f'{float(mpjpe):^7.3f}' for mpjpe in act_mpjpe) + f" | {float(act_mpjpe.mean()):^7.3f} |"
        eval_msg = eval_msg + '\n' + value

    avg_mpjpe = avg_mpjpe / len(action)
    avg_avg_mpjpe = avg_mpjpe.mean()
    value = f"{'avg':<18} | " + " | ".join(
        f'{float(mpjpe):^7.3f}' for mpjpe in avg_mpjpe) + f" | {float(avg_avg_mpjpe):^7.3f} |"
    eval_msg = eval_msg + '\n' + value

    return avg_mpjpe, avg_avg_mpjpe, eval_msg


def test_cmu(model, eval_frame, action, loader, dim_used, scale):
    eval_msg = ""
    header = f"{'milliseconds':<18} | " + " | ".join(f"{ms[1]:^7d}" for ms in eval_frame) + f" | {'avg':^7} |"
    eval_msg = eval_msg + header
    avg_mpjpe = np.zeros(len(eval_frame))
    model.eval()
    for act in action:
        res = {'loss': 0, 'counter': 0}
        act_mpjpe = np.zeros(len(eval_frame))
        with torch.no_grad():
            for batch_data in loader[act]:
                input_seq, output_seq, all_seq, _ = unpack_batch(batch_data)
                batch_size, _, _, pred_len = output_seq.shape
                all_seq = all_seq.to(DEVICE)
                input_seq = input_seq.to(DEVICE)

                outputs = model(input_seq)  # (B,N,3,T)
                outputs = outputs.permute(0, 3, 1, 2).flatten(2)  # (B,T,N*3)

                joint_to_ignore = np.array([16, 20, 29, 24, 27, 33, 36])
                index_to_ignore = np.concatenate(
                    (joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
                joint_equal = np.array([15, 15, 15, 23, 23, 32, 32])
                index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

                pred_3d = all_seq[:, -pred_len:].clone()
                targ_p3d = all_seq[:, -pred_len:].clone()
                pred_3d[:, :, dim_used] = outputs
                pred_3d[:, :, index_to_ignore] = pred_3d[:, :, index_to_equal]
                pred_p3d = pred_3d.contiguous().view(batch_size, pred_len, -1, 3)
                targ_p3d = targ_p3d.contiguous().view(batch_size, pred_len, -1, 3)

                for k in np.arange(0, len(eval_frame)):
                    j = eval_frame[k][0]
                    act_mpjpe[k] += torch.mean(torch.norm(
                        targ_p3d[:, j, :, :].contiguous().view(-1, 3) - pred_p3d[:, j, :, :].contiguous().view(-1, 3),
                        2,
                        1)).item() * batch_size

                res['counter'] += batch_size

        act_mpjpe = act_mpjpe * scale / res['counter']
        avg_mpjpe += act_mpjpe

        value = f"{act:<18} | " + " | ".join(
            f'{float(mpjpe):^7.3f}' for mpjpe in act_mpjpe) + f" | {float(act_mpjpe.mean()):^7.3f} |"
        eval_msg = eval_msg + '\n' + value

    avg_mpjpe = avg_mpjpe / len(action)
    avg_avg_mpjpe = avg_mpjpe.mean()
    value = f"{'avg':<18} | " + " | ".join(
        f'{float(mpjpe):^7.3f}' for mpjpe in avg_mpjpe) + f" | {float(avg_avg_mpjpe):^7.3f} |"
    eval_msg = eval_msg + '\n' + value

    return avg_mpjpe, avg_avg_mpjpe, eval_msg


def test_3dpw(model, eval_frame, action, loader, dim_used, scale):
    eval_msg = ""
    header = f"{'milliseconds':<18} | " + " | ".join(f"{ms[1]:^7d}" for ms in eval_frame) + f" | {'avg':^7} |"
    eval_msg = eval_msg + header
    avg_mpjpe = np.zeros(len(eval_frame))
    model.eval()
    for act in action:
        res = {'loss': 0, 'counter': 0}
        act_mpjpe = np.zeros(len(eval_frame))
        with torch.no_grad():
            for batch_data in loader[act]:
                input_seq, output_seq, all_seq, _ = unpack_batch(batch_data)
                batch_size, _, _, pred_len = output_seq.shape
                all_seq = all_seq.to(DEVICE)
                input_seq = input_seq.to(DEVICE)

                outputs = model(input_seq)  # (B,N,3,T)
                outputs = outputs.permute(0, 3, 1, 2).flatten(2)  # (B,T,N*3)

                pred_3d = all_seq[:, -pred_len:].clone()
                targ_p3d = all_seq[:, -pred_len:].clone()
                pred_3d[:, :, dim_used] = outputs
                pred_p3d = pred_3d.contiguous().view(batch_size, pred_len, -1, 3)
                targ_p3d = targ_p3d.contiguous().view(batch_size, pred_len, -1, 3)

                for k in np.arange(0, len(eval_frame)):
                    j = eval_frame[k][0]
                    act_mpjpe[k] += torch.mean(torch.norm(
                        targ_p3d[:, j, :, :].contiguous().view(-1, 3) - pred_p3d[:, j, :, :].contiguous().view(-1, 3),
                        2,
                        1)).item() * batch_size

                res['counter'] += batch_size

        act_mpjpe = act_mpjpe * scale / res['counter']
        avg_mpjpe += act_mpjpe

        value = f"{act:<18} | " + " | ".join(
            f'{float(mpjpe):^7.3f}' for mpjpe in act_mpjpe) + f" | {float(act_mpjpe.mean()):^7.3f} |"
        eval_msg = eval_msg + '\n' + value

    avg_mpjpe = avg_mpjpe / len(action)
    avg_avg_mpjpe = avg_mpjpe.mean()
    value = f"{'avg':<18} | " + " | ".join(
        f'{float(mpjpe):^7.3f}' for mpjpe in avg_mpjpe) + f" | {float(avg_avg_mpjpe):^7.3f} |"
    eval_msg = eval_msg + '\n' + value

    return avg_mpjpe, avg_avg_mpjpe, eval_msg


def viz_test(model, eval_frame, act, seq, loader, dim_used, save_dir, exp, pre_len=2, scale=1):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        seq_start = 1
        for batch_data in loader[act]:
            input_seq, output_seq, all_seq, _ = unpack_batch(batch_data)
            batch_size, _, _, pred_len = output_seq.shape
            if not (seq_start <= seq < seq_start + batch_size):
                seq_start += batch_size
                continue

            all_seq = all_seq.to(DEVICE)
            input_seq = input_seq.to(DEVICE)

            outputs = model(input_seq)  # (B,N,3,T)
            outputs = outputs.permute(0, 3, 1, 2).flatten(2)  # (B,T,N*3)

            joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
            index_to_ignore = np.concatenate(
                (joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
            joint_equal = np.array([13, 19, 22, 13, 27, 30])
            index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

            pred_3d = all_seq[:, -pred_len:].clone()
            targ_p3d = all_seq[:, -pred_len:].clone()
            pred_3d[:, :, dim_used] = outputs
            pred_3d[:, :, index_to_ignore] = pred_3d[:, :, index_to_equal]
            pred_p3d = pred_3d.cpu().data.numpy()
            targ_p3d = targ_p3d.cpu().data.numpy()
            select = [ms[0] for ms in eval_frame]

            gt_pre = all_seq[:, -pred_len - pre_len: -pred_len].clone()
            gt_pre = gt_pre.cpu().data.numpy()

            title = f'data:{exp} action:{act} seq:{seq}'
            save_path = os.path.join(save_dir, f'{exp}_{act}_{seq}_{pred_len}')
            viz.plot_predictions(targ_p3d[seq - seq_start] * scale, pred_p3d[seq - seq_start] * scale, gt_pre[seq - seq_start] * scale,
                                 title, save_path, select)

            return 'viz done'

        return 'seq is out of range'


def viz_test_cmu(model, eval_frame, act, seq, loader, dim_used, save_dir, exp, pre_len=2, scale=1):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        seq_start = 1
        for batch_data in loader[act]:
            input_seq, output_seq, all_seq, _ = unpack_batch(batch_data)
            batch_size, _, _, pred_len = output_seq.shape
            if not (seq_start <= seq < seq_start + batch_size):
                seq_start += batch_size
                continue

            all_seq = all_seq.to(DEVICE)
            input_seq = input_seq.to(DEVICE)

            outputs = model(input_seq)  # (B,N,3,T)
            outputs = outputs.permute(0, 3, 1, 2).flatten(2)  # (B,T,N*3)

            joint_to_ignore = np.array([16, 20, 29, 24, 27, 33, 36])
            index_to_ignore = np.concatenate(
                (joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
            joint_equal = np.array([15, 15, 15, 23, 23, 32, 32])
            index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

            pred_3d = all_seq[:, -pred_len:].clone()
            targ_p3d = all_seq[:, -pred_len:].clone()
            pred_3d[:, :, dim_used] = outputs
            pred_3d[:, :, index_to_ignore] = pred_3d[:, :, index_to_equal]
            pred_p3d = pred_3d.cpu().data.numpy()
            targ_p3d = targ_p3d.cpu().data.numpy()
            select = [ms[0] for ms in eval_frame]

            gt_pre = all_seq[:, -pred_len - pre_len: -pred_len].clone()
            gt_pre = gt_pre.cpu().data.numpy()

            title = f'data:{exp} action:{act} seq:{seq}'
            save_path = os.path.join(save_dir, f'{exp}_{act}_{seq}_{pred_len}')
            viz.plot_predictions_cmu(targ_p3d[seq - seq_start] * scale, pred_p3d[seq - seq_start] * scale, gt_pre[seq - seq_start] * scale, title, save_path, select)

            return 'viz done'

        return 'seq is out of range'


def save_ckpt(work_dir, exp_name, state, epoch, err, num):
    """Save checkpoint if it's in the top-N by error. Returns saved path or None."""
    files = os.listdir(work_dir)
    file_info = []

    for file in files:
        if file.startswith(f"{exp_name}_best_") and file.endswith(".pt"):
            match = re.search(rf'{exp_name}_best_([\d.]+)_(\d+)\.pt$', file)
            if match:
                last_err = float(match.group(1))
                last_epoch = int(match.group(2))
                file_path = os.path.join(work_dir, file)
                file_info.append((last_err, last_epoch, file_path))

    file_info.sort(reverse=True, key=lambda x: x[0])
    file_path = os.path.join(work_dir, f'{exp_name}_best_{err:.3f}_{epoch}.pt')
    if len(file_info) < num:
        torch.save(state, file_path)
        return file_path
    elif err <= file_info[0][0]:
        to_remove = file_info[0][2]
        os.remove(to_remove)
        torch.save(state, file_path)
        return file_path
    return None


def exp_create(exp_dir, exp_name, yml, model_file="./module/model.py"):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    work_dir = os.path.join(exp_dir, now)
    os.makedirs(work_dir, exist_ok=True)
    yml_back_up = os.path.join(work_dir, exp_name + '_backup.yml')
    shutil.copy(yml, yml_back_up)
    code_back_up = os.path.join(work_dir, exp_name + '_backup.py')
    shutil.copy(model_file, code_back_up)
    run_back_up = os.path.join(work_dir, exp_name + f'_run_backup.py')
    shutil.copy(f"./utils/runtime.py", run_back_up)
    log = os.path.join(work_dir, exp_name + '_eval.log')
    log = open(log, 'a')
    return work_dir, log
