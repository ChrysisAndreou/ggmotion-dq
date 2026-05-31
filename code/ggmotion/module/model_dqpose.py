"""GGMotion-DQPose: pose-level dual-quaternion motion predictor."""

import torch
from torch import nn

from module.dq_ops import dq_to_pos, dq_relative, dq_mul, dq_exp, dq_log, dq_project
from module.model import Block as Block_XYZ
from module.dq_block import Block_DQ
from module.modules import cosine_embedding


class GGMNet_DQPose(nn.Module):
    def __init__(self, config, group, edges):
        super().__init__()
        self.group = group
        self.edges = torch.from_numpy(edges).long()
        self.act = nn.SiLU()
        self.key_n = config.key_point
        self.input_n = config.past_length
        self.output_n = config.future_length
        self.e_dim = config.e_dim
        self.h_dim = config.h_dim
        self.norm = config.norm
        self.eps = 1e-6
        self.fk = config.fk
        self.n_layer = config.n_layer
        self.reference_joint = getattr(config, "dqpose_reference_joint", 8)
        self.decode_mode = getattr(config, "dqpose_decode_mode", "direct")
        self.cv_scale = getattr(config, "dqpose_cv_scale", 1.0)
        self.cv_decay = getattr(config, "dqpose_cv_decay", 1.0)
        self.pose_rot_scale = getattr(config, "dqpose_pose_rot_scale", 1.0)
        self.pose_trans_scale = getattr(config, "dqpose_pose_trans_scale", 1.0)
        self.vel_rot_scale = getattr(config, "dqpose_vel_rot_scale", 1.0)
        self.vel_trans_scale = getattr(config, "dqpose_vel_trans_scale", 1.0)
        self.out_rot_scale = getattr(config, "dqpose_out_rot_scale", 1.0)
        self.out_trans_scale = getattr(config, "dqpose_out_trans_scale", 1.0)
        self.out_rot_scale_start = getattr(config, "dqpose_out_rot_scale_start", self.out_rot_scale)
        self.out_rot_scale_end = getattr(config, "dqpose_out_rot_scale_end", self.out_rot_scale)
        self.out_trans_scale_start = getattr(config, "dqpose_out_trans_scale_start", self.out_trans_scale)
        self.out_trans_scale_end = getattr(config, "dqpose_out_trans_scale_end", self.out_trans_scale)
        self.block_mode = getattr(config, "dqpose_block_mode", "joint")

        self.x_emb = nn.Linear(self.input_n, self.e_dim, bias=False)
        self.v_emb = nn.Linear(self.input_n, self.e_dim, bias=False)
        if self.block_mode == "split":
            self.x_rot_emb = nn.Linear(self.input_n, self.e_dim, bias=False)
            self.x_trans_emb = nn.Linear(self.input_n, self.e_dim, bias=False)
            self.v_rot_emb = nn.Linear(self.input_n, self.e_dim, bias=False)
            self.v_trans_emb = nn.Linear(self.input_n, self.e_dim, bias=False)
        elif self.block_mode != "joint":
            raise ValueError(f"Unsupported DQPose block mode: {self.block_mode}")

        self.blocks = nn.ModuleList(
            [Block_DQ(self.key_n, self.h_dim, self.e_dim, self.group,
                      self.edges, self.act, self.eps, self.norm, self.fk)
             for _ in range(self.n_layer)])
        if self.block_mode == "split":
            self.rot_blocks = nn.ModuleList(
                [Block_XYZ(self.key_n, self.h_dim, self.e_dim, self.group,
                           self.edges, self.act, self.eps, self.norm, self.fk)
                 for _ in range(self.n_layer)])
            self.trans_blocks = nn.ModuleList(
                [Block_XYZ(self.key_n, self.h_dim, self.e_dim, self.group,
                           self.edges, self.act, self.eps, self.norm, self.fk)
                 for _ in range(self.n_layer)])

        self.center_mlp = nn.ModuleList(
            [nn.Linear(self.e_dim, self.output_n, bias=False)
             for _ in range(self.n_layer)])
        if self.block_mode == "split":
            self.rot_center_mlp = nn.ModuleList(
                [nn.Linear(self.e_dim, self.output_n, bias=False)
                 for _ in range(self.n_layer)])
            self.trans_center_mlp = nn.ModuleList(
                [nn.Linear(self.e_dim, self.output_n, bias=False)
                 for _ in range(self.n_layer)])

        self.pro = nn.Linear(self.e_dim, self.output_n, bias=False)
        if self.block_mode == "split":
            self.pro_rot = nn.Linear(self.e_dim, self.output_n, bias=False)
            self.pro_trans = nn.Linear(self.e_dim, self.output_n, bias=False)
        if self.decode_mode in {"residual", "cv_residual"}:
            nn.init.zeros_(self.pro.weight)
            if self.block_mode == "split":
                nn.init.zeros_(self.pro_rot.weight)
                nn.init.zeros_(self.pro_trans.weight)
        self.last_pred_dq = None
        self.last_pred_twist = None
        self.last_base_dq = None

    def forward(self, xdq):  # (B, N, 8, T_in) pose DQs
        batch_size = xdq.shape[0]
        xdq_p = xdq.permute(0, 1, 3, 2)  # (B, N, T, 8)

        center_dq = xdq_p[:, self.reference_joint:self.reference_joint + 1, -1:, :]

        rel_dq = dq_project(dq_relative(center_dq.expand_as(xdq_p), xdq_p))
        x_twist = dq_log(rel_dq).permute(0, 1, 3, 2)  # (B, N, 6, T)
        x_twist = self._scale_twist(
            x_twist, self.pose_rot_scale, self.pose_trans_scale, coord_dim=2,
        )

        vel_dq = dq_relative(xdq_p[:, :, :-1, :], xdq_p[:, :, 1:, :])
        vel_twist = dq_log(vel_dq)
        vel_twist = torch.cat([vel_twist[:, :, :1, :], vel_twist], dim=2)
        vel_twist = vel_twist.permute(0, 1, 3, 2)
        vel_twist = self._scale_twist(
            vel_twist, self.vel_rot_scale, self.vel_trans_scale, coord_dim=2,
        )

        attr1 = cosine_embedding(self.edges[:, 2], self.h_dim).to(xdq.device) \
            .unsqueeze(0).expand(batch_size, -1, -1)
        attr2 = cosine_embedding(torch.arange(0, self.key_n), self.h_dim).to(xdq.device) \
            .unsqueeze(0).expand(batch_size, -1, -1)

        if self.block_mode == "split":
            out_twist = self._run_split_backbone(x_twist, vel_twist, attr1, attr2)
        else:
            out_twist = self._run_joint_backbone(x_twist, vel_twist, attr1, attr2)

        out_twist = self._unscale_twist(
            out_twist, self.pose_rot_scale, self.pose_trans_scale, coord_dim=2,
        )
        out_twist = self._scale_twist_schedule(
            out_twist,
            self.out_rot_scale_start, self.out_rot_scale_end,
            self.out_trans_scale_start, self.out_trans_scale_end,
            coord_dim=2,
        )
        out_twist_p = out_twist.permute(0, 1, 3, 2)
        out_rel_dq = dq_exp(out_twist_p)
        self.last_base_dq = None
        if self.decode_mode == "residual":
            base_rel = rel_dq[:, :, -1:, :].expand(-1, -1, self.output_n, -1)
            out_rel_dq = dq_project(dq_mul(base_rel, out_rel_dq))
            self.last_base_dq = base_rel
        elif self.decode_mode == "cv_residual":
            base_rel = self._constant_velocity_base(rel_dq)
            out_rel_dq = dq_project(dq_mul(base_rel, out_rel_dq))
            self.last_base_dq = base_rel
        elif self.decode_mode != "direct":
            raise ValueError(f"Unsupported DQPose decode mode: {self.decode_mode}")
        out_dq = dq_mul(
            center_dq.expand(batch_size, out_rel_dq.shape[1], out_rel_dq.shape[2], -1),
            out_rel_dq,
        )

        self.last_pred_twist = out_twist_p
        self.last_pred_dq = out_dq

        out_pos = dq_to_pos(out_dq).permute(0, 1, 3, 2)
        return out_pos

    @staticmethod
    def _scale_values(twist, rot_scale, trans_scale, coord_dim):
        coord_dim = coord_dim % twist.dim()
        shape = [1] * twist.dim()
        shape[coord_dim] = 6
        return twist.new_tensor(
            [rot_scale, rot_scale, rot_scale, trans_scale, trans_scale, trans_scale]
        ).view(*shape)

    def _scale_twist(self, twist, rot_scale, trans_scale, coord_dim):
        return twist * self._scale_values(twist, rot_scale, trans_scale, coord_dim)

    def _unscale_twist(self, twist, rot_scale, trans_scale, coord_dim):
        return twist / self._scale_values(twist, rot_scale, trans_scale, coord_dim)

    def _scale_twist_schedule(self, twist, rot_start, rot_end, trans_start, trans_end, coord_dim):
        if rot_start == rot_end and trans_start == trans_end:
            return self._scale_twist(twist, rot_start, trans_start, coord_dim)

        coord_dim = coord_dim % twist.dim()
        time_dim = twist.dim() - 1
        steps = twist.shape[time_dim]
        rot = torch.linspace(float(rot_start), float(rot_end), steps, device=twist.device, dtype=twist.dtype)
        trans = torch.linspace(float(trans_start), float(trans_end), steps, device=twist.device, dtype=twist.dtype)
        values = torch.stack([rot, rot, rot, trans, trans, trans], dim=0)
        shape = [1] * twist.dim()
        shape[coord_dim] = 6
        shape[time_dim] = steps
        return twist * values.view(*shape)

    def _run_joint_backbone(self, x_twist, vel_twist, attr1, attr2):
        x_center = torch.mean(x_twist, dim=(1, -1), keepdim=True)
        x = self.x_emb(x_twist - x_center) + x_center
        v = self.v_emb(vel_twist)

        for i, block in enumerate(self.blocks):
            x, v = block(x, v, attr1, attr2, x_center)
            x_center = self.center_mlp[i](x)
            x_center = torch.mean(x_center, dim=(1, -1), keepdim=True)

        if self.decode_mode in {"residual", "cv_residual"}:
            return self.pro(x - x_center)
        return self.pro(x - x_center) + x_center

    def _run_split_backbone(self, x_twist, vel_twist, attr1, attr2):
        x_rot, x_trans = x_twist[:, :, :3], x_twist[:, :, 3:]
        v_rot, v_trans = vel_twist[:, :, :3], vel_twist[:, :, 3:]

        rot_center = torch.mean(x_rot, dim=(1, -1), keepdim=True)
        trans_center = torch.mean(x_trans, dim=(1, -1), keepdim=True)
        x_rot = self.x_rot_emb(x_rot - rot_center) + rot_center
        x_trans = self.x_trans_emb(x_trans - trans_center) + trans_center
        v_rot = self.v_rot_emb(v_rot)
        v_trans = self.v_trans_emb(v_trans)

        for i in range(self.n_layer):
            x_rot, v_rot = self.rot_blocks[i](x_rot, v_rot, attr1, attr2, rot_center)
            x_trans, v_trans = self.trans_blocks[i](x_trans, v_trans, attr1, attr2, trans_center)
            rot_center = self.rot_center_mlp[i](x_rot)
            trans_center = self.trans_center_mlp[i](x_trans)
            rot_center = torch.mean(rot_center, dim=(1, -1), keepdim=True)
            trans_center = torch.mean(trans_center, dim=(1, -1), keepdim=True)

        if self.decode_mode in {"residual", "cv_residual"}:
            out_rot = self.pro_rot(x_rot - rot_center)
            out_trans = self.pro_trans(x_trans - trans_center)
        else:
            out_rot = self.pro_rot(x_rot - rot_center) + rot_center
            out_trans = self.pro_trans(x_trans - trans_center) + trans_center
        return torch.cat([out_rot, out_trans], dim=2)

    def _constant_velocity_base(self, rel_dq):
        """Extrapolate relative pose DQs from the last observed DQ velocity."""
        last = rel_dq[:, :, -1:, :]
        if rel_dq.shape[2] < 2:
            return last.expand(-1, -1, self.output_n, -1)

        step = dq_project(dq_relative(rel_dq[:, :, -2:-1, :], last))
        step_twist = dq_log(step)
        current = last
        future = []
        for horizon in range(self.output_n):
            scale = self.cv_scale * (self.cv_decay ** horizon)
            step_h = dq_exp(step_twist * scale)
            current = dq_project(dq_mul(current, step_h))
            future.append(current)
        return torch.cat(future, dim=2)
