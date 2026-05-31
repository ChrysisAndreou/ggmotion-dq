"""Shared GGMotion block for 6D dual-quaternion log-coordinate streams."""

import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd

COORD_DIM = 6  # se(3) twist: [omega(3), v(3)]


class Block_DQ(nn.Module):
    """GGMotion block operating in 6D twist (se(3)) coordinate space."""

    def __init__(self, key_n, h_dim, e_dim, group, edges, act, eps, norm=False, fk=False):
        super().__init__()
        self.group = group
        self.edges = edges
        self.norm = norm
        self.key_n = key_n
        self.h_dim = h_dim
        self.eps = eps
        self.fk = fk

        self.spatio_mlp = nn.Linear(e_dim, e_dim, bias=False)
        self.temporal_mlp = nn.Linear(e_dim, e_dim, bias=False)

        # Scale parameters: scalar per joint (broadcast across 6D twist dims).
        # Using (N, 1) instead of (N, 6) preserves SE(3)-equivariance:
        # per-component scales break rotation equivariance because rotation
        # mixes the translation components (3,4,5) via R, but independent
        # scales treat them as separate channels.
        self.spatio_scale = nn.Parameter(torch.zeros(self.key_n, 1))
        self.temporal_scale = nn.Parameter(torch.zeros(self.key_n, 1))

        self.spatio_att_mlp = nn.Sequential(
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

        self.spatio_h_mlp = nn.Sequential(
            nn.Linear(e_dim, h_dim),
            act,
            nn.Linear(h_dim, e_dim)
        )

        self.temporal_h_mlp = nn.Sequential(
            nn.Linear(e_dim, h_dim),
            act,
            nn.Linear(h_dim, e_dim)
        )

        # Group force QKV
        self.group_force_emb = nn.ModuleList([
            nn.Linear(e_dim, e_dim, bias=False)
            for _ in range(3)])

        self.group_force_mlp = nn.Sequential(
            nn.Linear(len(self.group)**2, h_dim),
            act,
            nn.Linear(h_dim, len(self.group)**2),
        )
        self.group_force_out = nn.Linear(e_dim, e_dim, bias=False)

        # Part force QKV
        self.part_force_emb = nn.ModuleList([
            nn.Linear(e_dim, e_dim, bias=False)
            for _ in range(3)])

        self.part_force_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(part) ** 2, h_dim),
                act,
                nn.Linear(h_dim, len(part) ** 2),
            ) for part in self.group.values()
        ])
        self.part_force_out = nn.Linear(e_dim, e_dim, bias=False)

        self.a_out_mlp = nn.Linear(e_dim, e_dim, bias=False)

        if self.fk == 1:
            self.acc_emb = nn.ModuleList([
                nn.Linear(e_dim, e_dim, bias=False)
                for _ in range(3)])

            self.n_basis = 3
            self.acc_mlp = nn.Sequential(
                nn.Linear(self.n_basis ** 2, h_dim),
                act,
                nn.Linear(h_dim, self.n_basis),
            )
            self.acc_out = nn.Linear(e_dim, e_dim, bias=False)

    def forward(self, x, v, attr1, attr2, x_center):
        # x: (B, N, 6, E)   v: (B, N, 6, E)   x_center: (B, 1, 6, 1)
        batch_size = x.shape[0]

        # Spatial force
        spatio_x = x[:, self.edges[:, 0]] - x[:, self.edges[:, 1]]  # (B, G, 6, E)
        spatio_att = self.spatio_att_mlp(attr1).unsqueeze(-2)        # (B, G, 1, 1)
        scale = self.spatio_h_mlp(torch.norm(spatio_x, dim=-2, p=2)).unsqueeze(-2)  # (B, G, 1, E)
        spatio_x = spatio_att * scale * self.spatio_mlp(spatio_x)   # (B, G, 6, E)

        spatio_force = torch.zeros(x.shape, device=x.device)        # (B, N, 6, E)
        index = self.edges[:, 0][None, :, None, None].to(x.device)
        index = index.expand(spatio_x.shape[0], -1, spatio_x.shape[2], spatio_x.shape[3])
        spatio_force.scatter_add_(1, index, spatio_x)
        spatio_force = v + self.spatio_scale.unsqueeze(-1) * spatio_force

        # Temporal force
        temporal_x = x - x_center                                    # (B, N, 6, E)
        scale = self.temporal_h_mlp(torch.norm(temporal_x, dim=-2, p=2)).unsqueeze(-2)
        temporal_force = scale * self.temporal_mlp(temporal_x)
        temporal_force = v + self.temporal_scale.unsqueeze(-1) * temporal_force

        f = spatio_force + temporal_force                            # (B, N, 6, E)

        # Group force (QKV attention over body part groups)
        group_force = torch.zeros(f.shape[0], len(self.group), f.shape[2], f.shape[3],
                                  device=f.device)                   # (B, P, 6, E)
        for index, part in enumerate(self.group.values()):
            group_force[:, index] = torch.sum(f[:, part[:, 1]], dim=1)

        force_k = self.group_force_emb[0](group_force).permute(0, 3, 2, 1)  # (B, E, 6, P)
        force_q = self.group_force_emb[1](group_force).permute(0, 3, 2, 1)
        force_v = self.group_force_emb[2](group_force).permute(0, 3, 2, 1)
        invar_force = torch.matmul(force_q.transpose(2, 3), force_k)        # (B, E, P, P)
        if self.norm:
            invar_force = F.normalize(invar_force, dim=-1, p=2, eps=self.eps)
        invar_force = self.group_force_mlp(invar_force.flatten(-2)).view(
            batch_size, -1, invar_force.shape[-2], invar_force.shape[-1])
        group_force = self.group_force_out(
            torch.matmul(force_v, invar_force).permute(0, 3, 2, 1))         # (B, P, 6, E)

        act_force = torch.zeros(f.shape, device=f.device)                    # (B, N, 6, E)
        for index, part in enumerate(self.group.values()):
            act_force[:, part[:, 1]] = group_force[:, index].unsqueeze(1) \
                .repeat(1, len(part[:, 1]), 1, 1)

        f = f + act_force

        # Part force + FK + update
        x_out = torch.zeros_like(x, device=x.device)                        # (B, N, 6, E)
        v_out = torch.zeros_like(v, device=v.device)
        for index, part in enumerate(self.group.values()):
            act_force = f[:, part[:, 1]]                                     # (B, G, 6, E)
            force_k = self.part_force_emb[0](act_force).permute(0, 3, 2, 1) # (B, E, 6, G)
            force_q = self.part_force_emb[1](act_force).permute(0, 3, 2, 1)
            force_v = self.part_force_emb[2](act_force).permute(0, 3, 2, 1)
            invar_force = torch.matmul(force_q.transpose(2, 3), force_k)     # (B, E, G, G)
            if self.norm:
                invar_force = F.normalize(invar_force, dim=-1, p=2, eps=self.eps)
            invar_force = self.part_force_mlp[index](invar_force.flatten(-2)).view(
                batch_size, -1, invar_force.shape[-2], invar_force.shape[-1])
            act_force = self.part_force_out(
                torch.matmul(force_v, invar_force).permute(0, 3, 2, 1))      # (B, G, 6, E)

            r_diff = x[:, part[:, 1]] - x[:, part[:, 0]]                     # (B, G, 6, E)
            v_diff = v[:, part[:, 1]] - v[:, part[:, 0]]
            f_diff = f[:, part[:, 1]] + act_force

            if self.fk == 1:
                a = torch.stack((f_diff, r_diff, v_diff), dim=-1)             # (B, G, 6, E, 3)
                a_k = self.acc_emb[0](a.transpose(3, 4)).permute(0, 1, 4, 2, 3)
                a_q = self.acc_emb[1](a.transpose(3, 4)).permute(0, 1, 4, 2, 3)
                a_v = self.acc_emb[2](a.transpose(3, 4)).permute(0, 1, 4, 2, 3)
                invar_a = torch.matmul(a_q.transpose(3, 4), a_k)             # (B, G, E, 3, 3)
                if self.norm:
                    invar_a = F.normalize(invar_a, dim=-1, p=2, eps=1e-6)
                invar_a = self.acc_mlp(invar_a.flatten(-2)).unsqueeze(-1)     # (B, G, E, 3, 1)
                a = self.acc_out(
                    torch.matmul(a_v, invar_a).squeeze(-1).permute(0, 1, 3, 2))
                a_out = f_diff - a

            elif self.fk == 2:
                part_ind = pd.factorize(part[1:, :2].flatten(), sort=False)[0].reshape(-1, 2)
                a_out = torch.zeros_like(f_diff, device=x.device)
                a_out[:, 0] = torch.sum(f_diff, dim=1)

                for i, j in part_ind:
                    cur_a = a_out[:, i]
                    cur_r = r_diff[:, j]
                    cur_v = v_diff[:, j]
                    cur_f = f_diff[:, j]
                    # Cross product on first 3 dims (rotation part of twist)
                    # For 6D twists, fk=2 (manual FK) uses cross products which
                    # only work in 3D. This mode is not recommended for DQ.
                    # Fallback: treat as fk=0
                    a_out[:, j] = cur_f

            else:
                a_out = f_diff

            v_out[:, part[:, 1]] = v[:, part[:, 1]] + self.a_out_mlp(a_out)
            x_out[:, part[:, 1]] = x[:, part[:, 1]] + v_out[:, part[:, 1]]

        return x_out, v_out
