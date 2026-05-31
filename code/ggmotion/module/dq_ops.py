"""
Dual quaternion operations for GGMotion-DQPose.

DQ convention: [w, x, y, z, dw, dx, dy, dz] where real part = rotation quaternion,
dual part encodes translation via d = (1/2) * t_quat * r.
Twist convention: [omega(3), v(3)] - so(3) rotation followed by R^3 translation.
"""

import torch


# Quaternion primitives

def quat_mul(u, v):
    """Multiply two quaternions. [..., 4] x [..., 4] -> [..., 4]."""
    w1, x1, y1, z1 = u.unbind(-1)
    w2, x2, y2, z2 = v.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)


def quat_conj(q):
    """Quaternion conjugate. [..., 4] -> [..., 4]."""
    mask = torch.tensor([1, -1, -1, -1], device=q.device, dtype=q.dtype)
    return q * mask


def quat_normalize(q, eps=1e-8):
    """Normalize quaternion to unit length."""
    return q / (torch.norm(q, dim=-1, keepdim=True) + eps)


# Dual quaternion operations

def dq_mul(p, q):
    """Multiply two dual quaternions. [..., 8] x [..., 8] -> [..., 8]."""
    p_r, p_d = p[..., :4], p[..., 4:]
    q_r, q_d = q[..., :4], q[..., 4:]
    r_part = quat_mul(p_r, q_r)
    d_part = quat_mul(p_r, q_d) + quat_mul(p_d, q_r)
    return torch.cat([r_part, d_part], dim=-1)


def dq_conj(p):
    """DQ conjugate (= inverse for unit DQs). [..., 8] -> [..., 8]."""
    mask = torch.tensor([1, -1, -1, -1, 1, -1, -1, -1], device=p.device, dtype=p.dtype)
    return p * mask


def dq_canonicalize(p):
    """Canonicalize DQ sign so real part w >= 0."""
    sign = torch.where(p[..., :1] < 0, -1.0, 1.0)
    return p * sign


def dq_project(p, eps=1e-8):
    """Project onto unit DQ manifold: ||r||=1, r.d=0, w>=0."""
    r, d = p[..., :4], p[..., 4:]
    r = quat_normalize(r, eps=eps)
    dot = (r * d).sum(dim=-1, keepdim=True)
    d = d - dot * r
    return dq_canonicalize(torch.cat([r, d], dim=-1))


def dq_relative(a, b):
    """Relative transform from a to b: a^{-1} * b. [..., 8] -> [..., 8]."""
    return dq_mul(dq_conj(a), b)


def dq_to_translation(dq):
    """Extract translation from unit DQ: t = 2 * d * r*. [..., 8] -> [..., 3]."""
    r, d = dq[..., :4], dq[..., 4:]
    t_quat = 2.0 * quat_mul(d, quat_conj(r))
    return t_quat[..., 1:]


def dq_to_pos(dq):
    """Alias for dq_to_translation."""
    return dq_to_translation(dq)


def pos_to_dq(pos):
    """
    Convert 3D positions to pure translation DQs.
    Input:  pos [..., 3]
    Output: dq  [..., 8]  = [1, 0, 0, 0, 0, x/2, y/2, z/2]
    """
    batch_dims = pos.shape[:-1]
    ones = torch.ones((*batch_dims, 1), device=pos.device, dtype=pos.dtype)
    zeros3 = torch.zeros((*batch_dims, 3), device=pos.device, dtype=pos.dtype)
    zero1 = torch.zeros((*batch_dims, 1), device=pos.device, dtype=pos.dtype)
    real = torch.cat([ones, zeros3], dim=-1)       # [1, 0, 0, 0]
    dual = torch.cat([zero1, pos * 0.5], dim=-1)   # [0, x/2, y/2, z/2]
    return torch.cat([real, dual], dim=-1)


def quat_pos_to_dq(quat, pos, eps=1e-8):
    """
    Convert rotation quaternion + translation to a unit dual quaternion.

    Inputs:
      quat [..., 4] = [w, x, y, z]
      pos  [..., 3] = translation
    Output:
      dq   [..., 8] = [q_r, q_d], with q_d = 0.5 * t_quat * q_r
    """
    quat = quat_normalize(quat, eps=eps)
    t_quat = torch.cat([torch.zeros_like(pos[..., :1]), pos], dim=-1)
    dual = 0.5 * quat_mul(t_quat, quat)
    return dq_project(torch.cat([quat, dual], dim=-1), eps=eps)


def rotmat_to_quat(rot, eps=1e-8):
    """
    Numerically stable rotation-matrix to quaternion conversion.

    Branches on the largest quaternion component, following the same idea as
    PyTorch3D's matrix_to_quaternion, and returns [w, x, y, z].
    """
    m00 = rot[..., 0, 0]
    m01 = rot[..., 0, 1]
    m02 = rot[..., 0, 2]
    m10 = rot[..., 1, 0]
    m11 = rot[..., 1, 1]
    m12 = rot[..., 1, 2]
    m20 = rot[..., 2, 0]
    m21 = rot[..., 2, 1]
    m22 = rot[..., 2, 2]

    q_abs = torch.sqrt(torch.clamp(torch.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], dim=-1), min=eps))

    quat_by_wxyz = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m21 + m12], dim=-1),
        torch.stack([m10 - m01, m02 + m20, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
    ], dim=-2)

    denom = (2.0 * q_abs).clamp_min(0.1).unsqueeze(-1)
    quat_candidates = quat_by_wxyz / denom
    best = q_abs.argmax(dim=-1)
    gather_idx = best[..., None, None].expand(*best.shape, 1, 4)
    quat = quat_candidates.gather(-2, gather_idx).squeeze(-2)
    return quat_normalize(quat, eps=eps)


def pose_to_dq(rot, pos, eps=1e-8):
    """Convert rotation matrix + translation to a unit dual quaternion."""
    return quat_pos_to_dq(rotmat_to_quat(rot, eps=eps), pos, eps=eps)


def dq_fk(local_dq, parent, order=None, eps=1e-8):
    """
    Compose local parent-to-child DQs down a kinematic tree.

    local_dq: (B, N, T, 8), where root entries are global root poses and
      non-root entries are parent->child local transforms.
    parent: (N,) integer parent indices, with -1 for the root.
    order: optional topological order over nodes.
    """
    parent = torch.as_tensor(parent, device=local_dq.device, dtype=torch.long)
    n_nodes = local_dq.shape[1]
    if order is None:
        remaining = set(range(n_nodes))
        visited = set()
        order_list = []
        while remaining:
            progressed = False
            for node in list(remaining):
                p = int(parent[node].item())
                if p < 0 or p in visited:
                    order_list.append(node)
                    visited.add(node)
                    remaining.remove(node)
                    progressed = True
            if not progressed:
                raise ValueError("Parent array is cyclic or disconnected")
    else:
        order_list = [int(i) for i in torch.as_tensor(order).detach().cpu().tolist()]

    global_nodes = [None] * n_nodes
    for node in order_list:
        p = int(parent[node].item())
        if p < 0:
            global_nodes[node] = local_dq[:, node]
        else:
            global_nodes[node] = dq_mul(global_nodes[p], local_dq[:, node])

    return dq_project(torch.stack(global_nodes, dim=1), eps=eps)


# Lie algebra (so(3) / se(3))

def so3_exp(omega, eps=1e-8):
    """Exponential map so(3) -> unit quaternion. [..., 3] -> [..., 4]."""
    theta = torch.norm(omega, dim=-1, keepdim=True)
    half = 0.5 * theta
    theta2 = theta * theta
    k = torch.where(
        theta < eps,
        0.5 - theta2 / 48.0 + (theta2 * theta2) / 3840.0,
        torch.sin(half) / (theta + eps)
    )
    w = torch.cos(half)
    return torch.cat([w, omega * k], dim=-1)


def so3_log(q, eps=1e-8):
    """Log map unit quaternion -> so(3). [..., 4] -> [..., 3]."""
    q = quat_normalize(q, eps=eps)
    w = torch.clamp(q[..., :1], -1.0, 1.0)
    v = q[..., 1:]
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    theta = 2.0 * torch.atan2(v_norm, w)
    scale = torch.where(v_norm < eps, 2.0 * torch.ones_like(v_norm), theta / (v_norm + eps))
    return v * scale


def _skew(w):
    """Skew-symmetric matrix from 3-vector. [..., 3] -> [..., 3, 3]."""
    wx, wy, wz = w.unbind(-1)
    O = torch.zeros((*w.shape[:-1], 3, 3), device=w.device, dtype=w.dtype)
    O[..., 0, 1] = -wz
    O[..., 0, 2] = wy
    O[..., 1, 0] = wz
    O[..., 1, 2] = -wx
    O[..., 2, 0] = -wy
    O[..., 2, 1] = wx
    return O


def se3_exp(xi, eps=1e-8):
    """Exponential map se(3) -> unit DQ. xi = [omega(3), v(3)]. [..., 6] -> [..., 8]."""
    omega = xi[..., :3]
    v = xi[..., 3:]
    theta = torch.norm(omega, dim=-1, keepdim=True)
    theta2 = theta * theta
    B = torch.where(theta < eps, 0.5 - theta2 / 24.0 + (theta2 * theta2) / 720.0,
                    (1.0 - torch.cos(theta)) / (theta2 + eps))
    C = torch.where(theta < eps, 1.0 / 6.0 - theta2 / 120.0 + (theta2 * theta2) / 5040.0,
                    (theta - torch.sin(theta)) / (theta2 * theta + eps))

    Omega = _skew(omega)
    Omega2 = Omega @ Omega
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).view((1,) * (omega.dim() - 1) + (3, 3))
    J = I + B.unsqueeze(-1) * Omega + C.unsqueeze(-1) * Omega2
    t = (J @ v.unsqueeze(-1)).squeeze(-1)

    r = so3_exp(omega, eps=eps)
    t_quat = torch.cat([torch.zeros_like(t[..., :1]), t], dim=-1)
    d = 0.5 * quat_mul(t_quat, r)
    return torch.cat([r, d], dim=-1)


def se3_log(dq, eps=1e-8):
    """Log map unit DQ -> se(3) twist. [..., 8] -> [..., 6] = [omega(3), v(3)]."""
    dq = dq_project(dq, eps=eps)
    r = dq[..., :4]
    t = dq_to_translation(dq)
    omega = so3_log(r, eps=eps)

    theta = torch.norm(omega, dim=-1, keepdim=True)
    Omega = _skew(omega)
    Omega2 = Omega @ Omega
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).view((1,) * (omega.dim() - 1) + (3, 3))

    theta2 = theta * theta
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    coef = torch.where(
        theta < eps,
        1.0 / 12.0 + theta2 / 720.0 + (theta2 * theta2) / 30240.0,
        (1.0 / (theta2 + eps)) - (1.0 + cos_theta) / (2.0 * (theta + eps) * (sin_theta + eps))
    )
    J_inv = I - 0.5 * Omega + coef.unsqueeze(-1) * Omega2
    v = (J_inv @ t.unsqueeze(-1)).squeeze(-1)
    return torch.cat([omega, v], dim=-1)


# Convenience aliases
dq_exp = se3_exp
dq_log = se3_log
