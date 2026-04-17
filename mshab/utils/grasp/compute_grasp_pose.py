
import torch
from typing import Optional, Dict, Union
def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)

def _gather_cols(R: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # R: [B,3,3], idx: [B] -> [B,3]
    return R.gather(dim=2, index=idx.view(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)

@torch.no_grad()
def compute_grasp_pose_by_obb_torch(
    pose: torch.Tensor,                 # [B, 4, 4]
    size: torch.Tensor,                 # [B, 3] (extents along local x,y,z)
    approaching: Union[torch.Tensor, tuple, list] = (0, 0, -1),
    target_closing: Optional[Union[torch.Tensor, tuple, list]] = None,
    depth: Union[float, torch.Tensor] = 0.0,
    ortho: bool = True,
) -> torch.Tensor:
    """
    Return grasp pose [B,4,4] directly.

    Grasp frame (columns):
      x = approaching
      y = closing
      z = approaching x closing

    Extra constraint:
      angle(z, world_x) <= 90 deg
      <=> dot(z, [1,0,0]) >= 0
    """
    assert pose.ndim == 3 and pose.shape[-2:] == (4, 4)
    assert size.ndim == 2 and size.shape[-1] == 3
    B = pose.shape[0]
    device, dtype = pose.device, pose.dtype

    R = pose[:, :3, :3]       # [B,3,3]
    t0 = pose[:, :3, 3]       # [B,3]
    extents = size            # [B,3]

    # approaching -> [B,3], normalize
    if not isinstance(approaching, torch.Tensor):
        approaching = torch.tensor(approaching, device=device, dtype=dtype)
    else:
        approaching = approaching.to(device=device, dtype=dtype)
    approaching_b = approaching.view(1, 3).expand(B, 3) if approaching.ndim == 1 else approaching
    approaching_b = _normalize(approaching_b)

    # angles[b,j] = approaching · axis_j  (axis_j is column j of R)
    angles = (R * approaching_b.view(B, 3, 1)).sum(dim=1)    # [B,3]
    inds0 = angles.abs().argsort(dim=-1)                     # [B,3] ascending
    ind0 = inds0[:, 2]                                       # [B] most aligned

    # remaining two axes
    rem = inds0[:, :2]                                       # [B,2]
    rem_ext = extents.gather(1, rem)                         # [B,2]
    rem_order = rem_ext.argsort(dim=-1)                      # [B,2]
    ind1 = rem.gather(1, rem_order[:, 0:1]).squeeze(1)       # [B] shorter -> closing
    ind2 = rem.gather(1, rem_order[:, 1:2]).squeeze(1)       # [B]

    # target_closing tie-break (when sizes are close)
    target_b = None
    if target_closing is not None:
        if not isinstance(target_closing, torch.Tensor):
            target_closing = torch.tensor(target_closing, device=device, dtype=dtype)
        else:
            target_closing = target_closing.to(device=device, dtype=dtype)
        target_b = target_closing.view(1, 3).expand(B, 3) if target_closing.ndim == 1 else target_closing
        target_b = _normalize(target_b)

        e1 = extents.gather(1, ind1.view(B, 1)).squeeze(1)
        e2 = extents.gather(1, ind2.view(B, 1)).squeeze(1)
        ratio = e1 / e2.clamp_min(1e-12)
        close_size = (ratio > 0.99) & (ratio < 1.01)

        v1 = _gather_cols(R, ind1)
        v2 = _gather_cols(R, ind2)
        dot1 = (target_b * v1).sum(dim=-1).abs()
        dot2 = (target_b * v2).sum(dim=-1).abs()
        swap = close_size & (dot1 < dot2)

        ind1_old, ind2_old = ind1, ind2
        ind1 = torch.where(swap, ind2_old, ind1_old)
        ind2 = torch.where(swap, ind1_old, ind2_old)

    # closing axis (world)
    closing = _gather_cols(R, ind1)
    closing = _normalize(closing)

    # flip closing if against target_closing
    if target_closing is not None and target_b is not None:
        flip = (target_b * closing).sum(dim=-1) < 0
        closing = torch.where(flip.view(B, 1), -closing, closing)

    # reorder extents to [ind0, ind1, ind2] for surface center computation
    order = torch.stack([ind0, ind1, ind2], dim=1)           # [B,3]
    extents_out = extents.gather(1, order)                   # [B,3]

    # surface center along approaching with depth clamp
    half_size = 0.5 * extents_out[:, 0]                      # [B]
    depth_t = torch.tensor(depth, device=device, dtype=dtype) if not torch.is_tensor(depth) else depth.to(device=device, dtype=dtype)
    depth_b = depth_t.expand(B) if depth_t.ndim == 0 else depth_t.view(B)
    move = (-half_size + torch.minimum(depth_b, half_size))   # [B]
    center = t0 + approaching_b * move.view(B, 1)             # [B,3]

    # orthogonalize closing w.r.t approaching
    if ortho:
        proj = (approaching_b * closing).sum(dim=-1, keepdim=True)
        closing = closing - proj * approaching_b
        closing = _normalize(closing)

    # ============================================================
    # Build grasp z-axis and enforce:
    #   angle(z, world_x) <= 90 deg  <=>  z_x >= 0
    # ============================================================
    grasp_z = torch.cross(approaching_b, closing, dim=-1)    # [B,3]
    grasp_z = _normalize(grasp_z)

    # 如果 z 轴与世界 X 轴夹角 > 90°，则翻转 closing
    # 这样 z 轴也会同步翻转，且仍保持右手系
    flip_for_world_x = grasp_z[:, 0] < 0
    closing = torch.where(flip_for_world_x.view(B, 1), -closing, closing)
    grasp_z = torch.where(flip_for_world_x.view(B, 1), -grasp_z, grasp_z)

    T = torch.eye(4, device=device, dtype=dtype).view(1, 4, 4).expand(B, 4, 4).clone()
    T[:, :3, :3] = torch.stack([approaching_b, closing, grasp_z], dim=-1)  # [B,3,3]
    T[:, :3, 3] = center
    return T