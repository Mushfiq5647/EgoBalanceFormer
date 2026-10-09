"""
Pseudo-limb mask generation for EgoGlass (per paper Section 4).

Paper: "We connect the areas between joints of Shoulder, Elbow, and Wrist to generate
the mask for one arm and the areas between the joints of Hip, Knee and Ankle to
generate the mask for one leg."

UnrealEgo 15 heatmap channels map to: head, neck, upperarm_l, upperarm_r, lowerarm_l,
lowerarm_r, hand_l, hand_r, thigh_l, thigh_r, calf_l, calf_r, foot_l, foot_r, ball_l.

Limb connectivity (indices 0-14):
- Left arm:  upperarm_l(2) -> lowerarm_l(4) -> hand_l(6)
- Right arm: upperarm_r(3) -> lowerarm_r(5) -> hand_r(7)
- Left leg:  thigh_l(8) -> calf_l(10) -> foot_l(12)
- Right leg: thigh_r(9) -> calf_r(11) -> foot_r(13)
"""
import numpy as np
import torch

NUM_LIMBS = 4  # left_arm, right_arm, left_leg, right_leg

# (joint_a, joint_b) pairs for each limb - heatmap channel indices
LIMB_SEGMENTS = [
    [(2, 4), (4, 6)],   # left arm:  shoulder->elbow->wrist
    [(3, 5), (5, 7)],   # right arm
    [(8, 10), (10, 12)],  # left leg:  hip->knee->ankle
    [(9, 11), (11, 13)],  # right leg
]


def _soft_argmax_2d(heatmap: torch.Tensor) -> torch.Tensor:
    """Compute 2D (x,y) position from heatmap channel. Returns [B, 2] in normalized [0,1]."""
    B, H, W = heatmap.shape
    heatmap = heatmap.view(B, -1)
    heatmap = heatmap / (heatmap.sum(dim=1, keepdim=True) + 1e-8)
    coords = torch.arange(H * W, device=heatmap.device, dtype=heatmap.dtype)
    x = (coords % W).float()
    y = (coords // W).float()
    pos_x = (heatmap * x).sum(dim=1)
    pos_y = (heatmap * y).sum(dim=1)
    return torch.stack([pos_x, pos_y], dim=1)


def _draw_segment(mask: np.ndarray, p0: np.ndarray, p1: np.ndarray, thickness: float) -> None:
    """Draw thick line segment on mask (in-place). p0, p1 are (x,y) in pixel coords."""
    H, W = mask.shape
    n_samples = max(int(np.ceil(np.linalg.norm(p1 - p0)) * 2), 2)
    for t in np.linspace(0, 1, n_samples):
        pt = p0 * (1 - t) + p1 * t
        x, y = int(round(pt[0])), int(round(pt[1]))
        r = int(np.ceil(thickness))
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy <= thickness * thickness:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        mask[ny, nx] = 1.0


def heatmap_to_pseudo_limb_mask(
    heatmap: torch.Tensor,
    thickness: float = 3.0,
) -> torch.Tensor:
    """
    Generate pseudo-limb mask from joint heatmaps (EgoGlass paper style).

    Connects areas between joints: Shoulder-Elbow-Wrist for arms, Hip-Knee-Ankle for legs.

    Args:
        heatmap: [B, 15, H, W] or [B, J, H, W] (J>=14)
        thickness: line thickness in pixels for limb drawing

    Returns:
        mask: [B, 4, H, W] with values in [0,1] per limb channel
    """
    if heatmap.dim() == 3:
        heatmap = heatmap.unsqueeze(0)
    B, J, H, W = heatmap.shape
    device = heatmap.device

    # Get 2D joint positions from each heatmap channel (soft-argmax)
    joints_xy = []
    for j in range(min(J, 14)):
        hm = heatmap[:, j, :, :]
        xy = _soft_argmax_2d(hm)  # [B, 2] in [0,W-1], [0,H-1]
        joints_xy.append(xy)
    # Pad if fewer than 14 joints
    while len(joints_xy) < 14:
        joints_xy.append(joints_xy[-1].clone() if joints_xy else torch.zeros(B, 2, device=device))

    masks = []
    for limb_id, segments in enumerate(LIMB_SEGMENTS):
        limb_mask = np.zeros((B, H, W), dtype=np.float32)
        for b in range(B):
            for (ja, jb) in segments:
                p0 = joints_xy[ja][b].detach().cpu().numpy()
                p1 = joints_xy[jb][b].detach().cpu().numpy()
                _draw_segment(limb_mask[b], p0, p1, thickness)
        masks.append(limb_mask)
    out = np.stack(masks, axis=1)  # [B, 4, H, W]
    return torch.from_numpy(out).float().to(device)


def heatmap_to_pseudo_limb_mask_batch(
    heatmap: torch.Tensor,
    thickness: float = 3.0,
) -> torch.Tensor:
    """Same as heatmap_to_pseudo_limb_mask, alias for clarity."""
    return heatmap_to_pseudo_limb_mask(heatmap, thickness)


# Legacy: kept for any code that imports heatmap_to_part_mask (semantic parts)
def heatmap_to_part_mask(heatmap: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    """Legacy: joint-dominant part mask. Prefer heatmap_to_pseudo_limb_mask for EgoGlass."""
    JOINT_TO_PART = [1, 2, 3, 4, 3, 4, 3, 4, 5, 6, 5, 6, 5, 6, 5]
    if heatmap.dim() == 3:
        heatmap = heatmap.unsqueeze(0)
    B, J, H, W = heatmap.shape
    max_val, argmax = heatmap.max(dim=1)
    joint_idx = argmax.clamp(0, min(J, len(JOINT_TO_PART)) - 1)
    arr = np.array(JOINT_TO_PART, dtype=np.int64)
    part_mask = torch.from_numpy(arr).to(heatmap.device)[joint_idx.long()]
    part_mask = part_mask * (max_val >= threshold).long()
    return part_mask


NUM_PARTS = 7
