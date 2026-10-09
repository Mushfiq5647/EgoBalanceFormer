import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


class LossFuncLimb(nn.Module):
    list_joints = ["Neck", "Right_shoulder", "Right_elbow", "Right_wrist", "Left_shoulder", "Left_elbow",
                   "Left_wrist", "Right_hip", "Right_knee", "Right_ankle", "Right_foot", "Left_hip",
                   "Left_knee", "Left_ankle", "Left_foot"]
    lines = [(0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6), (1, 7), (4, 11), (7, 8), (8, 9), (9, 10),
             (11, 12), (12, 13), (13, 14), (7, 11)]
    kinematic_parents = [0, 0, 1, 2, 0, 4, 5, 1, 7, 8, 9, 4, 11, 12, 13]
    
    def __init__(self):
        super(LossFuncLimb, self).__init__()

        self.cos_loss = nn.CosineSimilarity(dim=2)

    def forward(self, pose_predicted, pose_gt):
        predicted_bone_vector = pose_predicted - pose_predicted[:, self.kinematic_parents, :]
        predicted_bone_vector = predicted_bone_vector[:, 1:, :]
        gt_bone_vector = pose_gt - pose_gt[:, self.kinematic_parents, :]
        gt_bone_vector = gt_bone_vector[:, 1:, :]

        cos_loss = self.cos_loss(predicted_bone_vector, gt_bone_vector)
        cos_loss = torch.mean(torch.sum(cos_loss, dim=1), dim=0)

        predicted_bone_length = torch.norm(predicted_bone_vector, dim=-1)
        gt_bone_length = torch.norm(gt_bone_vector, dim=-1)

        bone_length_loss = torch.mean(torch.sum((predicted_bone_length - gt_bone_length)**2, dim=1), dim=0)

        return cos_loss, bone_length_loss

class LossFuncCosSim(nn.Module):
    list_joints = ["Neck", "Right_shoulder", "Right_elbow", "Right_wrist", "Left_shoulder", "Left_elbow",
                   "Left_wrist", "Right_hip", "Right_knee", "Right_ankle", "Right_foot", "Left_hip",
                   "Left_knee", "Left_ankle", "Left_foot"]
    lines = [(0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6), (1, 7), (4, 11), (7, 8), (8, 9), (9, 10),
             (11, 12), (12, 13), (13, 14), (7, 11)]
    kinematic_parents = [0, 0, 1, 2, 0, 4, 5, 1, 7, 8, 9, 4, 11, 12, 13]
    
    def __init__(self):
        super(LossFuncCosSim, self).__init__()

        self.cos_loss = nn.CosineSimilarity(dim=2)

    def forward(self, pose_predicted, pose_gt):
        predicted_bone_vector = pose_predicted - pose_predicted[:, self.kinematic_parents, :]
        predicted_bone_vector = predicted_bone_vector[:, 1:, :]
        gt_bone_vector = pose_gt - pose_gt[:, self.kinematic_parents, :]
        gt_bone_vector = gt_bone_vector[:, 1:, :]

        cos_loss = self.cos_loss(predicted_bone_vector, gt_bone_vector)
        cos_loss = torch.mean(torch.sum(cos_loss, dim=1), dim=0)

        return cos_loss

class LossFuncMPJPE(nn.Module): 
    def __init__(self):
        super(LossFuncMPJPE, self).__init__()

    def forward(self, pred_pose, gt_pose):
        distance = torch.linalg.norm(gt_pose - pred_pose, dim=-1)
        return torch.mean(distance)


class CorrelationAwareLoss(nn.Module):
    """
    Correlation-aware loss for CoP prediction.
    Combines MSE/Huber with a correlation penalty to prevent mean collapse.
    
    Loss = base_loss(pred, target) + alpha * (1 - correlation(pred, target))
    
    Args:
        base_loss: 'mse' or 'huber'
        alpha: weight for correlation term (default: 0.5)
        huber_delta: delta for Huber loss (default: 1.0)
    """
    def __init__(self, base_loss: str = "huber", alpha: float = 0.5, huber_delta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.base_loss = base_loss
        if base_loss == "huber":
            self.criterion = nn.HuberLoss(delta=huber_delta)
        elif base_loss == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown base_loss: {base_loss}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 2) predicted CoP [X, Y]
            target: (B, 2) ground-truth CoP [X, Y]
        Returns:
            loss: scalar
        """
        # Base regression loss
        base = self.criterion(pred, target)
        
        if self.alpha <= 0:
            return base
        
        # Correlation penalty (per-axis, then average)
        corr_loss = 0.0
        for dim in range(pred.shape[1]):  # X and Y
            pred_dim = pred[:, dim]
            target_dim = target[:, dim]
            
            # Center
            pred_centered = pred_dim - pred_dim.mean()
            target_centered = target_dim - target_dim.mean()
            
            # Normalize
            pred_std = pred_centered.std() + 1e-8
            target_std = target_centered.std() + 1e-8
            pred_norm = pred_centered / pred_std
            target_norm = target_centered / target_std
            
            # Pearson correlation
            corr = (pred_norm * target_norm).mean()
            corr_loss += (1.0 - corr)  # minimize 1-corr → maximize corr
        
        corr_loss = corr_loss / pred.shape[1]  # average over X, Y
        
        return base + self.alpha * corr_loss


if __name__ == "__main__":

    loss = nn.MSELoss(reduction="none")
    input = torch.randn(4, 3, 5, 5, requires_grad=True)
    target = torch.randn(4, 3, 5, 5)
    output = loss(input, target)
    print(output)