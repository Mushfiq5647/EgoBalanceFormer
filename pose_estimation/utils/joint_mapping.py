import numpy as np


# Your joint order (preferred_order):
# 0: Root (pelvis)
# 1: LowerBack
# 2: Head
# 3: L_Collar
# 4: L_Humerus
# 5: L_Elbow
# 6: L_Wrist
# 7: L_Femur
# 8: L_Tibia
# 9: L_Foot
# 10: L_Toe
# 11: R_Collar
# 12: R_Humerus
# 13: R_Elbow
# 14: R_Wrist
# 15: R_Femur
# 16: R_Tibia
# 17: R_Foot
# 18: R_Toe
#
# UnrealEgo joint order (utils/loss.py:list_joints):
# 0: head
# 1: neck_01
# 2: upperarm_l
# 3: upperarm_r
# 4: lowerarm_l
# 5: lowerarm_r
# 6: hand_l
# 7: hand_r
# 8: thigh_l
# 9: thigh_r
# 10: calf_l
# 11: calf_r
# 12: foot_l
# 13: foot_r
# 14: ball_l
# 15: ball_r


def map_19_to_unrealego_16(joints_19: np.ndarray) -> np.ndarray:
    """
    Map your 19-joint skeleton to UnrealEgo's 16 joints, pelvis-relative.

    Args:
        joints_19: np.ndarray of shape (19, 3) in preferred_order

    Returns:
        np.ndarray of shape (16, 3) in UnrealEgo order (utils/loss.py:list_joints)
    """
    joints_19 = np.asarray(joints_19)
    if joints_19.shape != (19, 3):
        raise ValueError(f"Expected joints_19 shape (19, 3), got {joints_19.shape}")

    # pelvis-relative (Root is pelvis)
    pelvis = joints_19[0]
    J = joints_19 - pelvis

    head = J[2]
    neck_01 = (J[3] + J[11]) / 2.0  # (L_Collar + R_Collar) / 2

    joints_16 = np.stack(
        [
            head,  # head
            neck_01,  # neck_01
            J[4],  # upperarm_l   <- L_Humerus
            J[12],  # upperarm_r  <- R_Humerus
            J[5],  # lowerarm_l   <- L_Elbow
            J[13],  # lowerarm_r  <- R_Elbow
            J[6],  # hand_l       <- L_Wrist
            J[14],  # hand_r      <- R_Wrist
            J[7],  # thigh_l      <- L_Femur
            J[15],  # thigh_r     <- R_Femur
            J[8],  # calf_l       <- L_Tibia
            J[16],  # calf_r      <- R_Tibia
            J[9],  # foot_l       <- L_Foot
            J[17],  # foot_r      <- R_Foot
            J[10],  # ball_l      <- L_Toe
            J[18],  # ball_r      <- R_Toe
        ],
        axis=0,
    )

    return joints_16


# Preferred 15-joint order (common pose format):
# ["Neck", "Right_shoulder", "Right_elbow", "Right_wrist", "Left_shoulder", "Left_elbow",
#  "Left_wrist", "Right_hip", "Right_knee", "Right_ankle", "Right_foot", "Left_hip",
#  "Left_knee", "Left_ankle", "Left_foot"]


def map_19_to_preferred_15(joints_19: np.ndarray) -> np.ndarray:
    """
    Map your 19-joint skeleton (dataset order) to preferred 15 joints, pelvis-relative.

    Preferred 15 order: Neck, Right_shoulder, Right_elbow, Right_wrist, Left_shoulder,
    Left_elbow, Left_wrist, Right_hip, Right_knee, Right_ankle, Right_foot, Left_hip,
    Left_knee, Left_ankle, Left_foot.

    Args:
        joints_19: np.ndarray of shape (19, 3) in dataset order (Root, LowerBack, Head, ...).

    Returns:
        np.ndarray of shape (15, 3) in preferred order, pelvis-relative.
    """
    joints_19 = np.asarray(joints_19)
    if joints_19.shape != (19, 3):
        raise ValueError(f"Expected joints_19 shape (19, 3), got {joints_19.shape}")

    pelvis = joints_19[0]
    J = joints_19 - pelvis

    neck = (J[3] + J[11]) / 2.0  # (L_Collar + R_Collar) / 2

    joints_15 = np.stack(
        [
            neck,  # Neck
            J[11],  # Right_shoulder <- R_Collar
            J[13],  # Right_elbow   <- R_Elbow
            J[14],  # Right_wrist   <- R_Wrist
            J[3],  # Left_shoulder  <- L_Collar
            J[5],  # Left_elbow    <- L_Elbow
            J[6],  # Left_wrist    <- L_Wrist
            J[15],  # Right_hip     <- R_Femur
            J[16],  # Right_knee    <- R_Tibia
            J[17],  # Right_ankle    <- R_Foot
            J[18],  # Right_foot     <- R_Toe
            J[7],  # Left_hip      <- L_Femur
            J[8],  # Left_knee      <- L_Tibia
            J[9],  # Left_ankle      <- L_Foot
            J[10],  # Left_foot      <- L_Toe
        ],
        axis=0,
    )

    return joints_15

