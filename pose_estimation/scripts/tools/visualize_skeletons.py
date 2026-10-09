# visualize_skeletons.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import matplotlib.pyplot as plt
from options.test_options import TestOptions
from dataloader.data_loader import dataloader_full
from model.models import create_model

# Joint order and kinematic tree from utils/loss.py
JOINT_NAMES = [
    "head","neck_01","upperarm_l","upperarm_r","lowerarm_l","lowerarm_r",
    "hand_l","hand_r","thigh_l","thigh_r","calf_l","calf_r","foot_l","foot_r","ball_l","ball_r"
]
KINEMATIC_PARENTS = [0,0,1,1,2,3,4,5,2,3,8,9,10,11,12,13]

def plot_skeleton(ax, joints, color, title):
    # joints: (K,3) in cm, pelvis-relative
    x, y, z = joints[:,0], joints[:,1], joints[:,2]
    ax.scatter(x, y, z, c=color, s=12)
    # draw bones using parents (skip root self-edge 0->0)
    for j in range(1, len(JOINT_NAMES)):
        p = KINEMATIC_PARENTS[j]
        ax.plot([x[p], x[j]], [y[p], y[j]], [z[p], z[j]], c=color, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel('X (cm)'); ax.set_ylabel('Y (cm)'); ax.set_zlabel('Z (cm)')
    ax.view_init(elev=20, azim=-60)
    ax.set_box_aspect([1,1,1])

def main():
    opt = TestOptions().parse()
    opt.batch_size = 1
    opt.ntest = 5  # visualize a handful

    # Build and load model
    model = create_model(opt)           # constructs HeatMap + AutoEncoder
    model.load_networks("best")         # loads from log/<experiment_name>/

    # Data loader for test split
    test_loader = dataloader_full(opt, mode='test')

    os.makedirs("./pose_estimation/unrealego_vis", exist_ok=True)

    for i, data in enumerate(test_loader):
        model.set_input(data)
        with torch.no_grad():
            # Forward HeatMap and pose (same as evaluate)
            pred_heatmap_cat = model.net_HeatMap(
                data['input_rgb_left'].to(model.device),
                data['input_rgb_right'].to(model.device),
            )
            pred_pose = model.net_AutoEncoder.predict_pose(pred_heatmap_cat)  # (B,K,3)
        gt_pose = data['gt_local_pose']  # (B,K,3)

        pred = pred_pose[0].detach().cpu().numpy()
        gt = gt_pose[0].detach().cpu().numpy()

        fig = plt.figure(figsize=(8,4))
        ax1 = fig.add_subplot(1,2,1, projection='3d')
        plot_skeleton(ax1, gt, 'g', 'GT (pelvis-relative)')
        ax2 = fig.add_subplot(1,2,2, projection='3d')
        plot_skeleton(ax2, pred, 'r', 'Pred (pelvis-relative)')

        out_png = f"./pose_estimation/unrealego_vis/frame_{i}.png"
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Saved {out_png}")

        if i >= 4:
            break

if __name__ == "__main__":
    main()