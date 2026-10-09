#!/bin/bash
# Run EgoGlass pipeline on your Vicon dataset (19 joints -> 16) with body part fusion.
#
# Pipeline:
#   1. Self-supervise body part branch (train conv_body_part only)
#   2. Generate heatmaps + body part masks
#   3. Train AutoEncoder (38ch: heatmaps + body part)
#   4. Test
#
# Prerequisites:
#   - pose_estimation/utils/train.txt, pose_estimation/utils/test.txt
#   - Each sequence: ground_truth.json, color_frame_*.png
#   - EgoGlass pretrained heatmap weights in log/egoglass_B16/
#
# Run from project root: bash pose_estimation/scripts/autoencoder/egoglass_on_egocentric_dataset.sh

set -e

BODY_PART_LOG="log/egoglass_body_part_self_supervised"
AE_LOG="log/egocentric_egoglass_autoencoder"
EPOCHS=20

# Step 1: Self-supervised training of body part branch
echo "=== Step 1: Train body part branch (self-supervised) ==="
python pose_estimation/scripts/autoencoder/train_body_part_self_supervised.py \
    --train_list pose_estimation/utils/train.txt \
    --weights_left log/egoglass_B16/best_net_HeatMap_left.pth \
    --weights_right log/egoglass_B16/best_net_HeatMap_right.pth \
    --log_dir "$BODY_PART_LOG" \
    --epochs 5 \
    --batch_size 16

# Step 2a: Generate EgoGlass heatmaps + body part for train
echo "=== Step 2a: Generate heatmaps + body part (train) ==="
python pose_estimation/scripts/heatmaps/generate_egoglass_heatmaps_from_folders.py \
    --folders_txt pose_estimation/utils/train.txt \
    --weights_left "$BODY_PART_LOG/epoch_5_net_HeatMap_left.pth" \
    --weights_right "$BODY_PART_LOG/epoch_5_net_HeatMap_right.pth" \
    --output_subdir egoglass_pred_heatmaps \
    --save_body_part \
    --batch_size 16

# Step 2b: Generate for test
echo "=== Step 2b: Generate heatmaps + body part (test) ==="
python pose_estimation/scripts/heatmaps/generate_egoglass_heatmaps_from_folders.py \
    --folders_txt pose_estimation/utils/test.txt \
    --weights_left "$BODY_PART_LOG/epoch_5_net_HeatMap_left.pth" \
    --weights_right "$BODY_PART_LOG/epoch_5_net_HeatMap_right.pth" \
    --output_subdir egoglass_pred_heatmaps \
    --save_body_part \
    --batch_size 16

# Step 3: Train AutoEncoder (38ch with body part fusion)
echo "=== Step 3: Train AutoEncoder (heatmaps + body part) ==="
python pose_estimation/scripts/autoencoder/train_egocentric_autoencoder_from_heatmaps.py \
    --train_list pose_estimation/utils/train.txt \
    --heatmap_subdir egoglass_pred_heatmaps \
    --use_body_part \
    --log_dir "$AE_LOG" \
    --epochs $EPOCHS \
    --batch_size 64 \
    --lambda_mpjpe 1.0 \
    --lambda_cos_sim -0.01 \
    --lambda_heatmap_rec 0.001

# Step 4: Test
echo "=== Step 4: Test ==="
python pose_estimation/scripts/autoencoder/test_egocentric_autoencoder_from_heatmaps.py \
    --test_list pose_estimation/utils/test.txt \
    --checkpoint "$AE_LOG/epoch_${EPOCHS}_net_AutoEncoder.pth" \
    --heatmap_subdir egoglass_pred_heatmaps \
    --use_body_part
