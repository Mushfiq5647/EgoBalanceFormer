#!/bin/bash
python pose_estimation/test.py \
  --experiment_name unrealego_autoencoder_shared_B16 \
  --model unrealego_autoencoder \
  --data_dir ./pose_estimation/scripts/data/UnrealEgoData \
  --use_amp \
  --batch_size 1 \
  --path_to_trained_heatmap ./log/unrealego_heatmap_shared_B16/best_net_HeatMap.pth

