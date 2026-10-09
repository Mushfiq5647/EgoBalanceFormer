# Egocentric Pose -> Balance Prediction Pipeline

## 1) Data sources per sequence
- `ground_truth.json`:
  - timestamps
  - VR (`vrHMD`, `vrLeft`, `vrRight`)
  - optional GT rotations used by `estimated15` mode
- `balance.json`:
  - CoP target (`copX`, `copY`) with timestamps
- heatmaps directory (e.g. `custom_pred_heatmaps`):
  - `*_hm15x64x64.npy` for AE pose estimation

## 2) Pose estimation mode used for monocular pipeline
- Dataset: `estimated15`
- AE checkpoint predicts 15-joint pose from heatmaps.
- Lower-body joints are selected for prediction inputs.

## 3) Timestamp alignment
For each CoP timestamp:
1. Find nearest pose timestamp.
2. Build a causal 11-frame window ending at that pose frame.
3. Drop sample if nearest timestamp gap > `max_offset_ms`.

This is nearest-timestamp alignment, not fixed `i*11` pairing.

## 4) Feature construction (current simple/stable variants)
Common features available:
- `pose_pos` (lower-body)
- `pose_vel` (optional)
- `com`, `com_vel`
- `joint_groups` (optional)
- `joint_rotations` (optional, from GT rotations)
- `vr_pos`, `vr_vel` (optional)

Recommended simple setup for debugging bottlenecks:
- disable VR
- disable joint groups if needed
- disable joint rotations if needed
- optionally disable velocities

## 5) Normalization
- Inputs: feature-wise z-score (per-channel stats from train set only).
- CoP target:
  - `zscore` (default), or
  - `minmax` to `[-1,1]`.
- Test/validation use the same saved `normalizer.npz`.
- Metrics are computed after inverse-transform back to original units.

## 6) Model/loss
- Model typically used: `st_transformer_enhanced`.
- Loss options in training script:
  - `huber` (recommended default)
  - `mse`
  - `correlation`

## 7) Typical train/test commands
### Train
```bash
python -u balance_prediction/train.py \
  --train_list pose_estimation/utils/train.txt \
  --val_list pose_estimation/utils/test.txt \
  --dataset_type estimated15 \
  --model_type st_transformer_enhanced \
  --ae_checkpoint log/egocentric_ae_15j_monocular/epoch_50_net_AutoEncoder.pth \
  --heatmap_subdir custom_pred_heatmaps \
  --no_vr \
  --loss_type huber \
  --huber_delta 1.0 \
  --cop_norm zscore \
  --mad_threshold 4 \
  --batch_size 64 \
  --epochs 25 \
  --lr 1e-4 \
  --dropout 0.25 \
  --weight_decay 1e-4 \
  --log_dir log/cop_estimated15_run
```

### Test (epoch checkpoint)
```bash
python -u balance_prediction/test.py \
  --test_list pose_estimation/utils/test.txt \
  --dataset_type estimated15 \
  --model_type st_transformer_enhanced \
  --ae_checkpoint log/egocentric_ae_15j_monocular/epoch_50_net_AutoEncoder.pth \
  --no_vr \
  --mad_threshold 4 \
  --checkpoint log/cop_estimated15_run/epoch_0025.pth \
  --normalizer log/cop_estimated15_run/normalizer.npz \
  --output_json log/cop_estimated15_run/test_results_epoch25.json
```

## 8) Common failure checks
- Train flags and test flags must match branch toggles (`no_vr`, `no_joint_groups`, `no_joint_rotations`, `output_activation`).
- If `state_dict` mismatch occurs, architecture flags differ between train/test.
- If val rises while train falls, use `best_model.pth` for reporting and run ablations to find bottleneck branch.
