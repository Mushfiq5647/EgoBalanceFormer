# EgoBalanceFormer

Code for **EgoBalanceFormer: Predicting Human Balance from Egocentric Pose and VR-Native Motion**
(Md Mushfiqur Azam, Nipa Anjum, John Quarles, Kevin Desai — IEEE TVCG).

EgoBalanceFormer predicts the Center of Pressure (CoP<sub>X</sub>, CoP<sub>Y</sub>, in cm) of a person standing in VR. It uses two inputs:

- egocentric RGB from an HMD-mounted camera (Orbbec Gemini 336L)
- HMD tracking (VIVE Focus Vision)

The model is evaluated with subject-wise 10-fold cross-validation on the VRBalance dataset.

## Pipeline

```
egocentric RGB ─► 2D heatmaps ─► 3D pose (15 joints) ─► Procrustes ─► features ─► EgoBalanceFormer ─► CoP (x, y)
                  ResNet-18        heatmap autoencoder                 lower-body pos/vel,
                  (step 1)         (step 2, run on the fly)            CoM + CoM vel, HMD pos/vel
```

| Stage | Paper | Code |
|---|---|---|
| 2D heatmap network (15 × 64 × 64 per frame) | §4.1.1 | `pose_estimation/heatmaps/network_heatmap.py` |
| Heatmap → 3D pose autoencoder | §4.1.2 | `pose_estimation/model/network.py` (`AutoEncoder`), wrapped by `data/pose_estimator.py` |
| Sync, MAD filtering, windows, features | §3, §4.1.3–4.2 | `data/dataset_estimated15.py` |
| Normalization (train-split statistics only) | §3.2.3 | `data/normalization.py` |
| EgoBalanceFormer | §4.3 | `models/st_transformer_enhanced.py` (`--model_type st_transformer_enhanced`) |
| Training / testing / cross-validation | §5 | `balance_prediction/` |

The autoencoder runs inside the dataset loader. Pose estimates are computed from the saved heatmaps each time a
dataset is built, so no intermediate pose files are needed.

## Repository layout

```
.
├── balance_prediction/          # entry points for CoP prediction
│   ├── train.py, test.py        # one training / test run
│   ├── run_kfold_subjectwise.py # subject-wise k-fold driver (main experiments)
│   ├── run_lopo.py              # leave-one-participant-out driver
│   └── EGOPOSE_TO_BALANCE_PIPELINE.md  # data-flow notes and single-run examples
├── data/                        # datasets, feature construction, normalization
├── models/                      # EgoBalanceFormer and deep-learning baselines
├── utils/                       # CoP metrics
├── splits/                      # sequence lists (train_subjectwise_all.txt = k-fold list)
├── baselines/                   # XGBoost / RandomForest / ExtraTrees / Lasso k-fold runners
├── analysis/                    # trajectory plots, SHAP, integrated gradients, sync checks
├── calibration/                 # VIVE-to-Vicon coordinate transforms
└── pose_estimation/             # egocentric pose estimator, adapted from UnrealEgo
    ├── scripts/heatmaps/        # generate 2D heatmaps per sequence
    ├── scripts/autoencoder/     # train / test the heatmap-to-3D-pose autoencoder
    ├── scripts/profiling/       # params / FLOPs / latency (paper Table 1)
    └── utils/train.txt, test.txt  # pose-estimator train / test split
```

These are not tracked by git: `log/` (checkpoints, run outputs), `tmp/` (per-fold list files), `paper/`, and the dataset.

## 1. Environment

Requires Python ≥ 3.10 and a CUDA GPU. The paper's experiments ran on an RTX 5000 Ada (32 GB) with CUDA 12.8.

```bash
conda create -n egobalance python=3.10 -y
conda activate egobalance
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy scipy scikit-learn matplotlib pillow natsort

# optional
pip install xgboost        # baselines/run_kfold_subjectwise_xgboost.py
pip install captum         # analysis/plot_integrated_gradients.py
pip install thop fvcore    # pose_estimation/scripts/profiling/
```

Tested with Python 3.10.16, PyTorch 2.7.1+cu128, torchvision 0.22.1, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2,
matplotlib 3.10.1, xgboost 3.0.1 and captum 0.8.0. `analysis/plot_shap_analysis.py` needs `shap`, which depends on
numba. Numba requires NumPy ≤ 2.0, so run that script in a separate environment.

Run every command from the repository root, with the environment activated. The drivers start `train.py`/`test.py`
with the `python` on your `PATH`, and they resolve `log/`, `tmp/` and the list files relative to the current directory.

## 2. Data

The experiments use the **VRBalance** dataset (see the paper, ref. [4]). Each recording is one *sequence directory*:

```
<id>-<name>-trial-<k>-<condition>/
├── color_frame_<i>.png         # egocentric RGB, ~22 Hz
├── ground_truth.json           # per frame: timestamp, image_name, joints (Vicon), vrHMD, vrLeft, vrRight
├── balance.json                # balance-board CoP samples (~2 Hz): timestamp, copX, copY
└── custom_pred_heatmaps/       # created in step 3.1: color_frame_<i>_hm15x64x64.npy
```

Cross-validation groups sequences by subject. The subject key is the first two `-`-separated tokens of the
directory name (`<id>-<name>`), so keep that naming.

The list files contain absolute paths from the authors' machine. Point them at your copy of the data:

```bash
sed -i 's#/data/My_Backup/Dataset/gemini-data#/path/to/VRBalance#' \
    splits/*.txt pose_estimation/utils/*.txt
```

| List | Used for |
|---|---|
| `splits/train_subjectwise_all.txt` | CoP k-fold cross-validation (default `--list_file`) |
| `pose_estimation/utils/train.txt` / `test.txt` | training / evaluating the pose autoencoder |
| `pose_estimation/utils/all_sequence.txt` | all sequences with egocentric frames |

## 3. Running the pipeline

### Pretrained weights

The repository does not include checkpoints. Put them at these paths, which are the scripts' defaults:

| Checkpoint | Path | Produced by |
|---|---|---|
| 2D heatmap network, pretrained on EgoPW + SceneEgo | `log/egopwsceneego_heatmap_shared_B16/heatmap_best.ckpt` | not included; training code for it is not part of this repo |
| Heatmap → 3D pose autoencoder | `log/egocentric_ae_15j_monocular/epoch_50_net_AutoEncoder.pth` | step 3.2 |

### 3.1 Generate 2D heatmaps

Generate heatmaps for every sequence used by the pose estimator or by the k-fold list:

```bash
mkdir -p tmp
cat splits/train_subjectwise_all.txt pose_estimation/utils/train.txt pose_estimation/utils/test.txt \
    | sort -u > tmp/heatmap_sequences.txt

python pose_estimation/scripts/heatmaps/generate_custom_heatmaps_from_folders.py \
    --folders_txt tmp/heatmap_sequences.txt \
    --weights log/egopwsceneego_heatmap_shared_B16/heatmap_best.ckpt \
    --output_subdir custom_pred_heatmaps
```

This writes one `color_frame_<i>_hm15x64x64.npy` per frame into `<sequence>/custom_pred_heatmaps/`.

### 3.2 Train the heatmap → 3D pose autoencoder

Skip this step if you already have the autoencoder checkpoint. The flags below differ from the script's defaults, and
they are needed so the output lands where step 3.3 looks for it.

```bash
python pose_estimation/scripts/autoencoder/train_egocentric_autoencoder_from_heatmaps.py \
    --train_list pose_estimation/utils/train.txt \
    --heatmap_subdir custom_pred_heatmaps \
    --num_joints 15 \
    --epochs 50 \
    --log_dir log/egocentric_ae_15j_monocular
```

Evaluate it (MPJPE / PA-MPJPE in mm):

```bash
python pose_estimation/scripts/autoencoder/test_egocentric_autoencoder_from_heatmaps.py \
    --test_list pose_estimation/utils/test.txt \
    --checkpoint log/egocentric_ae_15j_monocular/epoch_50_net_AutoEncoder.pth \
    --heatmap_subdir custom_pred_heatmaps \
    --num_joints 15
```

### 3.3 CoP prediction: subject-wise 10-fold cross-validation

The full EgoBalanceFormer (estimated pose + CoM + HMD):

```bash
python balance_prediction/run_kfold_subjectwise.py \
    --rotation_source pose --no_joint_groups --no_joint_rotations \
    --base_log_dir log/kfold_subjectwise
```

For each fold the driver writes the train/test sequence lists to `tmp/kfold_subjectwise/`. It then trains
`balance_prediction/train.py`, tests the best checkpoint with `balance_prediction/test.py`, and aggregates results:

```
log/kfold_subjectwise/
├── fold_01/ … fold_10/
│   ├── best_model.pth, epoch_*.pth
│   ├── normalizer.npz           # train-split statistics, reused at test time
│   ├── history.json, loss_curve.png
│   └── test_results.json        # metrics + per-sample predictions
└── kfold_summary.json           # per-fold results and mean ± std; the paper reports "metrics_absolute"
```

The defaults that matter are the following. `python balance_prediction/run_kfold_subjectwise.py --help` lists them all.

| Flag | Default | Meaning |
|---|---|---|
| `--list_file` | `splits/train_subjectwise_all.txt` | sequences to split by subject |
| `--n_folds` / `--seed` | 10 / 42 | subject-wise folds |
| `--window_size` / `--fps` | 11 / 22 | input window T and sampling rate |
| `--max_offset_ms` / `--mad_threshold` | 50 / 4.0 | CoP↔pose sync tolerance; MAD outlier threshold |
| `--epochs` / `--early_stop_patience` | 70 / 20 | early stopping on validation RMSE |
| `--loss_type` / `--huber_delta` | huber / 0.5 | training objective |
| `--vr_mode` | hmd_only | VR input: HMD only (controllers excluded) |
| `--align_to_gt` | on | Procrustes-align each estimated pose frame to that frame's Vicon pose; no CLI flag turns it off |
| `--no_pose`, `--no_com`, `--no_vr` | off | drop a modality (ablations) |
| `--no_joint_rotations` | off | disable the rotation branch. `--rotation_source` only matters when this branch is enabled |
| `--forecast_horizon` | 0 | predict CoP *h* steps (×0.5 s) ahead |

A single train/test run without cross-validation is shown in `balance_prediction/EGOPOSE_TO_BALANCE_PIPELINE.md`.

## 4. Reproducing the paper's tables

All rows use `balance_prediction/run_kfold_subjectwise.py` with the default settings above, plus the flags listed.
Give each run its own `--base_log_dir` (and `--tmp_dir`) so runs don't overwrite each other.

**Table 2: methods.** The baselines use HMD motion only.

| Row | Command / extra flags |
|---|---|
| LSTM, GRU, DeepTCN, CNN-LSTM | `--model_type {lstm,gru,deeptcn,cnn_lstm} --no_pose --no_com --no_joint_groups --no_joint_rotations --epochs 50` |
| EgoBalanceFormer (HMD only) | `--no_pose --no_com --no_joint_groups --no_joint_rotations` |
| EgoBalanceFormer (EgoPose + HMD) | `--rotation_source pose --no_joint_groups --no_joint_rotations` |
| XGBoost, RandomForest, ExtraTrees, Lasso | `python baselines/run_kfold_subjectwise_{xgboost,random_forest,extra_trees,lasso}.py --ae_checkpoint log/egocentric_ae_15j_monocular/epoch_50_net_AutoEncoder.pth --no_pose --no_com --no_joint_groups --no_joint_rotations` |

**Table 3: input modalities.** Every row also uses `--no_joint_groups --no_joint_rotations`.

| Row | Extra flags |
|---|---|
| Estimated Pose | `--no_com --no_vr` |
| Pose + CoM | `--no_vr` |
| Pose + HMD | `--no_com` |
| Pose + Rotational | `--no_com --no_vr --rotation_source pose` |
| CoM + Rotational | `--no_pose --no_vr --rotation_source pose` |
| CoM + HMD | `--no_pose` |
| Pose + Rotation + CoM + HMD | `--rotation_source pose` |

**Table 4: Procrustes alignment.** The "Procrustes Aligned" row is the full model from Table 2. The "Raw" row needs
the dataset's `align_to_gt=False`. No CLI flag sets this at present.

**Table 5: forecast horizon.** Use the full-model flags plus `--forecast_horizon {1,2,3,4}` (0.5 s to 2.0 s).

**Table 1: pose-estimator comparison.** The scripts in `pose_estimation/scripts/profiling/` measure parameters,
FLOPs and latency. The autoencoder test script in step 3.2 reports MPJPE / PA-MPJPE.

## 5. Analysis scripts

Each script documents its usage in its module docstring and in `--help`.

| Script | Purpose |
|---|---|
| `analysis/plot_cop_trajectories.py` | predicted vs. ground-truth CoP over time, from `test_results.json` |
| `analysis/plot_sequence_forecast.py` | predicted vs. ground-truth CoP for one selected sequence |
| `analysis/export_cop_predictions_to_csv.py` | `test_results.json` → CSV |
| `analysis/plot_shap_analysis.py`, `plot_integrated_gradients.py` | feature attribution |
| `analysis/check_temporal_alignment.py`, `check_all_temporal_alignment.py`, `report_sync_stats.py` | CoP↔pose synchronization checks |

## Citation

```bibtex
@article{azam2026egobalanceformer,
  title   = {EgoBalanceFormer: Predicting Human Balance from Egocentric Pose and VR-Native Motion},
  author  = {Azam, Md Mushfiqur and Anjum, Nipa and Quarles, John and Desai, Kevin},
  journal = {IEEE Transactions on Visualization and Computer Graphics},
  year    = {2026}
}
```

## Acknowledgments and license

`pose_estimation/` is adapted from [UnrealEgo](https://github.com/hiroyasuakada/UnrealEgo). Its original README and
terms are in `pose_estimation/README.md`. This work was supported by the National Science Foundation under Grants
No. 2316240 and 2403411. The code is released under the MIT License (see `LICENSE`).
