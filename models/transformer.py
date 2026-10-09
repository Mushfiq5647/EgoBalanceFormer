"""
Transformer Encoder for CoP (Center of Pressure) prediction.

Architecture:
  1. Feature projection  – linear layer maps per-frame raw features → d_model
  2. Prepend [CLS] token – learnable aggregate token (BERT-style)
  3. Positional encoding – sinusoidal, added to (CLS + sequence)
  4. Transformer Encoder – N layers of multi-head self-attention
  5. CLS-token head      – MLP on the [CLS] output → (CoPX, CoPY)

Why Encoder-only (not Encoder-Decoder)?
  We predict a FIXED-SIZE output (2 scalars) from a FIXED-LENGTH input (11 frames).
  Encoder-Decoder adds a separate autoregressive decoder that is only useful for
  variable-length sequence outputs (e.g. translation, captioning).
  The CLS-token approach is equivalent and 2× simpler.

Input per frame (concatenated flat vector, size = input_dim):
  • joint positions            : Jp × 3
  • joint velocities (optional): Jv × 3
  • VR device positions        :     9  (HMD + left + right × 3)
  • VR device velocities (optional): 9
  ─────────────────────────────────────────────────────────────
  input_dim = (Jp*3) + (Jv*3 if used) + 9 + (9 if used)
"""

import torch
import torch.nn as nn

from models.positional_encoding import PositionalEncoding


class TransformerCoPPredictor(nn.Module):
    """
    Args:
        n_pose_joints:  number of pose joints per frame for positions
        n_vel_joints:   number of pose joints per frame for velocities
        n_vr_coords:    total VR position coordinates     (default 9 = 3 devices × 3)
        use_vr_vel:     whether to include VR velocities as input
        d_model:        transformer hidden dimension       (default 128)
        nhead:          attention heads (must divide d_model evenly)
        num_layers:     transformer encoder layers         (default 4)
        dim_feedforward: FFN inner dimension               (default 512)
        dropout:        dropout applied in encoder + PE    (default 0.1)
        window_size:    number of input frames             (default 11)
    """

    def __init__(
        self,
        n_pose_joints:   int   = 8,
        n_vel_joints:    int   = 8,
        n_vr_coords:     int   = 9,
        use_vr_vel:      bool  = True,
        d_model:         int   = 128,
        nhead:           int   = 8,
        num_layers:      int   = 4,
        dim_feedforward: int   = 512,
        dropout:         float = 0.1,
        window_size:     int   = 11,
    ):
        super().__init__()

        self.d_model     = d_model
        self.window_size = window_size
        self.n_pose_joints = n_pose_joints
        self.n_vel_joints = n_vel_joints
        self.use_pose_vel = n_vel_joints > 0
        self.use_vr_vel = use_vr_vel

        # ── raw feature dimension per frame ──────────────────────────────────
        # positions (+ optional velocities) for joints and VR
        pos_dim = n_pose_joints * 3   # joint positions
        vel_dim = n_vel_joints  * 3 if self.use_pose_vel else 0
        vr_dim  = n_vr_coords        # VR positions
        vv_dim  = n_vr_coords if self.use_vr_vel else 0
        self.input_dim = pos_dim + vel_dim + vr_dim + vv_dim

        # ── 1. Per-frame feature projection ──────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── 2. Learnable [CLS] token ─────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── 3. Positional encoding (covers CLS + window_size positions) ──────
        self.pos_enc = PositionalEncoding(d_model, max_len=window_size + 1, dropout=dropout)

        # ── 4. Transformer Encoder ───────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # input shape: (B, T, D)
            norm_first=True,    # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ── 5. Prediction head (CLS output → CoPX, CoPY) ────────────────────
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),   # → [CoPX, CoPY]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        pose_pos: torch.Tensor,   # (B, W, Jp, 3)
        pose_vel: torch.Tensor | None,   # (B, W, Jv, 3) or None
        vr_pos:   torch.Tensor,   # (B, W, 9)
        vr_vel:   torch.Tensor | None,   # (B, W, 9) or None
    ) -> torch.Tensor:
        """
        Returns:
            cop_pred: (B, 2)  — predicted [CoPX, CoPY]
        """
        B, W = pose_pos.shape[:2]
        if pose_pos.shape[2] != self.n_pose_joints:
            raise ValueError(
                f"pose_pos joints mismatch: expected {self.n_pose_joints}, got {pose_pos.shape[2]}"
            )
        parts = []
        if self.use_pose_vel:
            if pose_vel is None:
                raise ValueError("Model expects pose_vel but got None")
            if pose_vel.shape[2] != self.n_vel_joints:
                raise ValueError(
                    f"pose_vel joints mismatch: expected {self.n_vel_joints}, got {pose_vel.shape[2]}"
                )

        # Flatten joint dimensions → (B, W, J*3)
        pp = pose_pos.reshape(B, W, -1)
        parts.append(pp)
        if self.use_pose_vel:
            parts.append(pose_vel.reshape(B, W, -1))
        parts.append(vr_pos)
        if self.use_vr_vel:
            if vr_vel is None:
                raise ValueError("Model expects vr_vel but got None")
            parts.append(vr_vel)

        # Concatenate all active modalities → (B, W, input_dim)
        x = torch.cat(parts, dim=-1)

        # ── 1. Project to d_model ────────────────────────────────────────────
        x = self.input_proj(x)                          # (B, W, D)

        # ── 2. Prepend [CLS] token ───────────────────────────────────────────
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, D)
        x   = torch.cat([cls, x], dim=1)                # (B, W+1, D)

        # ── 3. Positional encoding ───────────────────────────────────────────
        x = self.pos_enc(x)                             # (B, W+1, D)

        # ── 4. Transformer Encoder ───────────────────────────────────────────
        x = self.encoder(x)                             # (B, W+1, D)

        # ── 5. Extract [CLS] token and predict ──────────────────────────────
        cls_out  = x[:, 0]                              # (B, D)
        cop_pred = self.head(cls_out)                   # (B, 2)

        return cop_pred
