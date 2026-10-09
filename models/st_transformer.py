"""
SpatioTemporalCoPPredictor
==========================
A multi-modal, two-stage Transformer for CoP prediction.

Inputs
------
  pose_pos : (B, W, J, 3)   19-joint positions, cm, root-relative
  vr_pos   : (B, W, 6)      [Left-HMD, Right-HMD], cm, HMD-relative

Data-flow (vision-inspired design)
------------------------------------

  Pose (B,W,J,3)                  VR (B,W,6)
       │                                │
  ┌────▼──────────────────┐      ┌──────▼──────────────┐
  │  Joint Embedding       │      │  VR Projection       │
  │  Linear(3, d_joint)    │      │  Linear(6, D)        │
  │  + learnable joint-    │      │  + Temporal PE       │
  │    type embed (J,d_j)  │      └──────┬──────────────┘
  └────┬──────────────────┘             │  (B,W,D)
       │  (B,W,J,d_j)                   │
  ┌────▼──────────────────┐      ┌──────▼──────────────┐
  │  SPATIAL Self-Attn    │      │ Temporal Self-Attn  │
  │  (ViT-style: treat    │      │  across W frames    │
  │   joints as tokens)   │      │  (B,W,D)            │
  │  per-frame, B*W batch │      └──────┬──────────────┘
  │  mean-pool → (B,W,D)  │             │
  └────┬──────────────────┘             │
       │                                │
  ┌────▼──────────────────┐             │
  │  Temporal DepthConv   │             │
  │  (local pattern,      │             │
  │   kernel=3 over W)    │             │
  │  + Temporal Self-Attn │             │
  │  (B,W,D)              │             │
  └────┬──────────────────┘             │
       │                                │
       │        ┌───────────────────────┘
       │        │
  ┌────▼────────▼──────────────────────────────┐
  │          Cross-Attention                    │
  │  Q = VR temporal (B,W,D)                   │
  │  K,V = Pose temporal (B,W,D)               │
  │  → cross_out (B,W,D)                        │
  └────────────────────┬───────────────────────┘
                       │
  ┌────────────────────▼───────────────────────┐
  │          Unified Memory                     │
  │  pose_temporal  +  cross_out               │
  │          (B, W, D)                          │
  └────────────────────┬───────────────────────┘
                       │
  ┌────────────────────▼───────────────────────┐
  │       Transformer Decoder                   │
  │  Q  = learnable CoP query  (B, 1, D)        │
  │  K,V = unified memory      (B, W, D)        │
  │        → (B, 1, D)                          │
  └────────────────────┬───────────────────────┘
                       │
  ┌────────────────────▼───────────────────────┐
  │          Prediction Head                    │
  │         MLP → (B, 2)                        │
  └────────────────────────────────────────────┘

Vision techniques used
----------------------
1. ViT-style joint tokenization – each joint = one spatial token per frame
2. Learnable joint-type embeddings – each of the J joints has a unique
   embedding (analogous to patch-position embeddings in ViT)
3. Spatial → Temporal hierarchy – mirrors Video-ViT (ViViT) where spatial
   attention within a frame precedes temporal attention across frames
4. Depthwise 1D temporal conv – local motion-pattern extraction before
   global temporal attention (analogous to CvT's conv-projection)
5. Cross-modal cross-attention – VR queries body-pose memory
   (inspired by cross-modal fusion in multimodal ViT / CLIP)
6. Transformer decoder with learned query – the CoP query "asks" the
   unified memory what it needs, rather than pooling blindly
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

class TemporalPositionalEncoding(nn.Module):
    """Sinusoidal PE over the time (W) axis."""

    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, : x.size(1)])


def _make_encoder(d_model, nhead, ff, dropout, n_layers):
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=ff,
        dropout=dropout, activation="gelu",
        batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(d_model))


def _make_decoder(d_model, nhead, ff, dropout, n_layers):
    layer = nn.TransformerDecoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=ff,
        dropout=dropout, activation="gelu",
        batch_first=True, norm_first=True,
    )
    return nn.TransformerDecoder(layer, n_layers, norm=nn.LayerNorm(d_model))


# ──────────────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────────────

class SpatioTemporalCoPPredictor(nn.Module):
    """
    Args
    ----
    n_joints        : number of pose joints J  (default 19)
    n_vr_coords     : VR feature dimension     (default 6)
    d_joint         : spatial-attention dim    (default 64)
    d_model         : main model dim           (default 256)
    nhead           : attention heads          (default 8)
    n_spatial_layers: spatial encoder layers   (default 2)
    n_temporal_layers: temporal encoder layers (default 3)
    n_cross_layers  : cross-attention layers   (default 2)
    n_decoder_layers: decoder layers           (default 2)
    dim_feedforward : FFN inner dim            (default 512)
    dropout         : dropout rate             (default 0.1)
    window_size     : W                        (default 11)
    """

    def __init__(
        self,
        n_joints:          int   = 19,
        n_vr_coords:       int   = 6,
        d_joint:           int   = 64,
        d_model:           int   = 256,
        nhead:             int   = 8,
        n_spatial_layers:  int   = 2,
        n_temporal_layers: int   = 3,
        n_cross_layers:    int   = 2,
        n_decoder_layers:  int   = 2,
        dim_feedforward:   int   = 512,
        dropout:           float = 0.1,
        window_size:       int   = 11,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.n_joints    = n_joints
        self.d_joint     = d_joint
        self.d_model     = d_model
        self.window_size = window_size

        # ── 1. SPATIAL branch ──────────────────────────────────────────────
        # 1a. Joint coordinate embedding  (3 → d_joint)
        self.joint_coord_embed = nn.Linear(3, d_joint)

        # 1b. Learnable joint-type embeddings  (J distinct identities)
        #     analogous to patch-position embeddings in ViT
        self.joint_type_embed = nn.Embedding(n_joints, d_joint)

        # 1c. Spatial Transformer (within each frame; joints are tokens)
        self.spatial_encoder = _make_encoder(
            d_joint, max(1, d_joint // 32), d_joint * 4, dropout, n_spatial_layers
        )

        # 1d. Project spatial output to d_model
        self.spatial_proj = nn.Sequential(
            nn.Linear(d_joint, d_model),
            nn.LayerNorm(d_model),
        )

        # 1e. Temporal depthwise conv  (local motion patterns, kernel=3)
        #     Depthwise = one filter per channel → cheap + effective
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.Conv1d(d_model, d_model, kernel_size=1),   # pointwise mix
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 1f. Temporal Transformer (across frames)
        self.pose_temporal_pe = TemporalPositionalEncoding(d_model, window_size + 1, dropout)
        self.pose_temporal_encoder = _make_encoder(
            d_model, nhead, dim_feedforward, dropout, n_temporal_layers
        )

        # ── 2. VR branch ───────────────────────────────────────────────────
        self.vr_proj = nn.Sequential(
            nn.Linear(n_vr_coords, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.vr_temporal_pe = TemporalPositionalEncoding(d_model, window_size + 1, dropout)
        self.vr_temporal_encoder = _make_encoder(
            d_model, nhead, dim_feedforward, dropout, n_temporal_layers
        )

        # ── 3. Cross-Attention  (VR queries Pose) ─────────────────────────
        self.cross_attn = _make_decoder(
            d_model, nhead, dim_feedforward, dropout, n_cross_layers
        )

        # ── 4. Unified memory fusion (add + norm) ─────────────────────────
        self.memory_norm = nn.LayerNorm(d_model)

        # ── 5. Transformer Decoder (learnable CoP query) ──────────────────
        self.cop_query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cop_query, std=0.02)

        self.decoder = _make_decoder(
            d_model, nhead, dim_feedforward, dropout, n_decoder_layers
        )

        # ── 6. Prediction Head ─────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        pose_pos: torch.Tensor,         # (B, W, J, 3)
        pose_vel: torch.Tensor | None,  # unused, kept for API compatibility
        vr_pos:   torch.Tensor,         # (B, W, 6)
        vr_vel:   torch.Tensor | None,  # unused
    ) -> torch.Tensor:
        """Returns cop_pred (B, 2)."""
        B, W, J, _ = pose_pos.shape

        # ── 1. Spatial branch ─────────────────────────────────────────────
        # 1a. Embed coordinates + add joint-type embeddings
        j_idx = torch.arange(J, device=pose_pos.device)        # (J,)
        coord_emb = self.joint_coord_embed(pose_pos)            # (B,W,J,d_j)
        type_emb  = self.joint_type_embed(j_idx)               # (J,d_j)
        x_spatial = coord_emb + type_emb.unsqueeze(0).unsqueeze(0)  # (B,W,J,d_j)

        # 1b. Spatial Transformer: batch over (B*W) frames, joints as tokens
        x_spatial = x_spatial.reshape(B * W, J, self.d_joint)  # (B*W, J, d_j)
        x_spatial = self.spatial_encoder(x_spatial)             # (B*W, J, d_j)
        x_spatial = x_spatial.mean(dim=1)                       # (B*W, d_j)  mean-pool joints
        x_spatial = x_spatial.reshape(B, W, self.d_joint)       # (B,W,d_j)

        # 1c. Project to d_model
        pose_feat = self.spatial_proj(x_spatial)                # (B,W,D)

        # 1d. Temporal depthwise conv  (B, D, W) → (B, W, D)
        pose_feat = self.temporal_conv(
            pose_feat.transpose(1, 2)                           # (B,D,W)
        ).transpose(1, 2)                                       # (B,W,D)

        # 1e. Temporal Transformer over W frames
        pose_feat = self.pose_temporal_pe(pose_feat)            # (B,W,D)
        pose_feat = self.pose_temporal_encoder(pose_feat)       # (B,W,D)

        # ── 2. VR branch ──────────────────────────────────────────────────
        vr_feat = self.vr_proj(vr_pos)                          # (B,W,D)
        vr_feat = self.vr_temporal_pe(vr_feat)                  # (B,W,D)
        vr_feat = self.vr_temporal_encoder(vr_feat)             # (B,W,D)

        # ── 3. Cross-Attention: VR queries Pose ───────────────────────────
        cross_out = self.cross_attn(
            tgt=vr_feat,                                        # Q: (B,W,D)
            memory=pose_feat,                                   # K,V: (B,W,D)
        )                                                       # (B,W,D)

        # ── 4. Unified memory: pose + cross-attended VR ───────────────────
        memory = self.memory_norm(pose_feat + cross_out)        # (B,W,D)

        # ── 5. Transformer Decoder with learnable CoP query ───────────────
        cop_q = self.cop_query.expand(B, -1, -1)               # (B,1,D)
        decoded = self.decoder(tgt=cop_q, memory=memory)       # (B,1,D)

        # ── 6. Predict ────────────────────────────────────────────────────
        cop_pred = self.head(decoded[:, 0])                     # (B,2)
        return cop_pred
