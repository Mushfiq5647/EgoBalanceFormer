"""
SpatioTemporalCoPPredictorEnhanced
===================================
Enhanced version that uses:
  - Smoothed pose positions + velocities
  - Smoothed VR positions + velocities
  - Joint group aggregates (5 groups)
  - Center of mass (3D)

Architecture:
  1. Spatial branch (joints)
     - Joint embedding + spatial attention
     - Temporal conv + temporal attention
  2. Joint groups branch (5 groups)
     - Group embedding + temporal attention
  3. Center of mass branch
     - MLP + temporal attention
  4. VR branch
     - VR pos+vel concat → temporal attention
  5. Multi-modal fusion
     - Cross-attention between all branches
  6. Decoder with learnable CoP query
  7. Prediction head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPositionalEncoding(nn.Module):
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
        self.register_buffer("pe", pe.unsqueeze(0))

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


class SpatioTemporalCoPPredictorEnhanced(nn.Module):
    """
    Args:
        n_joints:          19
        n_groups:          5 (joint groups)
        n_vr_coords:       6
        d_joint:           spatial dim for joints (default 64)
        d_model:           main model dim (default 256)
        nhead:             attention heads (default 8)
        n_spatial_layers:  spatial encoder layers (default 2)
        n_temporal_layers: temporal encoder layers (default 3)
        n_fusion_layers:   cross-attention fusion layers (default 2)
        n_decoder_layers:  decoder layers (default 2)
        dim_feedforward:   FFN inner dim (default 512)
        dropout:           dropout rate (default 0.1)
        window_size:       W (default 11)
    """

    def __init__(
        self,
        n_joints:          int   = 19,
        n_groups:          int   = 5,
        n_vr_coords:       int   = 6,
        d_joint:           int   = 64,
        d_model:           int   = 256,
        nhead:             int   = 8,
        n_spatial_layers:  int   = 2,
        n_temporal_layers: int   = 3,
        n_fusion_layers:   int   = 2,
        n_decoder_layers:  int   = 2,
        dim_feedforward:   int   = 512,
        dropout:           float = 0.1,
        window_size:       int   = 11,
        output_activation: str   = "linear",
        use_joint_groups:  bool  = True,
        use_joint_rotations: bool = True,
        use_rotation_vel: bool = True,
        use_pose_vel: bool = True,
        use_com: bool = True,
    ):
        super().__init__()
        assert d_model % nhead == 0

        self.n_joints    = n_joints
        self.n_groups    = n_groups
        self.n_vr_coords = n_vr_coords
        self.use_vr      = n_vr_coords > 0
        self.d_joint     = d_joint
        self.d_model     = d_model
        self.window_size = window_size
        self.output_activation = output_activation
        self.use_joint_groups = use_joint_groups
        self.use_joint_rotations = use_joint_rotations
        self.use_rotation_vel = use_rotation_vel
        self.use_pose_vel = use_pose_vel
        self.use_com = use_com

        # ── 1. JOINT SPATIAL BRANCH ──────────────────────────────────────
        # 1a. Joint coord + vel embedding
        self.joint_pos_embed = nn.Linear(3, d_joint)
        if self.use_pose_vel:
            self.joint_vel_embed = nn.Linear(3, d_joint)
        else:
            self.joint_vel_embed = None
        self.joint_type_embed = nn.Embedding(n_joints, d_joint)
        self.joint_pool_attn = nn.Linear(d_joint, 1)

        # 1b. Spatial Transformer
        self.spatial_encoder = _make_encoder(
            d_joint, max(1, d_joint // 32), d_joint * 4, dropout, n_spatial_layers
        )
        self.spatial_proj = nn.Sequential(
            nn.Linear(d_joint, d_model),
            nn.LayerNorm(d_model),
        )

        # 1c. Temporal processing
        self.joint_temporal_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.joint_temporal_pe = TemporalPositionalEncoding(d_model, window_size + 1, dropout)
        self.joint_temporal_encoder = _make_encoder(
            d_model, nhead, dim_feedforward, dropout, n_temporal_layers
        )

        # ── 2. JOINT GROUPS BRANCH ────────────────────────────────────────
        if self.use_joint_groups:
            self.group_embed = nn.Sequential(
                nn.Linear(3, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, d_model),
                nn.LayerNorm(d_model),
            )
            self.group_type_embed = nn.Embedding(n_groups, d_model)
            self.group_temporal_pe = TemporalPositionalEncoding(d_model, window_size + 1, dropout)
            self.group_temporal_encoder = _make_encoder(
                d_model, nhead, dim_feedforward, dropout, n_temporal_layers
            )
            self.group_pool_attn = nn.Linear(d_model, 1)
        else:
            self.group_embed = None
            self.group_type_embed = None
            self.group_temporal_pe = None
            self.group_temporal_encoder = None
            self.group_pool_attn = None

        # ── 3. CENTER OF MASS BRANCH (position + velocity) ─────────────────
        if self.use_com:
            self.com_embed = nn.Sequential(
                nn.Linear(6, d_model),  # concat [com, com_vel] -> (B, W, 6)
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.com_temporal_pe = TemporalPositionalEncoding(d_model, window_size + 1, dropout)
            self.com_temporal_encoder = _make_encoder(
                d_model, nhead, dim_feedforward, dropout, n_temporal_layers
            )
        else:
            self.com_embed = None
            self.com_temporal_pe = None
            self.com_temporal_encoder = None

        # ── 4. JOINT ROTATIONS BRANCH ─────────────────────────────────────
        # Input: (B, W, 8, 3) - 8 lower-body joints, 3 Euler angles each → flatten to (B, W, 24)
        if self.use_joint_rotations:
            rot_in_dim = 48 if self.use_rotation_vel else 24
            self.rotation_embed = nn.Sequential(
                nn.Linear(rot_in_dim, d_model),  # [rot, rot_vel] or rot only
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.rotation_temporal_pe = TemporalPositionalEncoding(d_model, window_size + 1, dropout)
            self.rotation_temporal_encoder = _make_encoder(
                d_model, nhead, dim_feedforward, dropout, n_temporal_layers
            )
        else:
            self.rotation_embed = None
            self.rotation_temporal_pe = None
            self.rotation_temporal_encoder = None

        # ── 5. VR BRANCH ──────────────────────────────────────────────────
        if self.use_vr:
            self.vr_embed = nn.Sequential(
                nn.Linear(n_vr_coords * 2, d_model),  # pos + vel concatenated
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.vr_temporal_pe = TemporalPositionalEncoding(d_model, window_size + 1, dropout)
            self.vr_temporal_encoder = _make_encoder(
                d_model, nhead, dim_feedforward, dropout, n_temporal_layers
            )
        else:
            self.vr_embed = None
            self.vr_temporal_pe = None
            self.vr_temporal_encoder = None

        # ── 6. TRUE CROSS-ATTENTION: joint <-> CoM ───────────────────────
        # These are real cross-attention blocks (different tgt/memory).
        self.joint_from_com = _make_decoder(
            d_model, nhead, dim_feedforward, dropout, n_fusion_layers
        )
        self.com_from_joint = _make_decoder(
            d_model, nhead, dim_feedforward, dropout, n_fusion_layers
        )
        self.cross_norm_joint = nn.LayerNorm(d_model)
        self.cross_norm_com = nn.LayerNorm(d_model)
        self.cross_norm_joint_com = nn.LayerNorm(d_model)

        # Explicit joint <-> group cross-attention
        if self.use_joint_groups:
            self.joint_from_group = _make_decoder(
                d_model, nhead, dim_feedforward, dropout, n_fusion_layers
            )
            self.group_from_joint = _make_decoder(
                d_model, nhead, dim_feedforward, dropout, n_fusion_layers
            )
            self.cross_norm_joint_group = nn.LayerNorm(d_model)
            self.cross_norm_group = nn.LayerNorm(d_model)
        else:
            self.joint_from_group = None
            self.group_from_joint = None
            self.cross_norm_joint_group = None
            self.cross_norm_group = None

        # Explicit joint <-> rotation cross-attention
        if self.use_joint_rotations:
            self.joint_from_rot = _make_decoder(
                d_model, nhead, dim_feedforward, dropout, n_fusion_layers
            )
            self.rot_from_joint = _make_decoder(
                d_model, nhead, dim_feedforward, dropout, n_fusion_layers
            )
            self.cross_norm_joint_rot = nn.LayerNorm(d_model)
            self.cross_norm_rot = nn.LayerNorm(d_model)
        else:
            self.joint_from_rot = None
            self.rot_from_joint = None
            self.cross_norm_joint_rot = None
            self.cross_norm_rot = None

        # Explicit joint <-> VR cross-attention
        if self.use_vr:
            self.joint_from_vr = _make_decoder(
                d_model, nhead, dim_feedforward, dropout, n_fusion_layers
            )
            self.vr_from_joint = _make_decoder(
                d_model, nhead, dim_feedforward, dropout, n_fusion_layers
            )
            self.cross_norm_joint_vr = nn.LayerNorm(d_model)
            self.cross_norm_vr = nn.LayerNorm(d_model)
        else:
            self.joint_from_vr = None
            self.vr_from_joint = None
            self.cross_norm_joint_vr = None
            self.cross_norm_vr = None

        # ── 7. MULTI-MODAL FUSION (self-attn over fused tokens) ───────────
        self.fusion_cross_attn = _make_decoder(
            d_model, nhead, dim_feedforward, dropout, n_fusion_layers
        )
        self.fusion_norm = nn.LayerNorm(d_model)

        # Modality identity tokens (improve disambiguation in fusion attention)
        self.mod_joint = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mod_group = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mod_com = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mod_rot = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mod_vr = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mod_joint, std=0.02)
        nn.init.trunc_normal_(self.mod_group, std=0.02)
        nn.init.trunc_normal_(self.mod_com, std=0.02)
        nn.init.trunc_normal_(self.mod_rot, std=0.02)
        nn.init.trunc_normal_(self.mod_vr, std=0.02)

        # ── 8. DECODER (learnable CoP query) ──────────────────────────────
        self.cop_query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cop_query, std=0.02)
        self.decoder = _make_decoder(
            d_model, nhead, dim_feedforward, dropout, n_decoder_layers
        )

        # ── 9. PREDICTION HEAD ────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

        self._init_weights()

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

    def forward(
        self,
        pose_pos:     torch.Tensor,  # (B, W, 8, 3) lower-body only
        pose_vel:     torch.Tensor | None,  # (B, W, 8, 3) or None
        vr_pos:          torch.Tensor | None,  # (B, W, C) or None
        vr_vel:          torch.Tensor | None,  # (B, W, C) or None
        joint_groups:    torch.Tensor | None,  # (B, W, 5, 3) or None
        joint_rotations: torch.Tensor | None,  # (B, W, 8, 3) lower-body rotations (Euler, radians)
        com:             torch.Tensor | None,  # (B, W, 3) or None
        joint_rot_vel:   torch.Tensor | None = None,  # (B, W, 8, 3) rad/s
        com_vel:         torch.Tensor | None = None,  # (B, W, 3) CoM velocity cm/s
    ) -> torch.Tensor:
        """Returns cop_pred (B, 2)."""
        B, W, J, _ = pose_pos.shape

        # ── 1. Joint spatial + temporal branch ────────────────────────────
        j_idx = torch.arange(J, device=pose_pos.device)
        pos_emb = self.joint_pos_embed(pose_pos)                      # (B,W,J,d_j)
        if self.use_pose_vel:
            if pose_vel is None:
                raise ValueError("Pose-velocity branch enabled but pose_vel is None")
            vel_emb = self.joint_vel_embed(pose_vel)                  # (B,W,J,d_j)
        else:
            vel_emb = torch.zeros_like(pos_emb)
        type_emb = self.joint_type_embed(j_idx)                       # (J,d_j)
        x_spatial = pos_emb + vel_emb + type_emb.unsqueeze(0).unsqueeze(0)

        x_spatial = x_spatial.reshape(B * W, J, self.d_joint)
        x_spatial = self.spatial_encoder(x_spatial)
        # Attention pooling preserves asymmetric joint contributions better than mean pooling.
        alpha_joint = torch.softmax(self.joint_pool_attn(x_spatial), dim=1)  # (B*W, J, 1)
        x_spatial = torch.sum(alpha_joint * x_spatial, dim=1)                # (B*W, d_j)
        x_spatial = x_spatial.reshape(B, W, self.d_joint)

        joint_feat = self.spatial_proj(x_spatial)                     # (B,W,D)
        joint_feat = self.joint_temporal_conv(
            joint_feat.transpose(1, 2)
        ).transpose(1, 2)
        joint_feat = self.joint_temporal_pe(joint_feat)
        joint_feat = self.joint_temporal_encoder(joint_feat)          # (B,W,D)

        # ── 2. Joint groups branch ────────────────────────────────────────
        if self.use_joint_groups:
            if joint_groups is None:
                raise ValueError("Joint-group branch enabled but joint_groups is None")
            g_idx = torch.arange(self.n_groups, device=joint_groups.device)
            group_emb = self.group_embed(joint_groups)                    # (B,W,5,D)
            group_type = self.group_type_embed(g_idx)                     # (5,D)
            group_emb = group_emb + group_type.unsqueeze(0).unsqueeze(0)
            group_emb = group_emb.reshape(B * W, self.n_groups, self.d_model)
            alpha_group = torch.softmax(self.group_pool_attn(group_emb), dim=1)  # (B*W,G,1)
            group_emb = torch.sum(alpha_group * group_emb, dim=1).reshape(B, W, self.d_model)
            group_feat = self.group_temporal_pe(group_emb)
            group_feat = self.group_temporal_encoder(group_feat)          # (B,W,D)

        # ── 3. Center of mass branch (position + velocity) ────────────────
        if self.use_com:
            if com is None:
                raise ValueError("CoM branch enabled but com is None")
            if com_vel is not None:
                com_input = torch.cat([com, com_vel], dim=-1)         # (B, W, 6)
            else:
                com_input = torch.cat([com, torch.zeros_like(com, device=com.device)], dim=-1)
            com_emb = self.com_embed(com_input)                       # (B,W,D)
            com_feat = self.com_temporal_pe(com_emb)
            com_feat = self.com_temporal_encoder(com_feat)            # (B,W,D)

        # ── 4. Joint rotations branch ─────────────────────────────────────
        if self.use_joint_rotations and joint_rotations is not None:
            B, W, J, C = joint_rotations.shape  # (B, W, 8, 3)
            rot_flat = joint_rotations.reshape(B, W, J * C)               # (B, W, 24)
            if self.use_rotation_vel:
                if joint_rot_vel is None:
                    raise ValueError("Rotation-velocity enabled but joint_rot_vel is None")
                rot_vel_flat = joint_rot_vel.reshape(B, W, J * C)         # (B, W, 24)
                rot_in = torch.cat([rot_flat, rot_vel_flat], dim=-1)      # (B, W, 48)
            else:
                rot_in = rot_flat
            rot_emb = self.rotation_embed(rot_in)                         # (B, W, D)
            rot_feat = self.rotation_temporal_pe(rot_emb)
            rot_feat = self.rotation_temporal_encoder(rot_feat)           # (B, W, D)
        else:
            rot_feat = None

        # ── 5. VR branch (pos + vel) ──────────────────────────────────────
        if self.use_vr:
            if vr_pos is None or vr_vel is None:
                raise ValueError("VR branch enabled but vr_pos/vr_vel is None")
            vr_concat = torch.cat([vr_pos, vr_vel], dim=-1)           # (B,W,2*C)
            vr_emb = self.vr_embed(vr_concat)                         # (B,W,D)
            vr_feat = self.vr_temporal_pe(vr_emb)
            vr_feat = self.vr_temporal_encoder(vr_feat)               # (B,W,D)

        # ── 6. True cross-attention: joint <-> CoM ───────────────────────
        # joint_ctx: joint tokens updated using CoM as memory
        # com_ctx:   CoM tokens updated using joint as memory
        if self.use_com:
            joint_ctx = self.joint_from_com(tgt=joint_feat, memory=com_feat)
            joint_ctx = self.cross_norm_joint_com(joint_ctx)
            com_ctx = self.com_from_joint(tgt=com_feat, memory=joint_feat)
            com_ctx = self.cross_norm_com(com_ctx)
        else:
            joint_ctx = joint_feat
            com_ctx = None

        # Explicit pairwise cross-attention with joint branch for other modalities.
        if self.use_joint_groups:
            joint_ctx = self.joint_from_group(tgt=joint_ctx, memory=group_feat)
            joint_ctx = self.cross_norm_joint_group(joint_ctx)
            group_ctx = self.group_from_joint(tgt=group_feat, memory=joint_ctx)
            group_ctx = self.cross_norm_group(group_ctx)
        else:
            group_ctx = None

        if rot_feat is not None:
            joint_ctx = self.joint_from_rot(tgt=joint_ctx, memory=rot_feat)
            joint_ctx = self.cross_norm_joint_rot(joint_ctx)
            rot_ctx = self.rot_from_joint(tgt=rot_feat, memory=joint_ctx)
            rot_ctx = self.cross_norm_rot(rot_ctx)
        else:
            rot_ctx = None

        if self.use_vr:
            joint_ctx = self.joint_from_vr(tgt=joint_ctx, memory=vr_feat)
            joint_ctx = self.cross_norm_joint_vr(joint_ctx)
            vr_ctx = self.vr_from_joint(tgt=vr_feat, memory=joint_ctx)
            vr_ctx = self.cross_norm_vr(vr_ctx)
        else:
            vr_ctx = None

        # ── 7. Multi-modal token fusion ───────────────────────────────────
        feat_list = [joint_ctx + self.mod_joint]
        if com_ctx is not None:
            feat_list.append(com_ctx + self.mod_com)
        if group_ctx is not None:
            feat_list.append(group_ctx + self.mod_group)
        if rot_ctx is not None:
            feat_list.append(rot_ctx + self.mod_rot)
        if vr_ctx is not None:
            feat_list.append(vr_ctx + self.mod_vr)
        all_feats = torch.cat(feat_list, dim=1)

        # Dynamic behavior:
        # - Pose-only: joint temporal self-attention is already done; skip extra fusion.
        # - Multi-branch: run fusion attention over concatenated branch tokens.
        if len(feat_list) == 1:
            fused = all_feats
        else:
            fused = self.fusion_cross_attn(tgt=all_feats, memory=all_feats)
            fused = self.fusion_norm(fused)

        # ── 7. Decoder with learnable CoP query ───────────────────────────
        cop_q = self.cop_query.expand(B, -1, -1)                      # (B,1,D)
        decoded = self.decoder(tgt=cop_q, memory=fused)               # (B,1,D)

        # ── 7. Predict ────────────────────────────────────────────────────
        cop_pred = self.head(decoded[:, 0])                           # (B,2)
        if self.output_activation == "tanh":
            cop_pred = torch.tanh(cop_pred)
        return cop_pred
