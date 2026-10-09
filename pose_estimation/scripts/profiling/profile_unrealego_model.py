"""
Profile UnrealEgo model: FLOPS, latency, and parameter count.

Usage:
    python pose_estimation/scripts/profiling/profile_unrealego_model.py --mode autoencoder [--batch_size 1]
    python pose_estimation/scripts/profiling/profile_unrealego_model.py --mode full_stereo

For EgoGlass profiling, use: python pose_estimation/scripts/profiling/profile_egoglass_model.py
For FLOPs, install: pip install thop
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class _PoseOnlyWrapper(torch.nn.Module):
    """Wraps AutoEncoder so thop can profile predict_pose (thop expects nn.Module)."""

    def __init__(self, ae):
        super().__init__()
        self.ae = ae

    def forward(self, x):
        return self.ae.predict_pose(x)


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_latency(
    forward_fn,
    dummy_input,
    device: torch.device,
    warmup: int = 50,
    repeats: int = 200,
) -> float:
    """forward_fn: callable that takes dummy_input (or *dummy_input if tuple)."""
    if isinstance(dummy_input, (list, tuple)):
        dummy_input = [x.to(device) for x in dummy_input]
    else:
        dummy_input = dummy_input.to(device)

    with torch.no_grad():
        for _ in range(warmup):
            if isinstance(dummy_input, (list, tuple)):
                _ = forward_fn(*dummy_input)
            else:
                _ = forward_fn(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            if isinstance(dummy_input, (list, tuple)):
                _ = forward_fn(*dummy_input)
            else:
                _ = forward_fn(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    return sum(times) / len(times) * 1000  # ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="autoencoder",
        choices=["autoencoder", "full_stereo"],
        help="Profile: autoencoder (heatmaps->pose), full_stereo (UnrealEgo images->pose). Use profile_egoglass_model.py for EgoGlass.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_body_part", action="store_true", help="38ch AE (with body part).")
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.mode == "autoencoder":
        from model.network import AutoEncoder

        opt = SimpleNamespace(
            num_heatmap=15,
            ae_hidden_size=20,
        )
        extra_ch = 8 if args.use_body_part else 0
        model = AutoEncoder(opt, input_channel_scale=2, extra_input_channels=extra_ch)
        # Input: [B, 30, 64, 64] or [B, 38, 64, 64] (heatmaps + optional body part)
        in_ch = 30 + extra_ch
        dummy_input = torch.randn(args.batch_size, in_ch, 64, 64)

        model.eval()
        model.to(device)

        def run_model(x):
            return model.predict_pose(x)

        n_trainable = count_trainable_params(model)
        n_total = count_all_params(model)

        latency_ms = measure_latency(run_model, dummy_input, device)

        # FLOPs via thop if available (thop expects nn.Module, not a method)
        flops_str = "N/A (pip install thop)"
        try:
            from thop import profile

            flops, params = profile(
                _PoseOnlyWrapper(model), inputs=(dummy_input.to(device),), verbose=False
            )
            flops_str = f"{flops / 1e9:.4f} G"
        except ImportError:
            pass

        print("=" * 60)
        print("UnrealEgo AutoEncoder (heatmaps -> pose)")
        print("=" * 60)
        print(f"  Input shape: [{args.batch_size}, {in_ch}, 64, 64]")
        print(f"  Trainable params: {n_trainable:,} ({n_trainable / 1e6:.4f} M)")
        print(f"  Total params:     {n_total:,} ({n_total / 1e6:.4f} M)")
        print(f"  Latency:          {latency_ms:.3f} ms  (batch={args.batch_size}, {args.device})")
        print(f"  FLOPs:            {flops_str}")
        print("=" * 60)

    elif args.mode == "full_stereo":
        from model.network import HeatMap_UnrealEgo_Shared, AutoEncoder

        opt = SimpleNamespace(
            num_heatmap=15,
            ae_hidden_size=20,
            model_name="resnet18",
            init_ImageNet=False,
        )
        heatmap_net = HeatMap_UnrealEgo_Shared(opt)
        ae_opt = SimpleNamespace(num_heatmap=15, ae_hidden_size=20)
        ae_net = AutoEncoder(ae_opt, input_channel_scale=2, extra_input_channels=0)

        heatmap_net.eval()
        ae_net.eval()
        heatmap_net.to(device)
        ae_net.to(device)

        # Full pipeline: stereo left+right 256x256 -> heatmaps -> pose
        left = torch.randn(args.batch_size, 3, 256, 256)
        right = torch.randn(args.batch_size, 3, 256, 256)

        def full_forward(l, r):
            hm = heatmap_net(l, r)  # [B, 30, H, W]
            return ae_net.predict_pose(hm)

        n_trainable_hm = count_trainable_params(heatmap_net)
        n_trainable_ae = count_trainable_params(ae_net)
        n_total_hm = count_all_params(heatmap_net)
        n_total_ae = count_all_params(ae_net)

        latency_ms = measure_latency(
            lambda l, r: full_forward(l, r),
            (left, right),
            device,
        )

        flops_str = "N/A (pip install thop)"
        try:
            from thop import profile

            flops_hm, _ = profile(
                heatmap_net, inputs=(left.to(device), right.to(device)), verbose=False
            )
            dummy_hm = heatmap_net(left.to(device), right.to(device))
            flops_ae, _ = profile(_PoseOnlyWrapper(ae_net), inputs=(dummy_hm,), verbose=False)
            flops_str = f"{(flops_hm + flops_ae) / 1e9:.4f} G (HeatMap: {flops_hm / 1e9:.2f}G, AE: {flops_ae / 1e9:.2f}G)"
        except ImportError:
            pass

        print("=" * 60)
        print("UnrealEgo Full Stereo (images -> heatmaps -> pose)")
        print("=" * 60)
        print(f"  Input shape: left/right [{args.batch_size}, 3, 256, 256]")
        print(f"  Trainable params: HeatMap {n_trainable_hm:,} | AE {n_trainable_ae:,}")
        print(f"  Total params:     HeatMap {n_total_hm / 1e6:.2f} M | AE {n_total_ae / 1e6:.2f} M")
        print(f"  Latency:          {latency_ms:.3f} ms  (batch={args.batch_size}, {args.device})")
        print(f"  FLOPs:            {flops_str}")
        print("=" * 60)


if __name__ == "__main__":
    main()
