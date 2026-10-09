"""
Profile EgoGlass model: FLOPS, latency, and parameter count.

Usage:
    python pose_estimation/scripts/profiling/profile_egoglass_model.py [--use_body_part] [--batch_size 1]

For FLOPs, install: pip install fvcore
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
    def __init__(self, ae):
        super().__init__()
        self.ae = ae

    def forward(self, x):
        return self.ae.predict_pose(x)


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_latency(forward_fn, dummy_input, device: torch.device, warmup: int = 50, repeats: int = 200) -> float:
    if isinstance(dummy_input, (list, tuple)):
        dummy_input = [x.to(device) for x in dummy_input]
    else:
        dummy_input = dummy_input.to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = forward_fn(dummy_input) if not isinstance(dummy_input, (list, tuple)) else forward_fn(*dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            _ = forward_fn(dummy_input) if not isinstance(dummy_input, (list, tuple)) else forward_fn(*dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    return sum(times) / len(times) * 1000  # ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_body_part", action="store_true", help="38ch AE with body part fusion.")
    args = parser.parse_args()

    device = torch.device(args.device)

    from model.network import HeatMap_EgoGlass, AutoEncoder

    opt_hm = SimpleNamespace(num_heatmap=15, init_ImageNet=False)
    net_left = HeatMap_EgoGlass(opt_hm).to(device)
    net_right = HeatMap_EgoGlass(opt_hm).to(device)
    extra_ch = 8 if args.use_body_part else 0
    ae_opt = SimpleNamespace(num_heatmap=15, ae_hidden_size=20)
    ae_net = AutoEncoder(ae_opt, input_channel_scale=2, extra_input_channels=extra_ch).to(device)

    net_left.eval()
    net_right.eval()
    ae_net.eval()

    image = torch.randn(args.batch_size, 3, 256, 256)

    def egoglass_forward(img):
        out_left = net_left(img)
        out_right = net_right(img)
        hm_left = out_left[0] if isinstance(out_left, tuple) else out_left
        hm_right = out_right[0] if isinstance(out_right, tuple) else out_right
        bp_left = out_left[1] if isinstance(out_left, tuple) and len(out_left) > 1 else None
        ae_in = torch.cat([hm_left, hm_right], dim=1)  # 30ch
        if bp_left is not None and args.use_body_part:
            bp_right = out_right[1] if isinstance(out_right, tuple) and len(out_right) > 1 else None
            ae_in = torch.cat([ae_in, bp_left, bp_right], dim=1)  # 38ch
        return ae_net.predict_pose(ae_in)

    n_trainable_hm = count_trainable_params(net_left) + count_trainable_params(net_right)
    n_trainable_ae = count_trainable_params(ae_net)
    n_total_hm = count_all_params(net_left) + count_all_params(net_right)
    n_total_ae = count_all_params(ae_net)

    latency_ms = measure_latency(egoglass_forward, image, device)

    # Try fvcore first (thop has known issues with newer PyTorch - leaves hooks that break fvcore)
    flops_str = "N/A (pip install fvcore)"
    try:
        from fvcore.nn import FlopCountAnalysis

        img_dev = image.to(device)
        flops_left = FlopCountAnalysis(net_left, (img_dev,)).total()
        flops_right = FlopCountAnalysis(net_right, (img_dev,)).total()
        out_left = net_left(img_dev)
        out_right = net_right(img_dev)
        hm_left = out_left[0] if isinstance(out_left, tuple) else out_left
        hm_right = out_right[0] if isinstance(out_right, tuple) else out_right
        bp_left = out_left[1] if isinstance(out_left, tuple) and len(out_left) > 1 else None
        ae_in = torch.cat([hm_left, hm_right], dim=1)
        if bp_left is not None and args.use_body_part:
            bp_right = out_right[1]
            ae_in = torch.cat([ae_in, bp_left, bp_right], dim=1)
        flops_ae = FlopCountAnalysis(_PoseOnlyWrapper(ae_net), (ae_in,)).total()
        total_flops = flops_left + flops_right + flops_ae
        flops_str = f"{total_flops / 1e9:.4f} G (HeatMap L+R: {(flops_left+flops_right)/1e9:.2f}G, AE: {flops_ae/1e9:.2f}G)"
    except Exception:
        pass

    print("=" * 60)
    print("EgoGlass Monocular (image -> heatmaps + body part -> pose)")
    print("=" * 60)
    print(f"  Input shape: [{args.batch_size}, 3, 256, 256]")
    print(f"  Use body part: {args.use_body_part}")
    print(f"  Trainable params: HeatMap (L+R) {n_trainable_hm:,} | AE {n_trainable_ae:,}")
    print(f"  Total params:     HeatMap (L+R) {n_total_hm / 1e6:.2f} M | AE {n_total_ae / 1e6:.2f} M")
    print(f"  Latency:          {latency_ms:.3f} ms  (batch={args.batch_size}, {args.device})")
    print(f"  FLOPs:            {flops_str}")
    print("=" * 60)


if __name__ == "__main__":
    main()
