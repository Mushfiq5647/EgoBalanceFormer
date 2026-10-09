#!/usr/bin/env python3
"""
Script to calculate total parameters and FLOPS for UnrealEgo models
"""

import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from thop import profile, clever_format
import argparse
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from model import network
from options.train_options import TrainOptions


def count_parameters(model):
    """Count total and trainable parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def calculate_flops(model, input_shape):
    """Calculate FLOPS using thop library"""
    try:
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape)
        
        # Calculate FLOPS
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return flops, params
    except Exception as e:
        print(f"Error calculating FLOPS: {e}")
        return 0, 0


def create_options(model_name='resnet18', num_heatmap=15):
    """Create options object for model initialization"""
    # Create a mock argument parser to avoid command line parsing
    class MockOpt:
        def __init__(self):
            self.model = 'unrealego_heatmap_shared'
            self.backbone = model_name
            self.num_heatmap = num_heatmap
            self.init_ImageNet = True
            self.init_type = 'normal'
            self.gpu_ids = []
            self.ae_hidden_size = 256
            self.output_nc = num_heatmap
    
    return MockOpt()


def analyze_heatmap_model(model_name='resnet18', num_heatmap=15):
    """Analyze HeatMap model"""
    print(f"\n{'='*60}")
    print(f"HEATMAP MODEL ANALYSIS - {model_name.upper()}")
    print(f"{'='*60}")
    
    opt = create_options(model_name, num_heatmap)
    
    # Create HeatMap model
    heatmap_model = network.define_HeatMap(opt, model=opt.model)
    
    # Count parameters
    total_params, trainable_params = count_parameters(heatmap_model)
    
    print(f"Model: HeatMap_UnrealEgo_Shared with {model_name}")
    print(f"Number of heatmaps: {num_heatmap}")
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Calculate FLOPS for stereo input (left + right images)
    input_shape = (3, 256, 256)  # Single image
    flops_single, _ = calculate_flops(heatmap_model.backbone, input_shape)
    
    # For stereo, we process both left and right images through the same backbone
    # So total FLOPS = 2 * single_image_FLOPS + after_backbone_FLOPS
    flops_after_backbone, _ = calculate_flops(heatmap_model.after_backbone, 
                                            (1024 if model_name in ['resnet50', 'resnet101'] else 512, 8, 8))
    
    total_flops = 2 * flops_single + flops_after_backbone
    
    print(f"FLOPS (single image): {flops_single:,} ({flops_single/1e9:.2f}G)")
    print(f"FLOPS (stereo total): {total_flops:,} ({total_flops/1e9:.2f}G)")
    
    return total_params, trainable_params, total_flops


def analyze_autoencoder_model(model_name='resnet18', num_heatmap=15):
    """Analyze AutoEncoder model"""
    print(f"\n{'='*60}")
    print(f"AUTOENCODER MODEL ANALYSIS - {model_name.upper()}")
    print(f"{'='*60}")
    
    opt = create_options(model_name, num_heatmap)
    
    # Create AutoEncoder model
    autoencoder_model = network.define_AutoEncoder(opt, model=opt.model)
    
    # Count parameters
    total_params, trainable_params = count_parameters(autoencoder_model)
    
    print(f"Model: AutoEncoder")
    print(f"Number of heatmaps: {num_heatmap}")
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Calculate FLOPS
    # Input is concatenated heatmaps from left and right (2 * num_heatmap channels)
    input_shape = (2 * num_heatmap, 64, 64)  # Heatmap input size
    flops, _ = calculate_flops(autoencoder_model, input_shape)
    
    print(f"FLOPS: {flops:,} ({flops/1e9:.2f}G)")
    
    return total_params, trainable_params, flops


def analyze_egoglass_model(model_name='resnet18', num_heatmap=15):
    """Analyze EgoGlass model (separate left/right networks)"""
    print(f"\n{'='*60}")
    print(f"EGOGLASS MODEL ANALYSIS - {model_name.upper()}")
    print(f"{'='*60}")
    
    opt = create_options(model_name, num_heatmap)
    opt.model = 'egoglass'
    
    # Create EgoGlass models (separate left and right)
    heatmap_left = network.define_HeatMap(opt, model=opt.model)
    heatmap_right = network.define_HeatMap(opt, model=opt.model)
    autoencoder = network.define_AutoEncoder(opt, model=opt.model)
    
    # Count parameters
    total_params_left, trainable_params_left = count_parameters(heatmap_left)
    total_params_right, trainable_params_right = count_parameters(heatmap_right)
    total_params_ae, trainable_params_ae = count_parameters(autoencoder)
    
    total_params = total_params_left + total_params_right + total_params_ae
    trainable_params = trainable_params_left + trainable_params_right + trainable_params_ae
    
    print(f"Model: EgoGlass (separate left/right networks)")
    print(f"Number of heatmaps: {num_heatmap}")
    print(f"Left HeatMap parameters: {total_params_left:,} ({total_params_left/1e6:.2f}M)")
    print(f"Right HeatMap parameters: {total_params_right:,} ({total_params_right/1e6:.2f}M)")
    print(f"AutoEncoder parameters: {total_params_ae:,} ({total_params_ae/1e6:.2f}M)")
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Calculate FLOPS
    input_shape = (3, 256, 256)
    flops_left, _ = calculate_flops(heatmap_left, input_shape)
    flops_right, _ = calculate_flops(heatmap_right, input_shape)
    flops_ae, _ = calculate_flops(autoencoder, (2 * num_heatmap, 64, 64))
    
    total_flops = flops_left + flops_right + flops_ae
    
    print(f"FLOPS (left): {flops_left:,} ({flops_left/1e9:.2f}G)")
    print(f"FLOPS (right): {flops_right:,} ({flops_right/1e9:.2f}G)")
    print(f"FLOPS (autoencoder): {flops_ae:,} ({flops_ae/1e9:.2f}G)")
    print(f"Total FLOPS: {total_flops:,} ({total_flops/1e9:.2f}G)")
    
    return total_params, trainable_params, total_flops


def main():
    parser = argparse.ArgumentParser(description='Calculate UnrealEgo model parameters and FLOPS')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'all'],
                       help='Backbone model to analyze')
    parser.add_argument('--num_heatmap', type=int, default=15,
                       help='Number of heatmaps')
    
    args = parser.parse_args()
    
    models_to_test = ['resnet18', 'resnet34', 'resnet50', 'resnet101'] if args.model == 'all' else [args.model]
    
    print("UNREALEGO MODEL ANALYSIS")
    print("=" * 80)
    print(f"Number of heatmaps: {args.num_heatmap}")
    print("=" * 80)
    
    results = {}
    
    for model_name in models_to_test:
        print(f"\n{'#'*80}")
        print(f"ANALYZING {model_name.upper()}")
        print(f"{'#'*80}")
        
        # Analyze UnrealEgo HeatMap Shared model
        hm_params, hm_trainable, hm_flops = analyze_heatmap_model(model_name, args.num_heatmap)
        
        # Analyze AutoEncoder model
        ae_params, ae_trainable, ae_flops = analyze_autoencoder_model(model_name, args.num_heatmap)
        
        # Analyze EgoGlass model (for comparison)
        eg_params, eg_trainable, eg_flops = analyze_egoglass_model(model_name, args.num_heatmap)
        
        # Store results
        results[model_name] = {
            'unrealego_heatmap': {'params': hm_params, 'trainable': hm_trainable, 'flops': hm_flops},
            'unrealego_autoencoder': {'params': ae_params, 'trainable': ae_trainable, 'flops': ae_flops},
            'egoglass': {'params': eg_params, 'trainable': eg_trainable, 'flops': eg_flops}
        }
    
    # Print summary table
    print(f"\n{'='*100}")
    print("SUMMARY TABLE")
    print(f"{'='*100}")
    print(f"{'Model':<20} {'Backbone':<10} {'Component':<20} {'Total Params (M)':<15} {'Trainable (M)':<15} {'FLOPS (G)':<12}")
    print(f"{'-'*100}")
    
    for model_name in models_to_test:
        for component, stats in results[model_name].items():
            print(f"{component:<20} {model_name:<10} {'All':<20} "
                  f"{stats['params']/1e6:<15.2f} {stats['trainable']/1e6:<15.2f} {stats['flops']/1e9:<12.2f}")
    
    # Calculate total UnrealEgo model (HeatMap + AutoEncoder)
    print(f"\n{'='*100}")
    print("COMPLETE UNREALEGO MODEL (HeatMap + AutoEncoder)")
    print(f"{'='*100}")
    print(f"{'Backbone':<10} {'Total Params (M)':<15} {'Trainable (M)':<15} {'FLOPS (G)':<12}")
    print(f"{'-'*60}")
    
    for model_name in models_to_test:
        total_params = results[model_name]['unrealego_heatmap']['params'] + results[model_name]['unrealego_autoencoder']['params']
        total_trainable = results[model_name]['unrealego_heatmap']['trainable'] + results[model_name]['unrealego_autoencoder']['trainable']
        total_flops = results[model_name]['unrealego_heatmap']['flops'] + results[model_name]['unrealego_autoencoder']['flops']
        
        print(f"{model_name:<10} {total_params/1e6:<15.2f} {total_trainable/1e6:<15.2f} {total_flops/1e9:<12.2f}")


if __name__ == "__main__":
    main()
