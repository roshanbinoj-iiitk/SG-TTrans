#!/usr/bin/env python3
"""
SG-TTrans: Latency, Throughput & GPU Memory Profiler.
Measures per-component latency as detailed in Section 3 of the paper.
"""

import time
import torch
import numpy as np
from sg_ttrans.config import SGTransConfig
from sg_ttrans.models.backbone import MobileNetV4SpatialBackbone
from sg_ttrans.models.fusion import CrossModalFusion
from sg_ttrans.models.transformer import TemporalTransformerEncoder
from sg_ttrans.models.heads import ClassificationHead
from sg_ttrans.risk_engine.rsi import compute_rsi

def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Profiling SG-TTrans on: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    cfg = SGTransConfig(sequence_length=60, d_model=256, num_layers=4, num_heads=4)
    backbone = MobileNetV4SpatialBackbone(out_dim=cfg.d_vis).to(device).eval()
    fusion = CrossModalFusion(d_vis=cfg.d_vis, d_geom=cfg.d_geom, d_model=cfg.d_model).to(device).eval()
    transformer = TemporalTransformerEncoder(d_model=cfg.d_model, num_heads=cfg.num_heads, num_layers=cfg.num_layers).to(device).eval()
    head = ClassificationHead(d_model=cfg.d_model, num_classes=cfg.num_classes).to(device).eval()

    num_warmup = 10
    num_runs = 50

    # Input dummies
    single_frame = torch.randn(1, 3, 224, 224, device=device)
    seq_vis = torch.randn(1, 60, cfg.d_vis, device=device)
    seq_geom = torch.randn(1, 60, cfg.d_geom, device=device)
    seq_fused = torch.randn(1, 60, cfg.d_model, device=device)
    pooled_feat = torch.randn(1, cfg.d_model, device=device)

    # 1. Warm-up
    for _ in range(num_warmup):
        _ = backbone(single_frame)
        _ = fusion(seq_vis, seq_geom)
        _ = transformer(seq_fused)
        _ = head(pooled_feat)
        _ = compute_rsi(0.5, 1.0, 90.0, 4.0, 5.0, -2.0, cfg)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # 2. Benchmark MobileNetV4 per frame
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = backbone(single_frame)
    if device.type == "cuda":
        torch.cuda.synchronize()
    lat_backbone = (time.perf_counter() - t0) / num_runs * 1000.0

    # 3. Benchmark Cross-Modal Fusion
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = fusion(seq_vis, seq_geom)
    if device.type == "cuda":
        torch.cuda.synchronize()
    lat_fusion = (time.perf_counter() - t0) / num_runs * 1000.0

    # 4. Benchmark TDDA Transformer (4 layers, 60 frames)
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = transformer(seq_fused)
    if device.type == "cuda":
        torch.cuda.synchronize()
    lat_transformer = (time.perf_counter() - t0) / num_runs * 1000.0

    # 5. Benchmark Classification Head
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = head(pooled_feat)
    if device.type == "cuda":
        torch.cuda.synchronize()
    lat_head = (time.perf_counter() - t0) / num_runs * 1000.0

    # 6. Benchmark RSI engine (CPU math)
    t0 = time.perf_counter()
    for _ in range(num_runs * 10):
        _ = compute_rsi(0.88, 2.2, 90.0, 3.5, 18.0, 0.0, cfg)
    lat_rsi = (time.perf_counter() - t0) / (num_runs * 10) * 1000.0

    total_latency = lat_backbone + lat_fusion + lat_transformer + lat_head + lat_rsi
    effective_fps = 1000.0 / total_latency

    print("\n" + "="*50)
    print(" SG-TTrans Component Latency Breakdown")
    print("="*50)
    print(f" {'Component':<30} {'Latency (ms)':<15}")
    print("-"*50)
    print(f" {'MobileNetV4 Spatial Backbone':<30} {lat_backbone:>8.2f} ms")
    print(f" {'Cross-Modal Fusion':<30} {lat_fusion:>8.2f} ms")
    print(f" {'TDDA 4-Block Transformer':<30} {lat_transformer:>8.2f} ms")
    print(f" {'Driver State Classifier':<30} {lat_head:>8.2f} ms")
    print(f" {'RSI Dynamic Engine':<30} {lat_rsi:>8.3f} ms")
    print("="*50)
    print(f" {'Total Processing Latency':<30} {total_latency:>8.2f} ms")
    print(f" {'Effective Throughput':<30} {effective_fps:>8.1f} FPS")
    print("="*50)

    if device.type == "cuda":
        alloc_mem = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_mem = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f" GPU Memory Allocated: {alloc_mem:.1f} MB | Reserved: {reserved_mem:.1f} MB\n")

if __name__ == "__main__":
    benchmark()
