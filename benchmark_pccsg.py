"""
PC-CSG Component Latency Benchmark.

Profiles each component of the PC-CSG pipeline on available hardware.
"""

import sys
import time
from typing import Dict

import numpy as np
import torch

from pc_csg.config import PCCSGConfig
from pc_csg.models.pc_csg_net import PCCSGNetwork
from pc_csg.models.csga import CounterfactualSceneGraphAttention
from pc_csg.models import PhysicsInformedKinematicValidator, KinematicBicycleModel
from pc_csg.risk_engine import CRTEngine


def benchmark_component(
    name: str,
    fn,
    num_warmup: int = 50,
    num_iters: int = 200,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Benchmark a single component, return mean latency in ms."""
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed runs
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(num_iters):
        with torch.no_grad():
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return (elapsed / num_iters) * 1000.0  # ms


def run_benchmark(device_str: str = "auto") -> Dict[str, float]:
    """Run full PC-CSG component benchmark."""
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    config = PCCSGConfig()
    B = 1
    N = 10
    K = config.num_counterfactuals
    H = config.prediction_horizon
    E = N * (N - 1)

    print("=" * 60)
    print(" PC-CSG Component Latency Benchmark")
    print("=" * 60)
    print(f" Device: {device}")
    if device.type == "cuda":
        print(f" GPU: {torch.cuda.get_device_name(0)}")
    print(f" Agents: {N}, Counterfactuals: {K}, Horizon: {H}")
    print("=" * 60)

    # Create inputs
    node_feat = torch.randn(B, N, config.node_feature_dim, device=device)
    edge_index = torch.zeros(2, E, dtype=torch.long, device=device)
    idx = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                edge_index[0, idx] = i
                edge_index[1, idx] = j
                idx += 1
    edge_attr = torch.randn(B, E, config.edge_feature_dim, device=device)
    hazard_w = torch.rand(B, N, device=device)
    node_mask = torch.ones(B, N, dtype=torch.bool, device=device)

    results = {}

    # 1. CSGA
    csga = CounterfactualSceneGraphAttention(
        node_dim=config.node_feature_dim,
        edge_dim=config.edge_feature_dim,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_counterfactuals=config.num_counterfactuals,
        dropout=0.0,
    ).to(device).eval()

    lat = benchmark_component(
        "CSGA",
        lambda: csga(node_feat, edge_index, edge_attr, hazard_w, node_mask),
        device=device,
    )
    results["Counterfactual Scene Graph Attention (CSGA)"] = lat

    # 2. Kinematic Bicycle Model
    bicycle = KinematicBicycleModel(config).to(device).eval()
    init_state = torch.randn(B * N * K, 4, device=device)
    controls = torch.randn(B * N * K, H, 2, device=device)

    lat = benchmark_component(
        "Bicycle",
        lambda: bicycle(init_state, controls),
        device=device,
    )
    results["Kinematic Bicycle Model Propagation"] = lat

    # 3. PIKV
    pikv = PhysicsInformedKinematicValidator(config).to(device).eval()
    traj = torch.randn(B * N * K, H, 2, device=device)
    speeds = torch.rand(B * N * K, device=device) * 20

    lat = benchmark_component(
        "PIKV",
        lambda: pikv(traj, speeds),
        device=device,
    )
    results["Physics-Informed Kinematic Validator (PIKV)"] = lat

    # 4. CRT Engine
    crt_engine = CRTEngine(config).to(device).eval()
    cf_risk = torch.rand(B, N, K, device=device)
    pikv_val = torch.ones(B, N, K, dtype=torch.bool, device=device)

    lat = benchmark_component(
        "CRT",
        lambda: crt_engine(cf_risk, pikv_val, hazard_w, node_mask),
        device=device,
    )
    results["CRT Engine & Intervention Arbiter"] = lat

    # 5. Full pipeline
    net = PCCSGNetwork(config).to(device).eval()

    lat = benchmark_component(
        "Full",
        lambda: net(node_feat, edge_index, edge_attr, hazard_w, node_mask, node_feat),
        device=device,
    )
    results["Total PC-CSG Pipeline"] = lat

    # Print results
    print(f" {'Component':<50} {'Latency (ms)':>12}")
    print("-" * 64)
    total = 0.0
    for name, ms in results.items():
        if name != "Total PC-CSG Pipeline":
            print(f" {name:<50} {ms:>10.3f} ms")
            total += ms
    print("-" * 64)
    total_pipeline = results["Total PC-CSG Pipeline"]
    print(f" {'Total PC-CSG Pipeline':<50} {total_pipeline:>10.3f} ms")
    print(f" {'Effective Throughput':<50} {1000.0/total_pipeline:>10.1f} FPS")
    print("=" * 60)

    if device.type == "cuda":
        mem_alloc = torch.cuda.memory_allocated() / 1e6
        mem_reserved = torch.cuda.memory_reserved() / 1e6
        print(f" GPU Memory Allocated: {mem_alloc:.1f} MB | Reserved: {mem_reserved:.1f} MB")
        print("=" * 60)

    return results


if __name__ == "__main__":
    run_benchmark()
