#!/usr/bin/env python3
"""Diagnostic BMG screen for the exact MiniMax H3 H56 shape/stride contract.

This is an operator screen, not workflow E2E or a publication benchmark.  It
uses random inputs with the canonical packed-QKV storage contract, compares
the CUTE/DPAS candidate against the slow independent SYCL reference once, and
records event timing after warm-up.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch


def event_sample(function) -> float:
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sequence", type=int, default=15787)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--tau", type=float, default=1.3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.xpu.is_available():
        raise RuntimeError("XPU is unavailable")
    if min(args.sequence, args.heads, args.warmup, args.samples) <= 0:
        parser.error("shape, warm-up, and sample counts must be positive")
    for path in (args.candidate, args.reference):
        if not path.is_file():
            raise FileNotFoundError(path)

    torch.ops.load_library(str(args.candidate.resolve()))
    torch.ops.load_library(str(args.reference.resolve()))
    from omni_xpu_kernel import cute

    generator = torch.Generator(device="xpu").manual_seed(args.seed)
    packed = torch.randn(
        (1, args.sequence, 3, args.heads, 128),
        device="xpu",
        dtype=torch.bfloat16,
        generator=generator,
    )
    q, k, v = (packed[:, :, index] for index in range(3))
    scale = 128**-0.5
    op_args = (
        q,
        k,
        v,
        scale,
        args.tau,
        0,
        0,
        0,
        0,
    )

    def prepare():
        return torch.ops.sol_attn_xpu.prepare(*op_args)

    def candidate():
        return torch.ops.sol_attn_xpu.forward_cute(q, k, v, *prepare(), scale)

    prepared = prepare()

    def mainloop():
        return torch.ops.sol_attn_xpu.forward_cute(q, k, v, *prepared, scale)

    q_bhld, k_bhld, v_bhld = (tensor.permute(0, 2, 1, 3) for tensor in (q, k, v))

    def dense_cute():
        return cute.sdp_bhld_d128(q_bhld, k_bhld, v_bhld)

    actual = candidate()
    expected = torch.ops.sol_attn_xpu.forward(*op_args)
    torch.xpu.synchronize()
    delta = (actual.float() - expected.float()).abs()
    correctness = {
        "finite": bool(torch.isfinite(actual).all().item()),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "rmse": float(delta.square().mean().sqrt().item()),
    }
    if len(prepared) == 3:
        route_density = float(prepared[2].float().mean().item())
    elif len(prepared) == 5:
        k_centroids, _, q_centroids, thresholds, key_sinks = prepared
        route_scores = torch.einsum(
            "bhqd,bhkd->bhqk", q_centroids, k_centroids.float()
        ) * (scale * math.log2(math.e))
        routes = route_scores > thresholds.unsqueeze(-1)
        block_ids = torch.arange(routes.shape[-1], device="xpu")
        routes |= (
            block_ids[:, None] - block_ids[None, :]
        ).abs()[None, None] <= 1
        routes |= key_sinks.bool().unsqueeze(-2)
        route_density = float(routes.float().mean().item())
        del route_scores, routes
    else:
        raise RuntimeError(f"unexpected prepare output count: {len(prepared)}")
    del actual, expected, delta

    operations = {
        "candidate_end_to_end_ms": candidate,
        "preprocess_ms": prepare,
        "candidate_mainloop_ms": mainloop,
        "dense_cute_ms": dense_cute,
    }
    for _ in range(args.warmup):
        for function in operations.values():
            function()
    torch.xpu.synchronize()
    timings = {
        name: [event_sample(function) for _ in range(args.samples)]
        for name, function in operations.items()
    }
    report = {
        "kind": "diagnostic_operator_screen",
        "torch": torch.__version__,
        "device": torch.xpu.get_device_name(0),
        "shape_bthd": list(q.shape),
        "stride_bthd": list(q.stride()),
        "tau": args.tau,
        "route_density": route_density,
        "correctness_vs_sycl_reference": correctness,
        "warmup": args.warmup,
        "samples": args.samples,
        "timings": {
            name: {"values_ms": values, "mean_ms": statistics.mean(values)}
            for name, values in timings.items()
        },
        "evidence_limits": [
            "random exact-shape/stride input, not captured model activations",
            "operator diagnostic, not canonical workflow E2E",
            "not a public wheel or installed-image result",
        ],
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
