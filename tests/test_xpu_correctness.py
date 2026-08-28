"""Focused correctness for the Triton-free XPU backend."""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SOL_ATTN_XPU_EXPERIMENTAL", "1")


def _load_backend():
    spec = importlib.util.spec_from_file_location(
        "sol_attn_xpu_backend", ROOT / "_xpu_fwd.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summaries(tensor, operation):
    batch, tokens, heads, dim = tensor.shape
    blocks = (tokens + 63) // 64
    values = []
    for block in range(blocks):
        part = tensor[:, block * 64 : min(tokens, (block + 1) * 64)].float()
        reduced = part.mean(dim=1) if operation == "mean" else part.sum(dim=1)
        values.append(reduced)
    return torch.stack(values, dim=2).to(tensor.dtype).reshape(batch, heads, blocks, dim)


def _reference(q, k, v, scale, tau, sink_blocks, sink_q):
    batch, tokens, heads, dim = q.shape
    blocks = (tokens + 63) // 64
    q_centroids = _summaries(q, "mean").float()
    k_centroids = _summaries(k, "mean")
    v_sums = _summaries(v, "sum")
    k_float = k_centroids.float()
    k_mean = k_float.mean(dim=2)
    k_variance = (k_float.square().mean(dim=2) - k_mean.square()).clamp_min(0)
    raw_mean = (q_centroids * k_mean.unsqueeze(2)).sum(dim=-1)
    raw_variance = (q_centroids.square() * k_variance.unsqueeze(2)).sum(dim=-1)
    log2_scale = scale * math.log2(math.e)
    thresholds = raw_mean * log2_scale + tau * torch.sqrt(
        raw_variance * log2_scale * log2_scale + 1.0e-6
    )
    route_scores = torch.einsum(
        "bhqd,bhkd->bhqk", q_centroids, k_centroids.float()
    ) * log2_scale
    routes = route_scores > thresholds.unsqueeze(-1)
    block_ids = torch.arange(blocks, device=q.device)
    routes |= (block_ids[:, None] - block_ids[None, :]).abs()[None, None] <= 1
    routes[:, :, :, sink_blocks[0] : sink_blocks[1]] = True
    routes[:, :, sink_q[0] : sink_q[1], :] = True

    output = torch.empty_like(q)
    for batch_index in range(batch):
        for head in range(heads):
            for query_token in range(tokens):
                query_block = query_token // 64
                query = q[batch_index, query_token, head].float()
                maximum = torch.tensor(float("-inf"), device=q.device)
                denominator = torch.zeros((), device=q.device)
                numerator = torch.zeros(dim, device=q.device)
                for key_block in range(blocks):
                    start = key_block * 64
                    stop = min(tokens, start + 64)
                    if routes[batch_index, head, query_block, key_block]:
                        scores = (
                            k[batch_index, start:stop, head].float() @ query
                        ) * scale
                        block_maximum = scores.max()
                        new_maximum = torch.maximum(maximum, block_maximum)
                        alpha = torch.exp(maximum - new_maximum)
                        probabilities = torch.exp(scores - new_maximum)
                        numerator = numerator * alpha + probabilities @ v[
                            batch_index, start:stop, head
                        ].float()
                        denominator = denominator * alpha + probabilities.sum()
                    else:
                        score = (
                            query @ k_centroids[batch_index, head, key_block].float()
                        ) * scale
                        new_maximum = torch.maximum(maximum, score)
                        alpha = torch.exp(maximum - new_maximum)
                        probability = torch.exp(score - new_maximum)
                        numerator = numerator * alpha + probability * v_sums[
                            batch_index, head, key_block
                        ].float()
                        denominator = denominator * alpha + probability * (stop - start)
                    maximum = new_maximum
                output[batch_index, query_token, head] = (numerator / denominator).to(q.dtype)
    return output


pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(),
    reason="Intel XPU is unavailable",
)


@pytest.mark.parametrize(
    "tokens,heads,tau,sink_blocks,sink_q,seed",
    [
        (31, 1, 1.0, (0, 0), (0, 0), 1),
        (65, 2, 1.3, (0, 0), (0, 0), 2),
        (257, 1, 100.0, (0, 0), (0, 0), 3),
        (193, 1, 100.0, (0, 1), (0, 1), 4),
    ],
)
def test_xpu_matches_algorithm_reference(tokens, heads, tau, sink_blocks, sink_q, seed):
    backend = _load_backend()
    assert backend.backend_available(), backend.backend_error()
    generator = torch.Generator(device="xpu").manual_seed(seed)
    q = torch.randn((1, tokens, heads, 128), generator=generator, device="xpu").bfloat16()
    k = torch.randn((1, tokens, heads, 128), generator=generator, device="xpu").bfloat16()
    v = torch.randn((1, tokens, heads, 128), generator=generator, device="xpu").bfloat16()
    scale = 128**-0.5
    actual = backend.sol_attn(
        q,
        k,
        v,
        scale=scale,
        tau=tau,
        sink_blocks=sink_blocks,
        sink_q=sink_q,
    )
    expected = _reference(q, k, v, scale, tau, sink_blocks, sink_q)
    torch.xpu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)
    has_serial_route_parent = hasattr(
        torch.ops.sol_attn_xpu, "forward_cute_serial_route_parent"
    )
    if os.environ.get("SOL_ATTN_XPU_REQUIRE_SERIAL_ROUTE_PARENT") == "1":
        assert has_serial_route_parent
    if has_serial_route_parent:
        serial_parent, comparison = backend.sol_attn_real_correctness_parent(
            q,
            k,
            v,
            scale=scale,
            tau=tau,
            sink_blocks=sink_blocks,
            sink_q=sink_q,
        )
        torch.xpu.synchronize()
        assert comparison == "candidate_vs_serial_route_parent"
        torch.testing.assert_close(actual, serial_parent, rtol=0, atol=0)
    assert torch.isfinite(actual).all()


def test_xpu_structured_h3_stride_and_zero_adversarial():
    backend = _load_backend()
    assert backend.backend_available(), backend.backend_error()
    tokens = 129
    packed = torch.zeros((1, tokens, 3, 2, 128), device="xpu", dtype=torch.bfloat16)
    q, k, v = (packed[:, :, index, :, :] for index in range(3))
    assert q.stride() == (tokens * 3 * 2 * 128, 3 * 2 * 128, 128, 1)
    actual = backend.sol_attn(q, k, v, tau=1.3)
    torch.xpu.synchronize()
    torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)
    assert actual.stride() == (tokens * 2 * 128, 2 * 128, 128, 1)
