"""Host-side source contracts for the Triton-free XPU implementation."""

import hashlib
import importlib.util
import json

from pathlib import Path

import torch
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_backend(name):
    backend_path = ROOT / "_xpu_fwd.py"
    spec = importlib.util.spec_from_file_location(name, backend_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parallel_route_candidate_embeds_a_serial_route_parent():
    mainloop = (ROOT / "csrc" / "sol_cute_mainloop.hpp").read_text(
        encoding="utf-8"
    )
    wrapper = (ROOT / "csrc" / "sol_attn_xpu_cute.cpp").read_text(
        encoding="utf-8"
    )

    assert "bool ParallelSharedRoute" in mainloop
    assert "if constexpr (ParallelSharedRoute)" in mainloop
    assert "forward_cute_serial_route_parent" in wrapper
    assert (
        "(SOL_ATTN_BMG_CACHEABLE_EXACT_KV_LOADS != 0),\n"
        "      false,\n      false,\n      true"
    ) in wrapper


def test_cross_query_route_columns_reuse_key_centroids_and_pack_shared_pairs():
    mainloop = (ROOT / "csrc" / "sol_cute_mainloop.hpp").read_text(
        encoding="utf-8"
    )

    assert "bool CrossQueryRouteColumns" in mainloop
    assert "route_columns_for_key" in mainloop
    assert "route_query_centroid_columns" in mainloop
    assert "route_query_centroid_columns[item][query_slot]" in mainloop
    assert "route_query_centroids[query_slot * 128 + dim]" in mainloop
    assert "kRouteQueryCentroidItems" in mainloop
    assert "SOL_ATTN_GROUP_LOAD_ROUTE_K_CENTROID" in mainloop
    assert "data_placement_striped" in mainloop
    assert "group_load(" in mainloop
    assert "address_space_cast<" in mainloop
    assert "address_space::global_space" in mainloop
    assert "route_key_centroid_ptr.get_decorated()" in mainloop
    assert "const float key_value" in mainloop
    assert "uint16_t route_column_pairs" in mainloop
    assert "owner * 2 + column" in mainloop


def test_real_correctness_prefers_a_pinned_independent_reference_then_serial_parent():
    backend = (ROOT / "_xpu_fwd.py").read_text(encoding="utf-8")
    runtime = (ROOT / "__init__.py").read_text(encoding="utf-8")

    assert '"SOL_ATTN_XPU_REAL_REFERENCE_LIBRARY"' in backend
    assert '"SOL_ATTN_XPU_REAL_REFERENCE_SHA256"' in backend
    assert "observed_sha256 != expected_sha256" in backend
    assert "torch.ops.sol_attn_xpu.forward(" in backend
    assert "torch.isclose(" in backend
    assert '"mismatched_elements"' in backend
    assert (
        'return output, "candidate_vs_independent_sycl_reference"'
        in backend
    )
    assert '"forward_cute_serial_route_parent"' in backend
    assert 'return output, "candidate_vs_serial_route_parent"' in backend
    assert "sol_attn_real_correctness_parent as _xpu_real_correctness_parent" in runtime
    assert '"comparison": comparison' in runtime


def test_real_reference_requires_paired_absolute_path_and_exact_sha(
    tmp_path, monkeypatch
):
    backend = _load_backend("sol_attn_xpu_reference_spec_contract")
    reference = tmp_path / "reference.so"
    reference.write_bytes(b"independent-reference-test")
    digest = hashlib.sha256(reference.read_bytes()).hexdigest()

    monkeypatch.setenv(
        "SOL_ATTN_XPU_REAL_REFERENCE_LIBRARY", str(reference)
    )
    monkeypatch.delenv("SOL_ATTN_XPU_REAL_REFERENCE_SHA256", raising=False)
    with pytest.raises(RuntimeError, match="requires both"):
        backend._real_reference_spec()

    monkeypatch.setenv("SOL_ATTN_XPU_REAL_REFERENCE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        backend._real_reference_spec()

    monkeypatch.setenv("SOL_ATTN_XPU_REAL_REFERENCE_SHA256", digest)
    assert backend._real_reference_spec() == (reference.resolve(), digest)


def test_real_correctness_records_elementwise_relative_and_absolute_policy(
    tmp_path, monkeypatch
):
    destination = tmp_path / "within-tolerance.jsonl"
    monkeypatch.setenv("SOL_ATTN_XPU_REAL_CORRECTNESS", str(destination))
    backend = _load_backend("sol_attn_xpu_real_tolerance_contract")
    q = torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16)
    reference = torch.tensor([[[2.0]]], dtype=torch.bfloat16)
    candidate = torch.tensor([[[2.125]]], dtype=torch.bfloat16)
    assert backend.record_real_correctness(candidate, reference, q)
    record = json.loads(destination.read_text(encoding="utf-8"))
    assert record["metrics"]["max_abs"] == 0.125
    assert record["elementwise_tolerance"] == {
        "rtol": 0.05,
        "atol": 0.05,
        "allclose": True,
        "mismatched_elements": 0,
        "total_elements": 1,
        "mismatch_fraction": 0.0,
    }

    rejected_destination = tmp_path / "outside-tolerance.jsonl"
    monkeypatch.setenv(
        "SOL_ATTN_XPU_REAL_CORRECTNESS", str(rejected_destination)
    )
    rejected_backend = _load_backend(
        "sol_attn_xpu_real_tolerance_rejection_contract"
    )
    zero = torch.zeros((1, 1, 1), dtype=torch.bfloat16)
    outside = torch.full_like(zero, 0.0625)
    assert rejected_backend.record_real_correctness(outside, zero, q)
    rejected = json.loads(
        rejected_destination.read_text(encoding="utf-8")
    )
    assert rejected["elementwise_tolerance"]["allclose"] is False
    assert rejected["elementwise_tolerance"]["mismatched_elements"] == 1
    assert rejected["elementwise_tolerance"]["mismatch_fraction"] == 1.0


def test_real_activation_attribution_keeps_route_work_outside_itt_window():
    backend = (ROOT / "_xpu_fwd.py").read_text(encoding="utf-8")
    runtime = (ROOT / "__init__.py").read_text(encoding="utf-8")

    assert '"SOL_ATTN_XPU_ATTRIBUTION"' in backend
    assert '"SOL_ATTN_XPU_ATTRIBUTION_LOG"' in backend
    assert '"SOL_ATTN_XPU_ATTRIBUTION_ITT_LIBRARY"' in backend
    assert "capture_index = _capture_prepared_routes(" in backend
    assert '    _itt_command(itt_library, "resume")' in backend
    assert "        output = torch.ops.sol_attn_xpu.forward_cute(" in backend
    assert "        torch.xpu.synchronize()" in backend
    assert '        _itt_command(itt_library, "pause")' in backend
    assert "    _record_attribution_call(" in backend
    assert '"raw_activation_or_route_persisted": False' in backend
    assert "attribution_enabled as _xpu_attribution_enabled" in runtime
    assert 'extra["capture_context"] = capture_context' in runtime
    assert 'if capture_active or attribution_active or real_correctness:' in runtime


def test_attribution_uses_exact_disjoint_descending_q256_pairs():
    backend = (ROOT / "_xpu_fwd.py").read_text(encoding="utf-8")

    assert "descending_groups = list(reversed(range(group_count)))" in backend
    assert "paired_first = descending_groups[0::2]" in backend
    assert "paired_second = descending_groups[1::2]" in backend
    assert "union[:, :, paired_first, :] & union[:, :, paired_second, :]" in backend
    assert '"paired_q256_pairs_per_batch_head"' in backend
    assert '"paired_q256_workgroups_per_batch_head"' in backend
    assert '"paired_q256_total_workgroups"' in backend
    assert '"paired_q256_second_tile_reused_key_tiles"' in backend
    assert '"paired_q256_second_tile_selected_key_tiles"' in backend
    assert '"paired_q256_second_tile_reuse_fraction"' in backend


def test_paired_q256_scheduler_is_explicit_default_off_and_preserves_order():
    builder = (ROOT / "scripts" / "build_xpu.py").read_text(
        encoding="utf-8"
    )
    wrapper = (ROOT / "csrc" / "sol_attn_xpu_cute.cpp").read_text(
        encoding="utf-8"
    )

    assert '"--paired-q256-scheduler"' in builder
    assert 'args.paired_q256_scheduler and args.q_tile != 256' in builder
    assert 'f"{int(args.paired_q256_scheduler)}"' in builder
    assert "#define SOL_ATTN_PAIRED_Q256_SCHEDULER 0" in wrapper
    assert "#if SOL_ATTN_PAIRED_Q256_SCHEDULER" in wrapper
    assert "struct SolPairedQ256TileScheduler" in wrapper
    assert "cute::ceil_div(q_tiles, 2)" in wrapper
    assert "params.q_tiles - 1 -" in wrapper
    assert "2 * int(::BlockIdxY()) + tile_in_pair" in wrapper
    assert "tile_in_pair < 2 && current_q_tile() >= 0" in wrapper
    assert "static_assert(QTile == 256" in wrapper
    assert "using TileScheduler = SolPairedQ256TileScheduler" in wrapper
    assert "XeFHMAIndividualTileScheduler<>" in wrapper


def test_disjoint_descending_pair_aggregate_uses_the_second_tile_denominator(
    tmp_path,
):
    backend_path = ROOT / "_xpu_fwd.py"
    spec = importlib.util.spec_from_file_location(
        "sol_attn_xpu_route_pair_contract", backend_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    route = torch.zeros((1, 1, 8, 8), dtype=torch.bool)
    route[:, :, :4, :2] = True
    route[:, :, 4:, 1:3] = True
    q = torch.zeros((1, 512, 1, 128), dtype=torch.bfloat16)
    destination = tmp_path / "route.jsonl"
    capture_index = module._capture_route_record(
        route,
        q,
        scale=128**-0.5,
        tau=1.3,
        sink_blocks=(0, 0),
        sink_q=(0, 0),
        context={"block": 0, "sigma": 0.5},
        destination=destination,
    )

    record = json.loads(destination.read_text(encoding="utf-8"))
    aggregate = record["global"]
    assert capture_index == 0
    assert aggregate["paired_q256_pairs_per_batch_head"] == 1
    assert aggregate["paired_q256_workgroups_per_batch_head"] == 1
    assert aggregate["paired_q256_total_workgroups"] == 1
    assert aggregate["paired_q256_second_tile_reused_key_tiles"] == 1
    assert aggregate["paired_q256_second_tile_selected_key_tiles"] == 2
    assert aggregate["paired_q256_second_tile_reuse_fraction"] == 0.5


def test_cute_grf_size_is_an_explicit_build_and_launch_contract():
    builder = (ROOT / "scripts" / "build_xpu.py").read_text(
        encoding="utf-8"
    )
    wrapper = (ROOT / "csrc" / "sol_attn_xpu_cute.cpp").read_text(
        encoding="utf-8"
    )

    assert '"--grf-size"' in builder
    assert 'choices=(128, 256)' in builder
    assert 'f"-DSOL_ATTN_GRF_SIZE={args.grf_size}"' in builder
    assert "#define SOL_ATTN_GRF_SIZE 256" in wrapper
    assert "static constexpr int GrfSize = SOL_ATTN_GRF_SIZE" in wrapper
    assert "launch_on_torch_queue<K, KT::GrfSize, ParentTag>" in wrapper


def test_routed_kv_prefetch_distance_is_explicit_and_bounded():
    builder = (ROOT / "scripts" / "build_xpu.py").read_text(
        encoding="utf-8"
    )
    mainloop = (ROOT / "csrc" / "sol_cute_mainloop.hpp").read_text(
        encoding="utf-8"
    )

    assert '"--routed-kv-prefetch-distance"' in builder
    assert 'choices=(1, 2)' in builder
    assert "SOL_ATTN_ROUTED_KV_PREFETCH_DISTANCE" in mainloop
    assert "int furthest_prefetched_block = -1" in mainloop
    assert (
        "initial_prefetches == SOL_ATTN_ROUTED_KV_PREFETCH_DISTANCE"
        in mainloop
    )
    assert "furthest_prefetched_block + 1" in mainloop
