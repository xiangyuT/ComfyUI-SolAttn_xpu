"""Host-side contracts for the packaged, Triton-free XPU adapter."""

import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(
    os.environ.get("SOL_ATTN_TEST_ROOT", Path(__file__).resolve().parents[1])
).resolve()


def _load_backend(name):
    backend_path = ROOT / "_xpu_fwd.py"
    spec = importlib.util.spec_from_file_location(name, backend_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_xpu_kernel_ownership_is_in_omni_xpu_kernel():
    backend = (ROOT / "_xpu_fwd.py").read_text(encoding="utf-8")
    readme = (ROOT / "readme.md").read_text(encoding="utf-8")

    assert not (ROOT / "csrc").exists()
    assert not (ROOT / "scripts" / "build_xpu.py").exists()
    assert not (ROOT / "tests" / "bench_xpu_h3_contract.py").exists()
    assert "from omni_xpu_kernel import cute as omni_cute" in backend
    assert "_OMNI_CUTE.sol_attn(" in backend
    assert "torch.ops.omni_xpu_sol_attn.prepare(" in backend
    assert "torch.ops.omni_xpu_sol_attn.forward_cute(" in backend
    assert "torch.ops.sol_attn_xpu.prepare(" not in backend
    assert "torch.ops.sol_attn_xpu.forward_cute(" not in backend
    assert "custom node intentionally contains no XPU C++ source" in readme


def test_packaged_backend_requires_gate_and_records_exact_dso(
    tmp_path, monkeypatch
):
    library = tmp_path / "cute_fmha_torch.so"
    library.write_bytes(b"packaged-sol-attn-test")
    fake_cute = types.SimpleNamespace(
        supports_sol_attn=lambda: True,
        _find_so=lambda: str(library),
    )
    fake_package = types.ModuleType("omni_xpu_kernel")
    fake_package.cute = fake_cute
    monkeypatch.setitem(sys.modules, "omni_xpu_kernel", fake_package)

    disabled = _load_backend("sol_attn_xpu_disabled_package_contract")
    monkeypatch.delenv("SOL_ATTN_XPU_EXPERIMENTAL", raising=False)
    assert not disabled.backend_available()
    assert "SOL_ATTN_XPU_EXPERIMENTAL=1" in str(disabled.backend_error())

    enabled = _load_backend("sol_attn_xpu_enabled_package_contract")
    monkeypatch.setenv("SOL_ATTN_XPU_EXPERIMENTAL", "1")
    assert enabled.backend_available()
    assert enabled._BACKEND == "omni-cute"
    assert enabled._OMNI_CUTE is fake_cute
    assert enabled._LOADED_LIBRARY == library.resolve()
    assert enabled._LOADED_LIBRARY_SHA256 == hashlib.sha256(
        library.read_bytes()
    ).hexdigest()


def test_normal_forward_uses_public_packaged_api():
    backend = _load_backend("sol_attn_xpu_public_api_contract")
    calls = []
    expected = torch.ones((1, 3, 2, 128), dtype=torch.bfloat16)

    def packaged_sol_attn(q, k, v, **kwargs):
        calls.append((q, k, v, kwargs))
        return expected

    backend._LOADED = True
    backend._BACKEND = "omni-cute"
    backend._OMNI_CUTE = types.SimpleNamespace(sol_attn=packaged_sol_attn)
    q = torch.zeros_like(expected)
    k = torch.zeros_like(expected)
    v = torch.zeros_like(expected)
    actual = backend.sol_attn(
        q,
        k,
        v,
        scale=0.125,
        tau=1.3,
        sink_blocks=(0, 2),
        sink_q=(1, 3),
        use_tma=True,
    )

    assert actual is expected
    assert calls == [
        (
            q,
            k,
            v,
            {
                "scale": 0.125,
                "tau": 1.3,
                "sink_blocks": (0, 2),
                "sink_q": (1, 3),
            },
        )
    ]


def test_real_correctness_prefers_pinned_reference_then_packaged_serial_parent():
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
    assert "torch.ops.omni_xpu_sol_attn" in backend
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

    monkeypatch.setenv("SOL_ATTN_XPU_REAL_REFERENCE_LIBRARY", str(reference))
    monkeypatch.delenv("SOL_ATTN_XPU_REAL_REFERENCE_SHA256", raising=False)
    with pytest.raises(RuntimeError, match="requires both"):
        backend._real_reference_spec()

    monkeypatch.setenv("SOL_ATTN_XPU_REAL_REFERENCE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        backend._real_reference_spec()

    monkeypatch.setenv("SOL_ATTN_XPU_REAL_REFERENCE_SHA256", digest)
    assert backend._real_reference_spec() == (reference.resolve(), digest)


def test_real_correctness_records_elementwise_policy(tmp_path, monkeypatch):
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
    monkeypatch.setenv("SOL_ATTN_XPU_REAL_CORRECTNESS", str(rejected_destination))
    rejected_backend = _load_backend(
        "sol_attn_xpu_real_tolerance_rejection_contract"
    )
    zero = torch.zeros((1, 1, 1), dtype=torch.bfloat16)
    outside = torch.full_like(zero, 0.0625)
    assert rejected_backend.record_real_correctness(outside, zero, q)
    rejected = json.loads(rejected_destination.read_text(encoding="utf-8"))
    assert rejected["elementwise_tolerance"]["allclose"] is False
    assert rejected["elementwise_tolerance"]["mismatched_elements"] == 1
    assert rejected["elementwise_tolerance"]["mismatch_fraction"] == 1.0


def test_real_activation_attribution_profiles_only_packaged_forward():
    backend = (ROOT / "_xpu_fwd.py").read_text(encoding="utf-8")
    runtime = (ROOT / "__init__.py").read_text(encoding="utf-8")

    assert '"SOL_ATTN_XPU_ATTRIBUTION"' in backend
    assert '"SOL_ATTN_XPU_ATTRIBUTION_LOG"' in backend
    assert '"SOL_ATTN_XPU_ATTRIBUTION_ITT_LIBRARY"' in backend
    assert "capture_index = _capture_prepared_routes(" in backend
    assert '    _itt_command(itt_library, "resume")' in backend
    assert "        output = torch.ops.omni_xpu_sol_attn.forward_cute(" in backend
    assert "        torch.xpu.synchronize()" in backend
    assert '        _itt_command(itt_library, "pause")' in backend
    assert "    _record_attribution_call(" in backend
    assert '"raw_activation_or_route_persisted": False' in backend
    assert "attribution_enabled as _xpu_attribution_enabled" in runtime
    assert 'extra["capture_context"] = capture_context' in runtime


def test_disjoint_descending_pair_aggregate_uses_second_tile_denominator(
    tmp_path,
):
    backend = _load_backend("sol_attn_xpu_route_pair_contract")
    route = torch.zeros((1, 1, 8, 8), dtype=torch.bool)
    route[:, :, :4, :2] = True
    route[:, :, 4:, 1:3] = True
    q = torch.zeros((1, 512, 1, 128), dtype=torch.bfloat16)
    destination = tmp_path / "route.jsonl"
    capture_index = backend._capture_route_record(
        route,
        q,
        scale=128**-0.5,
        tau=1.3,
        sink_blocks=(0, 0),
        sink_q=(0, 0),
        context={"block": 0, "sigma": 0.5},
        destination=destination,
    )

    aggregate = json.loads(destination.read_text(encoding="utf-8"))["global"]
    assert capture_index == 0
    assert aggregate["paired_q256_pairs_per_batch_head"] == 1
    assert aggregate["paired_q256_workgroups_per_batch_head"] == 1
    assert aggregate["paired_q256_total_workgroups"] == 1
    assert aggregate["paired_q256_second_tile_reused_key_tiles"] == 1
    assert aggregate["paired_q256_second_tile_selected_key_tiles"] == 2
    assert aggregate["paired_q256_second_tile_reuse_fraction"] == 0.5
