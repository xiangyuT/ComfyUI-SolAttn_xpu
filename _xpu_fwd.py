"""Intel XPU Sol-Attn adapter for the packaged ``omni_xpu_kernel`` backend.

The custom node owns ComfyUI dispatch and diagnostics only.  The AOT
SYCL-TLA/CUTE implementation and its build lifecycle live in
``omni_xpu_kernel``; importing this module remains Triton-free.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import threading
from pathlib import Path

import torch


_LOAD_LOCK = threading.Lock()
_LOADED = False
_LOAD_ERROR = None
_BACKEND = None
_OMNI_CUTE = None
_LOADED_LIBRARY = None
_LOADED_LIBRARY_SHA256 = None
_ROUTE_CAPTURE_LOCK = threading.Lock()
_ROUTE_CAPTURE_INDEX = 0
_ATTRIBUTION_LOCK = threading.Lock()
_ATTRIBUTION_INDEX = 0
_ATTRIBUTION_ITT_LIBRARY = None
_REAL_CORRECTNESS_LOCK = threading.Lock()
_REAL_CORRECTNESS_INDEX = 0
_REAL_REFERENCE_LOCK = threading.Lock()
_REAL_REFERENCE_LIBRARY = None
_REAL_REFERENCE_LIBRARY_SHA256 = None
_REAL_CORRECTNESS_RTOL = 5.0e-2
_REAL_CORRECTNESS_ATOL = 5.0e-2


def _enabled(name):
    return os.environ.get(name, "0") not in ("0", "", "false", "False")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_library():
    global _LOADED, _LOAD_ERROR, _BACKEND, _OMNI_CUTE
    global _LOADED_LIBRARY, _LOADED_LIBRARY_SHA256
    if _LOADED:
        return
    with _LOAD_LOCK:
        if _LOADED:
            return
        try:
            if not _enabled("SOL_ATTN_XPU_EXPERIMENTAL"):
                raise RuntimeError(
                    "Sol-Attn XPU is experimental and disabled by default; "
                    "start ComfyUI with SOL_ATTN_XPU_EXPERIMENTAL=1"
                )
            from omni_xpu_kernel import cute as omni_cute

            if not omni_cute.supports_sol_attn():
                raise RuntimeError(
                    "the installed omni_xpu_kernel build does not expose "
                    "the BMG Sol-Attn capability"
                )
            library_raw = omni_cute._find_so()
            library = Path(library_raw).resolve() if library_raw else None
            if library is None or not library.is_file():
                raise RuntimeError(
                    "omni_xpu_kernel loaded Sol-Attn but its CUTE DSO could "
                    "not be identified"
                )
        except Exception as exc:
            _LOAD_ERROR = exc
            raise RuntimeError(
                f"failed to load packaged Sol-Attn XPU backend: {exc}"
            ) from exc
        _LOADED = True
        _BACKEND = "omni-cute"
        _OMNI_CUTE = omni_cute
        _LOADED_LIBRARY = library
        _LOADED_LIBRARY_SHA256 = _sha256(_LOADED_LIBRARY)
        _LOAD_ERROR = None


def backend_available():
    """Whether an enabled packaged AOT Sol-Attn backend is available."""
    if _LOADED:
        return True
    try:
        _load_library()
    except Exception:
        return False
    return True


def backend_error():
    return _LOAD_ERROR


def route_capture_enabled():
    """Whether exact real-activation route aggregation was explicitly requested."""
    return bool(os.environ.get("SOL_ATTN_XPU_ROUTE_CAPTURE")) and not attribution_enabled()


def attribution_enabled():
    """Whether matched real-route plus exact-forward ITT attribution is enabled."""
    return _enabled("SOL_ATTN_XPU_ATTRIBUTION")


def reset_route_capture():
    global _ROUTE_CAPTURE_INDEX
    with _ROUTE_CAPTURE_LOCK:
        _ROUTE_CAPTURE_INDEX = 0


def reset_attribution():
    global _ATTRIBUTION_INDEX
    reset_route_capture()
    with _ATTRIBUTION_LOCK:
        _ATTRIBUTION_INDEX = 0


def _attribution_paths():
    route_raw = os.environ.get("SOL_ATTN_XPU_ROUTE_CAPTURE")
    calls_raw = os.environ.get("SOL_ATTN_XPU_ATTRIBUTION_LOG")
    itt_raw = os.environ.get("SOL_ATTN_XPU_ATTRIBUTION_ITT_LIBRARY")
    missing = [
        name
        for name, value in (
            ("SOL_ATTN_XPU_ROUTE_CAPTURE", route_raw),
            ("SOL_ATTN_XPU_ATTRIBUTION_LOG", calls_raw),
            ("SOL_ATTN_XPU_ATTRIBUTION_ITT_LIBRARY", itt_raw),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Sol-Attn XPU attribution requires " + ", ".join(missing)
        )
    return Path(route_raw).expanduser(), Path(calls_raw).expanduser(), Path(itt_raw).expanduser()


def _attribution_itt_library():
    global _ATTRIBUTION_ITT_LIBRARY
    _, _, path = _attribution_paths()
    if not path.is_file():
        raise RuntimeError(f"attribution ITT library does not exist: {path}")
    if _ATTRIBUTION_ITT_LIBRARY is None:
        _ATTRIBUTION_ITT_LIBRARY = ctypes.CDLL(str(path))
        for name in ("resume", "pause"):
            function = getattr(_ATTRIBUTION_ITT_LIBRARY, f"__itt_{name}")
            function.argtypes = []
            function.restype = None
    return _ATTRIBUTION_ITT_LIBRARY


def _itt_command(library, command):
    if command not in ("resume", "pause"):
        raise ValueError(f"unsupported ITT attribution command: {command}")
    getattr(library, f"__itt_{command}")()


def real_correctness_configured():
    """Whether dense-trajectory real-activation checking was requested."""
    return bool(os.environ.get("SOL_ATTN_XPU_REAL_CORRECTNESS"))


def _real_correctness_limit():
    raw = os.environ.get("SOL_ATTN_XPU_REAL_CORRECTNESS_CALLS", "1")
    try:
        limit = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            "SOL_ATTN_XPU_REAL_CORRECTNESS_CALLS must be a positive integer"
        ) from exc
    if limit <= 0:
        raise RuntimeError(
            "SOL_ATTN_XPU_REAL_CORRECTNESS_CALLS must be a positive integer"
        )
    return limit


def real_correctness_pending():
    if not real_correctness_configured():
        return False
    with _REAL_CORRECTNESS_LOCK:
        return _REAL_CORRECTNESS_INDEX < _real_correctness_limit()


def reset_real_correctness():
    global _REAL_CORRECTNESS_INDEX
    with _REAL_CORRECTNESS_LOCK:
        _REAL_CORRECTNESS_INDEX = 0


def _real_reference_spec():
    library_raw = os.environ.get("SOL_ATTN_XPU_REAL_REFERENCE_LIBRARY")
    sha256_raw = os.environ.get("SOL_ATTN_XPU_REAL_REFERENCE_SHA256")
    if bool(library_raw) != bool(sha256_raw):
        raise RuntimeError(
            "independent real-activation correctness requires both "
            "SOL_ATTN_XPU_REAL_REFERENCE_LIBRARY and "
            "SOL_ATTN_XPU_REAL_REFERENCE_SHA256"
        )
    if not library_raw:
        return None
    library = Path(library_raw).expanduser()
    if not library.is_absolute():
        raise RuntimeError(
            "SOL_ATTN_XPU_REAL_REFERENCE_LIBRARY must be an absolute path"
        )
    library = library.resolve()
    if not library.is_file():
        raise RuntimeError(
            f"independent real-activation reference does not exist: {library}"
        )
    expected_sha256 = sha256_raw
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise RuntimeError(
            "SOL_ATTN_XPU_REAL_REFERENCE_SHA256 must be a lowercase SHA-256"
        )
    observed_sha256 = _sha256(library)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            "independent real-activation reference SHA-256 mismatch: "
            f"observed {observed_sha256}, expected {expected_sha256}"
        )
    return library, observed_sha256


def _load_real_reference():
    """Load one pinned independent SYCL reference alongside the candidate."""
    global _REAL_REFERENCE_LIBRARY, _REAL_REFERENCE_LIBRARY_SHA256
    spec = _real_reference_spec()
    if spec is None:
        raise RuntimeError(
            "independent real-activation reference was not configured"
        )
    library, library_sha256 = spec
    with _REAL_REFERENCE_LOCK:
        if _REAL_REFERENCE_LIBRARY is not None:
            if (
                _REAL_REFERENCE_LIBRARY != library
                or _REAL_REFERENCE_LIBRARY_SHA256 != library_sha256
            ):
                raise RuntimeError(
                    "cannot replace the loaded independent real-activation "
                    "reference in one process"
                )
            return
        try:
            torch.ops.load_library(str(library))
        except Exception as exc:
            raise RuntimeError(
                f"failed to load independent Sol-Attn reference {library}: {exc}"
            ) from exc
        if not hasattr(torch.ops.sol_attn_xpu, "forward"):
            raise RuntimeError(
                "independent Sol-Attn reference did not register "
                "sol_attn_xpu.forward"
            )
        _REAL_REFERENCE_LIBRARY = library
        _REAL_REFERENCE_LIBRARY_SHA256 = library_sha256


def record_real_correctness(
    sparse: torch.Tensor,
    reference: torch.Tensor,
    q: torch.Tensor,
    *,
    context: dict | None = None,
) -> bool:
    """Append compact real-input error metrics without retaining activations."""
    global _REAL_CORRECTNESS_INDEX
    if not real_correctness_configured():
        raise RuntimeError("real-activation correctness was not explicitly enabled")
    if sparse.shape != reference.shape:
        raise RuntimeError(
            "real-activation correctness output shape mismatch: "
            f"{tuple(sparse.shape)} vs {tuple(reference.shape)}"
        )
    if sparse.device != reference.device:
        raise RuntimeError(
            "real-activation correctness outputs are on different devices"
        )

    candidate = sparse.float()
    dense = reference.float()
    delta = candidate - dense
    dense_norm = torch.linalg.vector_norm(dense)
    close = torch.isclose(
        candidate,
        dense,
        rtol=_REAL_CORRECTNESS_RTOL,
        atol=_REAL_CORRECTNESS_ATOL,
    )
    mismatched_elements = int((~close).sum().item())
    total_elements = close.numel()
    finite = bool(
        torch.isfinite(candidate).all().item()
        and torch.isfinite(dense).all().item()
    )
    record_context = dict(context or {})
    comparison = str(
        record_context.get("comparison", "candidate_vs_reference")
    )
    record = {
        "schema_version": 1,
        "kind": "sol_attn_xpu_real_activation_correctness",
        "context": record_context,
        "contract": {
            "q_shape": list(q.shape),
            "q_stride": list(q.stride()),
            "q_dtype": str(q.dtype).removeprefix("torch."),
            "output_shape": list(sparse.shape),
            "output_dtype": str(sparse.dtype).removeprefix("torch."),
        },
        "metrics": {
            "finite": finite,
            "max_abs": float(delta.abs().max().item()),
            "mean_abs": float(delta.abs().mean().item()),
            "rmse": float(delta.square().mean().sqrt().item()),
            "relative_l2": (
                float(torch.linalg.vector_norm(delta).item() / dense_norm.item())
                if dense_norm.item() > 0
                else None
            ),
        },
        "elementwise_tolerance": {
            "rtol": _REAL_CORRECTNESS_RTOL,
            "atol": _REAL_CORRECTNESS_ATOL,
            "allclose": mismatched_elements == 0,
            "mismatched_elements": mismatched_elements,
            "total_elements": total_elements,
            "mismatch_fraction": (
                float(mismatched_elements / total_elements)
                if total_elements
                else None
            ),
        },
        "evidence_limit": (
            f"one {comparison} real workflow attention call on "
            "a clean dense trajectory; raw activations and outputs are not "
            "persisted and this is not a timing sample"
        ),
    }
    destination = Path(
        os.environ["SOL_ATTN_XPU_REAL_CORRECTNESS"]
    ).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _REAL_CORRECTNESS_LOCK:
        if _REAL_CORRECTNESS_INDEX >= _real_correctness_limit():
            return False
        record["correctness_index"] = _REAL_CORRECTNESS_INDEX
        _REAL_CORRECTNESS_INDEX += 1
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    return True


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _capture_route_record(
    routes: torch.Tensor,
    q: torch.Tensor,
    *,
    scale: float,
    tau: float,
    sink_blocks: tuple[int, int],
    sink_q: tuple[int, int],
    context: dict | None,
    destination: Path | None = None,
) -> int:
    """Reduce an exact route matrix to compact scheduler/reuse statistics.

    The raw activation and route matrix remain transient. Only aggregate counts
    cross to the host and are appended to the explicitly configured JSONL.
    This path is diagnostic and intentionally synchronizing; it is never a
    timing benchmark.
    """
    global _ROUTE_CAPTURE_INDEX
    if destination is None:
        destination = Path(
            os.environ["SOL_ATTN_XPU_ROUTE_CAPTURE"]
        ).expanduser()
    route = routes.to(device="cpu", dtype=torch.bool)
    batch, heads, blocks, key_blocks = route.shape
    if blocks != key_blocks:
        raise RuntimeError(f"expected a square route matrix, found {route.shape}")

    row_counts = route.sum(dim=-1)
    key_counts = route.sum(dim=-2)
    group_count = (blocks + 3) // 4
    padded_rows = group_count * 4
    if padded_rows != blocks:
        padding = torch.zeros(
            (batch, heads, padded_rows - blocks, blocks), dtype=torch.bool
        )
        grouped_source = torch.cat((route, padding), dim=-2)
    else:
        grouped_source = route
    grouped = grouped_source.reshape(batch, heads, group_count, 4, blocks)
    reuse_counts = grouped.sum(dim=-2)
    union = reuse_counts > 0
    group_uses = reuse_counts.sum(dim=-1)
    group_union_counts = union.sum(dim=-1)

    valid_rows = torch.arange(padded_rows).reshape(group_count, 4) < blocks
    grouped_row_counts = torch.zeros(
        (batch, heads, group_count, 4), dtype=row_counts.dtype
    )
    grouped_row_counts[..., valid_rows] = row_counts
    group_max = grouped_row_counts.max(dim=-1).values
    min_source = grouped_row_counts.clone()
    min_source[..., ~valid_rows] = blocks + 1
    group_min = min_source.min(dim=-1).values
    group_imbalance = group_max - group_min

    adjacent_q64_intersection = (
        route[:, :, :-1, :] & route[:, :, 1:, :]
    ).sum(dim=(-1, -2))
    adjacent_q64_union = (
        route[:, :, :-1, :] | route[:, :, 1:, :]
    ).sum(dim=(-1, -2))
    if group_count > 1:
        adjacent_q256_intersection = (
            union[:, :, :-1, :] & union[:, :, 1:, :]
        ).sum(dim=(-1, -2))
        adjacent_q256_union = (
            union[:, :, :-1, :] | union[:, :, 1:, :]
        ).sum(dim=(-1, -2))
    else:
        adjacent_q256_intersection = torch.zeros((batch, heads), dtype=torch.int64)
        adjacent_q256_union = torch.zeros_like(adjacent_q256_intersection)

    descending_groups = list(reversed(range(group_count)))
    paired_first = descending_groups[0::2]
    paired_second = descending_groups[1::2]
    paired_workgroups_per_batch_head = len(paired_first)
    if paired_second:
        paired_first = paired_first[:len(paired_second)]
        paired_q256_intersection = (
            union[:, :, paired_first, :] & union[:, :, paired_second, :]
        ).sum(dim=(-1, -2))
        paired_q256_second_selected = union[
            :, :, paired_second, :
        ].sum(dim=(-1, -2))
    else:
        paired_q256_intersection = torch.zeros(
            (batch, heads), dtype=torch.int64
        )
        paired_q256_second_selected = torch.zeros_like(
            paired_q256_intersection
        )

    exact_edges = int(row_counts.sum().item())
    union_tiles = int(group_union_counts.sum().item())
    selected_key_tiles = int((key_counts > 0).sum().item())
    q64_adjacency_intersection = int(adjacent_q64_intersection.sum().item())
    q64_adjacency_union = int(adjacent_q64_union.sum().item())
    q256_adjacency_intersection = int(adjacent_q256_intersection.sum().item())
    q256_adjacency_union = int(adjacent_q256_union.sum().item())
    paired_q256_reused = int(paired_q256_intersection.sum().item())
    paired_q256_second = int(paired_q256_second_selected.sum().item())

    def nested(values):
        return values.tolist()

    per_batch_head = {
        "row_count_mean": nested(row_counts.float().mean(dim=-1)),
        "row_count_min": nested(row_counts.min(dim=-1).values),
        "row_count_max": nested(row_counts.max(dim=-1).values),
        "key_use_mean": nested(key_counts.float().mean(dim=-1)),
        "key_use_max": nested(key_counts.max(dim=-1).values),
        "selected_key_tiles": nested((key_counts > 0).sum(dim=-1)),
        "q256_union_mean": nested(group_union_counts.float().mean(dim=-1)),
        "q64_uses_per_q256_union_tile": nested(
            group_uses.sum(dim=-1).float()
            / group_union_counts.sum(dim=-1).clamp_min(1)
        ),
        "q256_union_uses_per_selected_key_tile": nested(
            group_union_counts.sum(dim=-1).float()
            / (key_counts > 0).sum(dim=-1).clamp_min(1)
        ),
        "q256_row_imbalance_mean": nested(group_imbalance.float().mean(dim=-1)),
        "q256_row_imbalance_max": nested(group_imbalance.max(dim=-1).values),
        "adjacent_q64_jaccard_weighted": nested(
            adjacent_q64_intersection.float()
            / adjacent_q64_union.clamp_min(1)
        ),
        "adjacent_q256_union_jaccard_weighted": nested(
            adjacent_q256_intersection.float()
            / adjacent_q256_union.clamp_min(1)
        ),
        "paired_q256_second_tile_reuse_fraction": nested(
            paired_q256_intersection.float()
            / paired_q256_second_selected.clamp_min(1)
        ),
        "q256_reuse_histogram_1_to_4": [
            nested((reuse_counts == uses).sum(dim=(-1, -2)))
            for uses in range(1, 5)
        ],
    }
    record = {
        "schema_version": 1,
        "kind": "sol_attn_xpu_exact_route_capture",
        "context": dict(context or {}),
        "contract": {
            "shape_bthd": list(q.shape),
            "stride_bthd": list(q.stride()),
            "dtype": str(q.dtype).removeprefix("torch."),
            "scale": float(scale),
            "tau": float(tau),
            "sink_blocks": list(sink_blocks),
            "sink_q": list(sink_q),
        },
        "global": {
            "exact_route_edges": exact_edges,
            "possible_route_edges": batch * heads * blocks * blocks,
            "route_density": _ratio(
                exact_edges, batch * heads * blocks * blocks
            ),
            "q256_union_tiles": union_tiles,
            "selected_key_tiles": selected_key_tiles,
            "q64_exact_uses_per_q256_union_tile": _ratio(
                exact_edges, union_tiles
            ),
            "q256_union_uses_per_selected_key_tile": _ratio(
                union_tiles, selected_key_tiles
            ),
            "q256_reuse_histogram_1_to_4": {
                str(uses): int((reuse_counts == uses).sum().item())
                for uses in range(1, 5)
            },
            "q256_row_imbalance_mean": float(group_imbalance.float().mean().item()),
            "q256_row_imbalance_max": int(group_imbalance.max().item()),
            "adjacent_q64_jaccard_weighted": _ratio(
                q64_adjacency_intersection, q64_adjacency_union
            ),
            "adjacent_q64_intersection": q64_adjacency_intersection,
            "adjacent_q64_union": q64_adjacency_union,
            "adjacent_q256_union_jaccard_weighted": _ratio(
                q256_adjacency_intersection, q256_adjacency_union
            ),
            "adjacent_q256_union_intersection": q256_adjacency_intersection,
            "adjacent_q256_union_union": q256_adjacency_union,
            "paired_q256_pairs_per_batch_head": len(paired_second),
            "paired_q256_workgroups_per_batch_head": (
                paired_workgroups_per_batch_head
            ),
            "paired_q256_total_workgroups": (
                batch * heads * paired_workgroups_per_batch_head
            ),
            "paired_q256_second_tile_reused_key_tiles": paired_q256_reused,
            "paired_q256_second_tile_selected_key_tiles": paired_q256_second,
            "paired_q256_second_tile_reuse_fraction": _ratio(
                paired_q256_reused, paired_q256_second
            ),
        },
        "per_batch_head": per_batch_head,
        "evidence_limit": (
            "diagnostic aggregate only; raw activations and route matrices "
            "are not persisted and this record is not a timing sample"
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _ROUTE_CAPTURE_LOCK:
        capture_index = _ROUTE_CAPTURE_INDEX
        record["capture_index"] = capture_index
        _ROUTE_CAPTURE_INDEX += 1
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    return capture_index


def _capture_prepared_routes(
    prepared: tuple[torch.Tensor, ...],
    q: torch.Tensor,
    *,
    scale: float,
    tau: float,
    sink_blocks: tuple[int, int],
    sink_q: tuple[int, int],
    context: dict | None,
    destination: Path | None = None,
) -> int:
    if len(prepared) != 5:
        raise RuntimeError(
            "route capture requires a packaged backend with five prepare outputs"
        )
    k_centroids, _, q_centroids, thresholds, key_sinks = prepared
    routes = torch.ops.omni_xpu_sol_attn.materialize_routes(
        k_centroids, q_centroids, thresholds, key_sinks, float(scale)
    )
    return _capture_route_record(
        routes,
        q,
        scale=scale,
        tau=tau,
        sink_blocks=sink_blocks,
        sink_q=sink_q,
        context=context,
        destination=destination,
    )


def capture_routes(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    tau: float,
    sink_blocks: tuple[int, int],
    sink_q: tuple[int, int],
    context: dict | None = None,
) -> None:
    """Materialize and aggregate the exact inline route, without sparse forward."""
    if not route_capture_enabled():
        raise RuntimeError("route capture was not explicitly enabled")
    _load_library()
    if _BACKEND != "omni-cute":
        raise RuntimeError("route capture requires packaged CUTE Sol-Attn")
    prepared = torch.ops.omni_xpu_sol_attn.prepare(
        q,
        k,
        v,
        float(scale),
        float(tau),
        int(sink_blocks[0]),
        int(sink_blocks[1]),
        int(sink_q[0]),
        int(sink_q[1]),
    )
    _capture_prepared_routes(
        prepared,
        q,
        scale=scale,
        tau=tau,
        sink_blocks=sink_blocks,
        sink_q=sink_q,
        context=context,
    )


def _record_attribution_call(
    capture_index: int,
    q: torch.Tensor,
    output: torch.Tensor,
    *,
    context: dict | None,
) -> None:
    global _ATTRIBUTION_INDEX
    _, destination, _ = _attribution_paths()
    if _LOADED_LIBRARY is None or _LOADED_LIBRARY_SHA256 is None:
        raise RuntimeError("attribution cannot identify the packaged XPU DSO")
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "kind": "sol_attn_xpu_real_activation_profiled_call",
        "context": dict(context or {}),
        "route_capture_index": capture_index,
        "library": str(_LOADED_LIBRARY),
        "library_sha256": _LOADED_LIBRARY_SHA256,
        "contract": {
            "q_shape": list(q.shape),
            "q_stride": list(q.stride()),
            "q_dtype": str(q.dtype).removeprefix("torch."),
            "output_shape": list(output.shape),
            "output_dtype": str(output.dtype).removeprefix("torch."),
        },
        "collection_window": (
            "prepare, route materialization, compact reduction and host write "
            "before ITT resume; forward_cute plus XPU synchronize inside; "
            "ITT pause before compact call record"
        ),
        "raw_activation_or_route_persisted": False,
        "performance_claim": False,
    }
    with _ATTRIBUTION_LOCK:
        record["attribution_index"] = _ATTRIBUTION_INDEX
        _ATTRIBUTION_INDEX += 1
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def sol_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    tau: float = 1.0,
    sink_blocks: tuple[int, int] = (0, 0),
    sink_q: tuple[int, int] = (0, 0),
    use_tma: bool = False,
    capture_context: dict | None = None,
) -> torch.Tensor:
    """Run the BF16 XPU backend on BTHD tensors.

    ``use_tma`` is accepted for call-site compatibility with the CUDA backend
    and has no effect on XPU.
    """
    del use_tma
    _load_library()
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    args = (
        q,
        k,
        v,
        scale,
        float(tau),
        int(sink_blocks[0]),
        int(sink_blocks[1]),
        int(sink_q[0]),
        int(sink_q[1]),
    )
    if not attribution_enabled():
        return _OMNI_CUTE.sol_attn(
            q,
            k,
            v,
            scale=scale,
            tau=float(tau),
            sink_blocks=sink_blocks,
            sink_q=sink_q,
        )
    prepared = torch.ops.omni_xpu_sol_attn.prepare(*args)

    route_path, _, _ = _attribution_paths()
    capture_index = _capture_prepared_routes(
        prepared,
        q,
        scale=scale,
        tau=float(tau),
        sink_blocks=sink_blocks,
        sink_q=sink_q,
        context=capture_context,
        destination=route_path,
    )
    torch.xpu.synchronize()
    itt_library = _attribution_itt_library()
    _itt_command(itt_library, "resume")
    try:
        output = torch.ops.omni_xpu_sol_attn.forward_cute(
            q, k, v, *prepared, scale
        )
        torch.xpu.synchronize()
    finally:
        _itt_command(itt_library, "pause")
    _record_attribution_call(
        capture_index,
        q,
        output,
        context=capture_context,
    )
    return output


def _sol_attn_parent_op(
    op_name: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    tau: float = 1.0,
    sink_blocks: tuple[int, int] = (0, 0),
    sink_q: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    _load_library()
    ops = torch.ops.omni_xpu_sol_attn
    if _BACKEND != "omni-cute" or not hasattr(ops, op_name):
        raise RuntimeError(
            f"the packaged candidate DSO has no {op_name} correctness op"
        )
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    prepared = ops.prepare(
        q,
        k,
        v,
        scale,
        float(tau),
        int(sink_blocks[0]),
        int(sink_blocks[1]),
        int(sink_q[0]),
        int(sink_q[1]),
    )
    return getattr(ops, op_name)(q, k, v, *prepared, scale)


def sol_attn_parent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    tau: float = 1.0,
    sink_blocks: tuple[int, int] = (0, 0),
    sink_q: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """Run the matched non-cacheable parent embedded in a candidate DSO."""
    return _sol_attn_parent_op(
        "forward_cute_parent",
        q,
        k,
        v,
        scale=scale,
        tau=tau,
        sink_blocks=sink_blocks,
        sink_q=sink_q,
    )


def sol_attn_real_correctness_parent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    tau: float = 1.0,
    sink_blocks: tuple[int, int] = (0, 0),
    sink_q: tuple[int, int] = (0, 0),
) -> tuple[torch.Tensor, str]:
    """Select a pinned independent reference or the narrowest embedded parent."""
    _load_library()
    if _real_reference_spec() is not None:
        _load_real_reference()
        scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
        output = torch.ops.sol_attn_xpu.forward(
            q,
            k,
            v,
            scale,
            float(tau),
            int(sink_blocks[0]),
            int(sink_blocks[1]),
            int(sink_q[0]),
            int(sink_q[1]),
        )
        return output, "candidate_vs_independent_sycl_reference"
    if _BACKEND == "omni-cute" and hasattr(
        torch.ops.omni_xpu_sol_attn, "forward_cute_serial_route_parent"
    ):
        output = _sol_attn_parent_op(
            "forward_cute_serial_route_parent",
            q,
            k,
            v,
            scale=scale,
            tau=tau,
            sink_blocks=sink_blocks,
            sink_q=sink_q,
        )
        return output, "candidate_vs_serial_route_parent"
    return (
        sol_attn_parent(
            q,
            k,
            v,
            scale=scale,
            tau=tau,
            sink_blocks=sink_blocks,
            sink_q=sink_q,
        ),
        "candidate_vs_matched_parent",
    )


__all__ = [
    "attribution_enabled",
    "backend_available",
    "backend_error",
    "capture_routes",
    "real_correctness_configured",
    "real_correctness_pending",
    "record_real_correctness",
    "reset_real_correctness",
    "reset_attribution",
    "reset_route_capture",
    "route_capture_enabled",
    "sol_attn",
    "sol_attn_parent",
    "sol_attn_real_correctness_parent",
]
