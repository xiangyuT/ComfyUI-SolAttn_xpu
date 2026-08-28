#!/usr/bin/env python3
"""Build a Triton-free Intel XPU Sol-Attn sidecar with Intel icpx."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=("bmg", "ptl-h"),
        default=os.environ.get(
            "SOL_ATTN_XPU_TARGET", os.environ.get("OMNI_XPU_DEVICE", "bmg")
        ),
        help="AOT Intel GPU target (default: environment, then bmg)",
    )
    parser.add_argument(
        "--backend",
        choices=("cute", "reference"),
        default="cute",
        help="CUTE/DPAS candidate or slow correctness reference",
    )
    parser.add_argument(
        "--sycl-tla-root",
        type=Path,
        default=Path(os.environ["CUTLASS_SYCL_ROOT"])
        if os.environ.get("CUTLASS_SYCL_ROOT")
        else None,
        help="SYCL-TLA source tree (required by --backend cute)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output shared library (default depends on --backend)",
    )
    parser.add_argument(
        "--q-tile",
        type=int,
        default=128,
        help="CUTE query tile for scheduler experiments (default: 128)",
    )
    parser.add_argument(
        "--subgroup-layout-q",
        type=int,
        default=16,
        help="CUTE Q subgroup count per workgroup (default: 16)",
    )
    parser.add_argument(
        "--paired-q256-scheduler",
        action="store_true",
        help=(
            "opt in to two descending Q256 tiles per workgroup; disabled by "
            "default and valid only with --q-tile 256"
        ),
    )
    parser.add_argument(
        "--grf-size",
        type=int,
        choices=(128, 256),
        default=256,
        help=(
            "requested Intel GRF size for the CUTE kernel; 128 is an "
            "experimental latency-hiding candidate (default: 256)"
        ),
    )
    parser.add_argument(
        "--nested-exact",
        action="store_true",
        help=(
            "consume each summary chunk's routed exact blocks immediately "
            "instead of using separate summary/exact traversals"
        ),
    )
    parser.add_argument(
        "--inline-route",
        action="store_true",
        help=(
            "compute route decisions inside each summary chunk and omit the "
            "global route-matrix preprocessing kernel"
        ),
    )
    parser.add_argument(
        "--shared-inline-route",
        action="store_true",
        help=(
            "share each inline Q64 route mask through workgroup-local memory "
            "instead of recomputing it in every subgroup"
        ),
    )
    parser.add_argument(
        "--prefetch-routed-kv",
        action="store_true",
        help=(
            "prefetch one routed exact K/V tile ahead using the shared "
            "workgroup route-mask union"
        ),
    )
    parser.add_argument(
        "--routed-kv-prefetch-distance",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "routed exact K/V lookahead distance in tiles; 1 preserves "
            "the accepted schedule (default: 1)"
        ),
    )
    parser.add_argument(
        "--split-routed-kv-prefetch",
        action="store_true",
        help=(
            "keep routed K one exact tile ahead but defer each routed V "
            "prefetch until immediately before its QK/softmax work"
        ),
    )
    parser.add_argument(
        "--stagger-routed-k-prefetch",
        action="store_true",
        help=(
            "prefetch routed V one exact tile ahead before QK, then issue "
            "the matching K prefetch after the current QK work"
        ),
    )
    parser.add_argument(
        "--register-pipeline-exact-k",
        action="store_true",
        help=(
            "replace the exact-K cache-hint prefetch with a one-slice "
            "global-to-register pipeline while retaining routed V prefetch"
        ),
    )
    parser.add_argument(
        "--double-buffer-route-masks",
        action="store_true",
        help=(
            "ping-pong the shared Q64 route masks so routed prefetch needs "
            "one workgroup barrier per summary chunk instead of two"
        ),
    )
    parser.add_argument(
        "--parallel-shared-inline-route",
        action="store_true",
        help=(
            "split each Q64 route-mask chunk across all of its subgroups "
            "before the existing shared-mask workgroup barrier"
        ),
    )
    parser.add_argument(
        "--subgroup-reduce-shared-route-masks",
        action="store_true",
        help=(
            "load parallel shared route fragments across subgroup lanes and "
            "reduce them with bitwise OR instead of serial uniform loads"
        ),
    )
    parser.add_argument(
        "--cache-route-query-centroid",
        action="store_true",
        help=(
            "load each lane's invariant route-query centroid values once "
            "before the parallel route loop instead of once per key block"
        ),
    )
    parser.add_argument(
        "--shared-cache-route-query-centroid",
        action="store_true",
        help=(
            "stage all cross-query Q64 centroid rows once per workgroup in "
            "shared memory instead of loading them in every owner subgroup"
        ),
    )
    parser.add_argument(
        "--group-load-route-k-centroid",
        action="store_true",
        help=(
            "load each contiguous route K128 centroid with one striped "
            "subgroup block read instead of eight scalar SIMD16 loads"
        ),
    )
    parser.add_argument(
        "--hierarchical-shared-route-masks",
        action="store_true",
        help=(
            "reduce parallel route fragments to one shared mask per Q64 "
            "before all subgroups reconstruct their row and union masks"
        ),
    )
    parser.add_argument(
        "--cross-query-route-columns",
        action="store_true",
        help=(
            "reuse each K centroid across every Q64 row in the workgroup and "
            "exchange packed two-key route columns through shared memory"
        ),
    )
    parser.add_argument(
        "--bmg-cacheable-exact-kv-loads",
        action="store_true",
        help=(
            "use the BMG two-level .ca.ca LSC policy for routed exact K/V "
            "demand loads; summary/Q loads and the pinned SYCL-TLA tree "
            "remain unchanged"
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.backend != "cute" and (
        args.q_tile != 128
        or args.subgroup_layout_q != 16
        or args.grf_size != 256
        or args.nested_exact
        or args.inline_route
        or args.shared_inline_route
        or args.prefetch_routed_kv
        or args.routed_kv_prefetch_distance != 1
        or args.split_routed_kv_prefetch
        or args.stagger_routed_k_prefetch
        or args.register_pipeline_exact_k
        or args.double_buffer_route_masks
        or args.parallel_shared_inline_route
        or args.subgroup_reduce_shared_route_masks
        or args.cache_route_query_centroid
        or args.shared_cache_route_query_centroid
        or args.group_load_route_k_centroid
        or args.hierarchical_shared_route_masks
        or args.cross_query_route_columns
        or args.bmg_cacheable_exact_kv_loads
        or args.paired_q256_scheduler
    ):
        raise SystemExit("Q-tile options apply only to --backend cute")
    if args.shared_inline_route and not args.inline_route:
        raise SystemExit("--shared-inline-route requires --inline-route")
    if args.prefetch_routed_kv and not args.shared_inline_route:
        raise SystemExit(
            "--prefetch-routed-kv requires --inline-route and "
            "--shared-inline-route"
        )
    if args.routed_kv_prefetch_distance != 1 and not args.prefetch_routed_kv:
        raise SystemExit(
            "--routed-kv-prefetch-distance 2 requires --prefetch-routed-kv"
        )
    if args.routed_kv_prefetch_distance != 1 and (
        args.split_routed_kv_prefetch
        or args.stagger_routed_k_prefetch
        or args.register_pipeline_exact_k
    ):
        raise SystemExit(
            "--routed-kv-prefetch-distance 2 is mutually exclusive with "
            "split, staggered and register-pipelined schedules"
        )
    if args.split_routed_kv_prefetch and not args.prefetch_routed_kv:
        raise SystemExit(
            "--split-routed-kv-prefetch requires --prefetch-routed-kv"
        )
    if args.stagger_routed_k_prefetch and not args.prefetch_routed_kv:
        raise SystemExit(
            "--stagger-routed-k-prefetch requires --prefetch-routed-kv"
        )
    if args.split_routed_kv_prefetch and args.stagger_routed_k_prefetch:
        raise SystemExit(
            "--split-routed-kv-prefetch and --stagger-routed-k-prefetch "
            "are mutually exclusive"
        )
    if args.register_pipeline_exact_k and not args.prefetch_routed_kv:
        raise SystemExit(
            "--register-pipeline-exact-k requires --prefetch-routed-kv"
        )
    if args.register_pipeline_exact_k and (
        args.split_routed_kv_prefetch or args.stagger_routed_k_prefetch
    ):
        raise SystemExit(
            "--register-pipeline-exact-k is mutually exclusive with the "
            "split and staggered prefetch schedules"
        )
    if args.double_buffer_route_masks and not args.prefetch_routed_kv:
        raise SystemExit(
            "--double-buffer-route-masks requires --prefetch-routed-kv"
        )
    if args.parallel_shared_inline_route and not args.shared_inline_route:
        raise SystemExit(
            "--parallel-shared-inline-route requires --shared-inline-route"
        )
    if (
        args.subgroup_reduce_shared_route_masks
        and not args.parallel_shared_inline_route
    ):
        raise SystemExit(
            "--subgroup-reduce-shared-route-masks requires "
            "--parallel-shared-inline-route"
        )
    if args.cache_route_query_centroid and not args.parallel_shared_inline_route:
        raise SystemExit(
            "--cache-route-query-centroid requires "
            "--parallel-shared-inline-route"
        )
    if (
        args.shared_cache_route_query_centroid
        and not args.cross_query_route_columns
    ):
        raise SystemExit(
            "--shared-cache-route-query-centroid requires "
            "--cross-query-route-columns"
        )
    if (
        args.shared_cache_route_query_centroid
        and args.cache_route_query_centroid
    ):
        raise SystemExit(
            "--shared-cache-route-query-centroid and "
            "--cache-route-query-centroid are mutually exclusive"
        )
    if (
        args.group_load_route_k_centroid
        and not args.parallel_shared_inline_route
    ):
        raise SystemExit(
            "--group-load-route-k-centroid requires "
            "--parallel-shared-inline-route"
        )
    if (
        args.hierarchical_shared_route_masks
        and not args.parallel_shared_inline_route
    ):
        raise SystemExit(
            "--hierarchical-shared-route-masks requires "
            "--parallel-shared-inline-route"
        )
    if args.cross_query_route_columns and not args.parallel_shared_inline_route:
        raise SystemExit(
            "--cross-query-route-columns requires "
            "--parallel-shared-inline-route"
        )
    if args.cross_query_route_columns and (
        args.subgroup_reduce_shared_route_masks
        or args.hierarchical_shared_route_masks
    ):
        raise SystemExit(
            "--cross-query-route-columns is mutually exclusive with the "
            "subgroup-reduced and hierarchical route-mask refinements"
        )
    if args.bmg_cacheable_exact_kv_loads and args.target != "bmg":
        raise SystemExit(
            "--bmg-cacheable-exact-kv-loads is valid only with --target bmg"
        )
    if args.bmg_cacheable_exact_kv_loads and not args.prefetch_routed_kv:
        raise SystemExit(
            "--bmg-cacheable-exact-kv-loads requires --prefetch-routed-kv"
        )
    if args.paired_q256_scheduler and args.q_tile != 256:
        raise SystemExit("--paired-q256-scheduler requires --q-tile 256")
    if args.q_tile <= 0 or args.q_tile % 64 != 0:
        raise SystemExit("--q-tile must be a positive multiple of 64")
    query_blocks = args.q_tile // 64
    if (
        args.subgroup_layout_q <= 0
        or args.subgroup_layout_q % query_blocks != 0
    ):
        raise SystemExit(
            "--subgroup-layout-q must be positive and divide evenly across "
            "the Q64 blocks in --q-tile"
        )
    if args.output is None and (
        args.q_tile != 128
        or args.subgroup_layout_q != 16
        or args.grf_size != 256
        or args.nested_exact
        or args.inline_route
        or args.shared_inline_route
        or args.prefetch_routed_kv
        or args.routed_kv_prefetch_distance != 1
        or args.split_routed_kv_prefetch
        or args.stagger_routed_k_prefetch
        or args.register_pipeline_exact_k
        or args.double_buffer_route_masks
        or args.parallel_shared_inline_route
        or args.subgroup_reduce_shared_route_masks
        or args.cache_route_query_centroid
        or args.shared_cache_route_query_centroid
        or args.group_load_route_k_centroid
        or args.hierarchical_shared_route_masks
        or args.cross_query_route_columns
        or args.bmg_cacheable_exact_kv_loads
        or args.paired_q256_scheduler
    ):
        raise SystemExit(
            "non-default CUTE geometry requires --output so the baseline DSO "
            "is not overwritten"
        )
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch must be installed in the build environment") from exc

    if not hasattr(torch, "xpu"):
        raise SystemExit(f"torch {torch.__version__} has no XPU backend")
    icpx = shutil.which("icpx")
    if icpx is None:
        candidate = Path("/opt/intel/oneapi/compiler/latest/bin/icpx")
        if candidate.is_file():
            icpx = str(candidate)
    if icpx is None:
        raise SystemExit("Intel icpx was not found; install/source oneAPI first")

    torch_root = Path(torch.__file__).resolve().parent
    torch_include = torch_root / "include"
    torch_lib = torch_root / "lib"
    torch_xpu = torch_lib / "libtorch_xpu.so"
    if not torch_xpu.is_file():
        raise SystemExit(f"torch {torch.__version__} is not an XPU wheel: {torch_xpu} missing")

    if args.backend == "cute":
        if args.sycl_tla_root is None:
            raise SystemExit(
                "--backend cute requires --sycl-tla-root or CUTLASS_SYCL_ROOT"
            )
        sycl_tla_root = args.sycl_tla_root.resolve()
        required = (
            sycl_tla_root / "include",
            sycl_tla_root / "applications" / "flash_attention_v2",
        )
        if not all(path.is_dir() for path in required):
            raise SystemExit(
                f"invalid SYCL-TLA source tree {sycl_tla_root}; "
                "include/ and applications/flash_attention_v2 are required"
            )
        try:
            source_revision = subprocess.run(
                ["git", "-c", f"safe.directory={sycl_tla_root}", "rev-parse", "HEAD"],
                cwd=sycl_tla_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SystemExit(
                f"cannot resolve SYCL-TLA source revision at {sycl_tla_root}"
            ) from exc
        sources = [
            ROOT / "csrc" / "sol_attn_xpu_prepare.cpp",
            ROOT / "csrc" / "sol_attn_xpu_cute.cpp",
        ]
        default_output = ROOT / "_sol_attn_xpu_cute.so"
    else:
        sycl_tla_root = None
        source_revision = None
        sources = [ROOT / "csrc" / "sol_attn_xpu.cpp"]
        default_output = ROOT / "_sol_attn_xpu_reference.so"

    output = (args.output or default_output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    abi = int(bool(torch.compiled_with_cxx11_abi()))
    command = [
        icpx,
        "-fsycl",
        "-fsycl-targets=spir64_gen",
        "-Xsycl-target-backend",
        f"-device {args.target}",
        "-fno-sycl-instrument-device-code",
        "-O3",
        "-DNDEBUG",
        "-fPIC",
        "-shared",
        "-std=c++17",
        f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
        f"-I{torch_include}",
        f"-I{torch_include / 'torch' / 'csrc' / 'api' / 'include'}",
        f"-I{ROOT / 'csrc'}",
        f"-I{sysconfig.get_path('include')}",
        f"-L{torch_lib}",
        "-ltorch",
        "-ltorch_cpu",
        "-ltorch_xpu",
        "-lc10",
        "-lc10_xpu",
        "-o",
        str(temporary),
    ]
    if args.backend == "cute":
        arch_macro = (
            "OMNI_XPU_ARCH_PTL_H" if args.target == "ptl-h" else "OMNI_XPU_ARCH_BMG"
        )
        command[command.index("-O3"):command.index("-O3")] = [
            "-Xspirv-translator",
            "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,"
            "+SPV_INTEL_subgroup_matrix_multiply_accumulate",
            "-DCUTLASS_ENABLE_SYCL",
            "-DSYCL_INTEL_TARGET",
            f"-D{arch_macro}=1",
            f"-DSOL_ATTN_Q_TILE={args.q_tile}",
            f"-DSOL_ATTN_SUBGROUP_LAYOUT_Q={args.subgroup_layout_q}",
            f"-DSOL_ATTN_GRF_SIZE={args.grf_size}",
            f"-DSOL_ATTN_NESTED_EXACT={int(args.nested_exact)}",
            f"-DSOL_ATTN_INLINE_ROUTE={int(args.inline_route)}",
            f"-DSOL_ATTN_SHARED_INLINE_ROUTE={int(args.shared_inline_route)}",
            f"-DSOL_ATTN_PREFETCH_ROUTED_KV={int(args.prefetch_routed_kv)}",
            "-DSOL_ATTN_ROUTED_KV_PREFETCH_DISTANCE="
            f"{args.routed_kv_prefetch_distance}",
            "-DSOL_ATTN_SPLIT_ROUTED_KV_PREFETCH="
            f"{int(args.split_routed_kv_prefetch)}",
            "-DSOL_ATTN_STAGGER_ROUTED_K_PREFETCH="
            f"{int(args.stagger_routed_k_prefetch)}",
            "-DSOL_ATTN_REGISTER_PIPELINE_EXACT_K="
            f"{int(args.register_pipeline_exact_k)}",
            "-DSOL_ATTN_DOUBLE_BUFFER_ROUTE_MASKS="
            f"{int(args.double_buffer_route_masks)}",
            "-DSOL_ATTN_PARALLEL_SHARED_INLINE_ROUTE="
            f"{int(args.parallel_shared_inline_route)}",
            "-DSOL_ATTN_SUBGROUP_REDUCE_SHARED_ROUTE_MASKS="
            f"{int(args.subgroup_reduce_shared_route_masks)}",
            "-DSOL_ATTN_CACHE_ROUTE_QUERY_CENTROID="
            f"{int(args.cache_route_query_centroid)}",
            "-DSOL_ATTN_SHARED_CACHE_ROUTE_QUERY_CENTROID="
            f"{int(args.shared_cache_route_query_centroid)}",
            "-DSOL_ATTN_GROUP_LOAD_ROUTE_K_CENTROID="
            f"{int(args.group_load_route_k_centroid)}",
            "-DSOL_ATTN_HIERARCHICAL_SHARED_ROUTE_MASKS="
            f"{int(args.hierarchical_shared_route_masks)}",
            "-DSOL_ATTN_CROSS_QUERY_ROUTE_COLUMNS="
            f"{int(args.cross_query_route_columns)}",
            "-DSOL_ATTN_BMG_CACHEABLE_EXACT_KV_LOADS="
            f"{int(args.bmg_cacheable_exact_kv_loads)}",
            "-DSOL_ATTN_PAIRED_Q256_SCHEDULER="
            f"{int(args.paired_q256_scheduler)}",
            f"-I{sycl_tla_root / 'include'}",
            f"-I{sycl_tla_root / 'tools' / 'util' / 'include'}",
            f"-I{sycl_tla_root / 'examples' / 'common'}",
            f"-I{sycl_tla_root / 'applications'}",
            "-Wno-unknown-pragmas",
            "-Wno-unused-variable",
            "-Wno-unused-but-set-variable",
            "-Wno-unused-local-typedef",
            "-Wno-uninitialized",
            "-Wno-reorder-ctor",
            "-Wno-logical-op-parentheses",
            "-Wno-unused-function",
            "-Wno-deprecated-copy",
            "-Wno-deprecated-declarations",
            "-Wno-c++20-extensions",
        ]
    command.extend(str(source) for source in sources)
    if args.verbose:
        print(" ".join(command))
    subprocess.run(command, check=True)
    os.replace(temporary, output)
    provenance = (
        f", SYCL-TLA {source_revision}" if source_revision is not None else ""
    )
    print(
        f"built {output} ({args.backend}) for {args.target} "
        f"with torch {torch.__version__}{provenance}; "
        f"Q{args.q_tile}/SG{args.subgroup_layout_q}/GRF{args.grf_size}; "
        f"nested_exact={'on' if args.nested_exact else 'off'}; "
        f"inline_route={'on' if args.inline_route else 'off'}; "
        f"shared_inline_route={'on' if args.shared_inline_route else 'off'}; "
        f"prefetch_routed_kv={'on' if args.prefetch_routed_kv else 'off'}; "
        "routed_kv_prefetch_distance="
        f"{args.routed_kv_prefetch_distance}; "
        "split_routed_kv_prefetch="
        f"{'on' if args.split_routed_kv_prefetch else 'off'}; "
        "stagger_routed_k_prefetch="
        f"{'on' if args.stagger_routed_k_prefetch else 'off'}; "
        "register_pipeline_exact_k="
        f"{'on' if args.register_pipeline_exact_k else 'off'}; "
        "double_buffer_route_masks="
        f"{'on' if args.double_buffer_route_masks else 'off'}; "
        "parallel_shared_inline_route="
        f"{'on' if args.parallel_shared_inline_route else 'off'}; "
        "subgroup_reduce_shared_route_masks="
        f"{'on' if args.subgroup_reduce_shared_route_masks else 'off'}; "
        "cache_route_query_centroid="
        f"{'on' if args.cache_route_query_centroid else 'off'}; "
        "shared_cache_route_query_centroid="
        f"{'on' if args.shared_cache_route_query_centroid else 'off'}; "
        "group_load_route_k_centroid="
        f"{'on' if args.group_load_route_k_centroid else 'off'}; "
        "hierarchical_shared_route_masks="
        f"{'on' if args.hierarchical_shared_route_masks else 'off'}; "
        "cross_query_route_columns="
        f"{'on' if args.cross_query_route_columns else 'off'}; "
        "bmg_cacheable_exact_kv_loads="
        f"{'on' if args.bmg_cacheable_exact_kv_loads else 'off'}; "
        "paired_q256_scheduler="
        f"{'on' if args.paired_q256_scheduler else 'off'}"
    )


if __name__ == "__main__":
    main()
