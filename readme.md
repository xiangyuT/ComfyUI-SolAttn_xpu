# ComfyUI-SolAttn XPU

> [!IMPORTANT]
> **Experimental Intel XPU fork — limited validation scope**
>
> The XPU changes in this fork are developed and performance-tuned for work
> related to [Intel llm-scaler](https://github.com/intel/llm-scaler), primarily
> on Ubuntu 24.04 LTS with Intel Arc Pro B70.
>
> Other operating systems, GPU models, software environments, workloads, and
> configurations are outside the validated scope. Their compatibility,
> correctness, output quality, stability, and performance are not guaranteed.
>
> If you have a specific requirement outside the current scope, you may open a
> relevant issue with the complete environment, expected use case, and a
> reproducible example. Opening an issue provides a way to document and discuss
> the request. Whether it can be explored or supported will depend on its
> relevance, reproducibility, and available capacity. The validated scope will
> be updated only after any additional support has been implemented and
> validated.
>
> This work is experimental. Ongoing support, maintenance, compatibility
> updates, and future development depend on available time, capacity, and
> project priorities.

## Intel XPU integration

This fork adds an experimental Intel XPU dispatch path to the upstream
ComfyUI-SolAttn node. The native XPU implementation is provided by the
`omni_xpu_kernel` package from
[Intel llm-scaler](https://github.com/intel/llm-scaler) and uses a
SYCL-TLA/CUTE/DPAS backend. The XPU path does not import or require Triton.

Kernel ownership, compilation, and packaging remain in `omni_xpu_kernel`; this
custom node intentionally contains no XPU C++ source or local build script. It
only provides ComfyUI nodes, dispatch, configuration, and a guarded dense
fallback. The upstream CUDA Triton implementation remains available for CUDA
environments.

### Installation

The maintained XPU integration is included by the focused Omni image build in
[Intel llm-scaler](https://github.com/intel/llm-scaler/tree/main/omni). For a
manual ComfyUI installation, clone this fork into `custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xiangyuT/ComfyUI-SolAttn_xpu.git ComfyUI-SolAttn
```

Installing this custom node alone does not enable XPU execution. The ComfyUI
environment must also contain a compatible Intel XPU build of
[`omni_xpu_kernel`](https://github.com/intel/llm-scaler/tree/main/omni/omni_xpu_kernel)
for the installed PyTorch ABI and GPU target. Verify that the package exports
the Sol-Attn capability before starting ComfyUI:

```python
from omni_xpu_kernel import cute

assert cute.supports_sol_attn()
```

The packaged Sol-Attn backend currently targets Intel Arc
B-series/Battlemage. Its operator contract is limited to matching BF16 Q/K/V
tensors on XPU, self-attention without a mask, head dimension 128, and a
contiguous final dimension. It accepts the interleaved QKV views used by
supported ComfyUI model paths without requiring them to be materialized.
Cross-attention and other unsupported tensor contracts retain the existing
dense attention path.

INT8 QK/PV and TMA remain CUDA-only. XPU execution uses the packaged BF16
backend even when a CUDA-oriented option is present in the workflow. Set
`int8_qk=false` and leave `use_tma=false` when configuring an XPU workflow so
the requested behavior is explicit.

### Enable the experimental backend

Start ComfyUI with the XPU gate enabled after installing the matching package:

```bash
SOL_ATTN_XPU_EXPERIMENTAL=1 python main.py
```

Place **Patch Sol-Attn** after the model loader and connect the patched model to
the sampler. The patch is per model and remains opt-in; workflows that do not
apply it keep their existing attention implementation. `start_percent`,
`end_percent`, `tau`, and `min_tokens` control when sparse attention is
requested. Begin from the model's maintained workflow and evaluate output
quality for the intended prompts and settings before relying on the sparse
route.

`SOL_ATTN=1` enables the global attention override for command-line diagnostics,
but it does not replace `SOL_ATTN_XPU_EXPERIMENTAL=1`. Normal ComfyUI workflows
should use the per-model **Patch Sol-Attn** node.

Sol-Attn changes the attention computation and can change generated output.
Backend availability and a successful run do not establish compatibility,
quality, stability, or a performance benefit for an unvalidated environment or
workflow.

---

## Upstream documentation

> The following is the original README from
> [kijai/ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton)
> at revision
> [`dfc2e31a41afd72bd53dd2137fc8b2931d5ec192`](https://github.com/kijai/ComfyUI-SolAttn_triton/commit/dfc2e31a41afd72bd53dd2137fc8b2931d5ec192),
> retained unchanged for reference.

<h1 align="center">ComfyUI-SolAttn</h1>

<h4 align="center">
  Experimental Triton implementation of Sol-Attn for ComfyUI
</h4>

<p align="center">
  <a href="https://arxiv.org/abs/2607.24027"><img src="https://img.shields.io/badge/📄_Paper-arXiv-b31b1b?style=flat-square" alt="Paper"/></a>
  <a href="https://github.com/NVlabs/Sana/tree/sol-engine/techniques/sparse_backends/sol_attn"><img src="https://img.shields.io/badge/💻_Code-Sol--Attn-76b900?style=flat-square" alt="Code"/></a>
  <a href="https://nvlabs.github.io/Sana/Sol-Attn/"><img src="https://img.shields.io/badge/🌐_Project-Page-blue?style=flat-square" alt="Project Page"/></a>
</p>

---

## Overview

[Sol-Attn](https://arxiv.org/abs/2607.24027) is a training-free sparse attention
method for accelerating image and video generation. This community extension
integrates a Triton implementation of Sol-Attn into ComfyUI.

> [!NOTE]
> This project is a work in progress. It has currently been tested on RTX 4090
> and RTX 5090 GPUs with MiniMax H3.

## Usage notes

Triton kernels are compiled on first use, so the first run will be slower.

Use `start_percent`, `end_percent`, and `tau` to balance generation quality and
speed.

## Examples

### Test output

https://github.com/user-attachments/assets/8d9ed820-0417-4d68-9d1c-5199534bed3b

### SageAttention vs. Sol-Attn

<table>
<tr>
<td align="center"><b>SageAttention</b></td>
<td align="center"><b>Sol-Attn</b></td>
</tr>
<tr>
<td width="50%">
<video src="https://github.com/user-attachments/assets/27f201ea-6bfc-4f43-826c-51809eed9d15" controls muted loop></video>
</td>
<td width="50%">
<video src="https://github.com/user-attachments/assets/73f63d14-2166-4f62-b098-e817ec1d7704" controls muted loop></video>
</td>
</tr>
</table>

<img width="482" height="500" alt="Sol-Attn example result" src="https://github.com/user-attachments/assets/27ae9886-aa3e-4470-a507-3a7c52b24be5" />

## Citation

If you find Sol-Attn useful in your work, please cite the paper:

```bibtex
@article{li2026solattn,
  title={Sol-Attn: Accelerating Video Generation Inference via On-the-Fly Attention Sparsification},
  author={Li, Haopeng and Li, Yitong and Chen, Junsong and Ye, Tian and Liu, Haozhe and Yu, Jincheng and Wang, Duomin and Zhang, Ruihua and Xie, Zeke and Xie, Enze and Han, Song},
  journal={arXiv preprint arXiv:2607.24027},
  year={2026}
}
```
