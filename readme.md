<h1 align="center">ComfyUI-SolAttn</h1>

<h4 align="center">
  Experimental CUDA Triton and Intel XPU SYCL implementation of Sol-Attn for ComfyUI
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

## Intel XPU

The XPU backend does not import or depend on Triton. It is built as an AOT SYCL
sidecar and submitted on the current PyTorch XPU stream. CUDA Triton sources are
kept unchanged for upstream maintenance.

Current XPU scope is intentionally narrow: BF16, self-attention without a mask,
head dimension 128, and Q/K/V with a contiguous last dimension. This includes
MiniMax H3's interleaved QKV views without materializing them. INT8 QK/PV and
TMA are CUDA-only; on XPU the node logs once and uses the BF16 SYCL path.

Build the sidecar with the same Python environment that runs ComfyUI and a
SYCL-TLA checkout:

```bash
cd ComfyUI/custom_nodes/ComfyUI-SolAttn_xpu
python scripts/build_xpu.py --target bmg --sycl-tla-root /path/to/sycl-tla
```

The build requires an XPU PyTorch wheel, Intel `icpx`, and the Intel SYCL-TLA
sources used by the CUTE/DPAS mainloop. `CUTLASS_SYCL_ROOT` can be used instead
of `--sycl-tla-root`. Use `--target ptl-h` for a PTL-H build; BMG is the initial
development and validation target.

The CUTE/DPAS backend is still behind an explicit experimental gate while its
performance and workflow-quality acceptance work is incomplete. Restart
ComfyUI with the gate enabled after building:

```bash
SOL_ATTN_XPU_EXPERIMENTAL=1 python main.py
```

To enable the node in a workflow, place **Patch Sol-Attn** after the model loader
and connect its model output to the sampler. For the first XPU tests, set
`int8_qk=false`, leave `use_tma=false`, keep the canonical H3 workflow unchanged,
and start with `start_percent=0.2`, `end_percent=0.9`, `min_tokens=4096`.

`SOL_ATTN=1` still enables the global attention override for command-line
diagnostics, but it does not replace the XPU experimental gate. The per-model
patch node is preferred.

The repository also has a slow regular-SYCL correctness reference, built with
`--backend reference` and selected with `SOL_ATTN_XPU_REFERENCE=1`. It is for
diagnostics only. The CUTE/DPAS implementation is not yet a claimed performance
replacement for the maintained dense XPU attention route; canonical workflow
E2E and package/image validation remain acceptance gates.

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
