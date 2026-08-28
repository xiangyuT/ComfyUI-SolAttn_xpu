// Copyright 2026
//
// Apache-2.0 licensed correctness-first Intel XPU backend for Sol-Attn.
// The kernels run on PyTorch's current XPU stream and preserve BTHD strides.

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/library.h>

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <limits>

namespace sol_attn_xpu {
namespace {

using bf16 = sycl::ext::oneapi::bfloat16;

constexpr int64_t kBlock = 64;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kWorkGroup = 128;
constexpr float kLog2E = 1.4426950408889634f;

template <typename KernelFunc>
void submit(
    sycl::queue& queue,
    int64_t groups,
    KernelFunc&& kernel) {
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<size_t>(groups * kWorkGroup)),
            sycl::range<1>(static_cast<size_t>(kWorkGroup))),
        std::forward<KernelFunc>(kernel));
  });
}

struct SummaryKernel;
struct KStatsKernel;
struct ThresholdKernel;
struct RouteKernel;
struct ForwardKernel;

void check_input(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_xpu(), name, " must be on XPU");
  TORCH_CHECK(
      tensor.scalar_type() == at::kBFloat16,
      name,
      " must be bfloat16");
  TORCH_CHECK(tensor.dim() == 4, name, " must be BTHD");
  TORCH_CHECK(tensor.size(1) > 0, name, " must have a non-empty sequence");
  TORCH_CHECK(
      tensor.size(3) == kHeadDim,
      name,
      " must have head_dim 128");
  TORCH_CHECK(
      tensor.stride(3) == 1,
      name,
      " must have a contiguous head dimension");
}

at::Tensor forward(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    double scale_value,
    double tau_value,
    int64_t sink_start,
    int64_t sink_end,
    int64_t sink_q_start,
    int64_t sink_q_end) {
  check_input(q, "q");
  check_input(k, "k");
  check_input(v, "v");
  TORCH_CHECK(q.sizes() == k.sizes(), "q and k must share BTHD shape");
  TORCH_CHECK(q.sizes() == v.sizes(), "q and v must share BTHD shape");
  TORCH_CHECK(q.device() == k.device(), "q and k must share an XPU device");
  TORCH_CHECK(q.device() == v.device(), "q and v must share an XPU device");
  TORCH_CHECK(std::isfinite(scale_value), "scale must be finite");
  TORCH_CHECK(std::isfinite(tau_value), "tau must be finite");

  const int64_t batch = q.size(0);
  const int64_t tokens = q.size(1);
  const int64_t heads = q.size(2);
  const int64_t blocks = (tokens + kBlock - 1) / kBlock;
  TORCH_CHECK(
      0 <= sink_start && sink_start <= sink_end && sink_end <= blocks,
      "sink_blocks must be an ordered range inside the block count");
  TORCH_CHECK(
      0 <= sink_q_start && sink_q_start <= sink_q_end && sink_q_end <= blocks,
      "sink_q must be an ordered range inside the block count");

  const float scale = static_cast<float>(scale_value);
  const float tau = static_cast<float>(tau_value);
  const float scale_log2 = scale * kLog2E;
  const int64_t summary_groups = batch * heads * blocks;

  auto float_options = q.options().dtype(at::kFloat);
  auto byte_options = q.options().dtype(at::kByte);
  auto q_centroids = at::empty(
      {batch, heads, blocks, kHeadDim}, float_options);
  auto k_centroids = at::empty(
      {batch, heads, blocks, kHeadDim}, q.options());
  auto v_sums = at::empty_like(k_centroids);
  auto k_mean = at::empty({batch, heads, kHeadDim}, float_options);
  auto k_variance = at::empty_like(k_mean);
  auto thresholds = at::empty({batch, heads, blocks}, float_options);
  auto routes = at::empty(
      {batch, heads, blocks, blocks}, byte_options);
  auto output = at::empty({batch, tokens, heads, kHeadDim}, q.options());

  const auto* q_ptr = reinterpret_cast<const bf16*>(q.data_ptr());
  const auto* k_ptr = reinterpret_cast<const bf16*>(k.data_ptr());
  const auto* v_ptr = reinterpret_cast<const bf16*>(v.data_ptr());
  auto* qc_ptr = q_centroids.data_ptr<float>();
  auto* kc_ptr = reinterpret_cast<bf16*>(k_centroids.data_ptr());
  auto* vc_ptr = reinterpret_cast<bf16*>(v_sums.data_ptr());
  auto* mean_ptr = k_mean.data_ptr<float>();
  auto* variance_ptr = k_variance.data_ptr<float>();
  auto* threshold_ptr = thresholds.data_ptr<float>();
  auto* route_ptr = routes.data_ptr<uint8_t>();
  auto* output_ptr = reinterpret_cast<bf16*>(output.data_ptr());

  const int64_t sq_b = q.stride(0);
  const int64_t sq_t = q.stride(1);
  const int64_t sq_h = q.stride(2);
  const int64_t sk_b = k.stride(0);
  const int64_t sk_t = k.stride(1);
  const int64_t sk_h = k.stride(2);
  const int64_t sv_b = v.stride(0);
  const int64_t sv_t = v.stride(1);
  const int64_t sv_h = v.stride(2);

  auto& queue = c10::xpu::getCurrentXPUStream(q.device().index()).queue();

  submit(queue, summary_groups, [=](sycl::nd_item<1> item) {
    const int64_t group = item.get_group_linear_id();
    const int64_t dim = item.get_local_linear_id();
    const int64_t block = group % blocks;
    const int64_t batch_head = group / blocks;
    const int64_t head = batch_head % heads;
    const int64_t batch_index = batch_head / heads;
    const int64_t start = block * kBlock;
    const int64_t length = sycl::min(kBlock, tokens - start);

    float q_sum = 0.0f;
    float k_sum = 0.0f;
    float v_sum = 0.0f;
    for (int64_t offset = 0; offset < length; ++offset) {
      const int64_t token = start + offset;
      q_sum += static_cast<float>(
          q_ptr[batch_index * sq_b + token * sq_t + head * sq_h + dim]);
      k_sum += static_cast<float>(
          k_ptr[batch_index * sk_b + token * sk_t + head * sk_h + dim]);
      v_sum += static_cast<float>(
          v_ptr[batch_index * sv_b + token * sv_t + head * sv_h + dim]);
    }
    const int64_t summary = (batch_head * blocks + block) * kHeadDim + dim;
    qc_ptr[summary] = q_sum / static_cast<float>(length);
    kc_ptr[summary] = static_cast<bf16>(k_sum / static_cast<float>(length));
    vc_ptr[summary] = static_cast<bf16>(v_sum);
  });

  submit(queue, batch * heads, [=](sycl::nd_item<1> item) {
    const int64_t batch_head = item.get_group_linear_id();
    const int64_t dim = item.get_local_linear_id();
    float total = 0.0f;
    float total_square = 0.0f;
    for (int64_t block = 0; block < blocks; ++block) {
      const float value = static_cast<float>(
          kc_ptr[(batch_head * blocks + block) * kHeadDim + dim]);
      total += value;
      total_square += value * value;
    }
    const float mean = total / static_cast<float>(blocks);
    mean_ptr[batch_head * kHeadDim + dim] = mean;
    variance_ptr[batch_head * kHeadDim + dim] =
        sycl::fmax(total_square / static_cast<float>(blocks) - mean * mean, 0.0f);
  });

  submit(queue, summary_groups, [=](sycl::nd_item<1> item) {
    const int64_t group = item.get_group_linear_id();
    const int64_t dim = item.get_local_linear_id();
    const int64_t block = group % blocks;
    const int64_t batch_head = group / blocks;
    const int64_t summary = (batch_head * blocks + block) * kHeadDim + dim;
    const float centroid = qc_ptr[summary];
    const float mean_part =
        centroid * mean_ptr[batch_head * kHeadDim + dim];
    const float variance_part = centroid * centroid *
        variance_ptr[batch_head * kHeadDim + dim];
    const float raw_mean = sycl::reduce_over_group(
        item.get_group(), mean_part, sycl::plus<float>());
    const float raw_variance = sycl::reduce_over_group(
        item.get_group(), variance_part, sycl::plus<float>());
    if (dim == 0) {
      const float mean = raw_mean * scale_log2;
      const float variance = raw_variance * scale_log2 * scale_log2;
      threshold_ptr[batch_head * blocks + block] =
          mean + tau * sycl::sqrt(sycl::fmax(variance, 0.0f) + 1.0e-6f);
    }
  });

  submit(queue, summary_groups, [=](sycl::nd_item<1> item) {
    const int64_t group = item.get_group_linear_id();
    const int64_t dim = item.get_local_linear_id();
    const int64_t query_block = group % blocks;
    const int64_t batch_head = group / blocks;
    const float q_value = qc_ptr[
        (batch_head * blocks + query_block) * kHeadDim + dim];
    const bool query_sink =
        query_block >= sink_q_start && query_block < sink_q_end;
    const float threshold = threshold_ptr[batch_head * blocks + query_block];

    for (int64_t key_block = 0; key_block < blocks; ++key_block) {
      const float partial = q_value * static_cast<float>(
          kc_ptr[(batch_head * blocks + key_block) * kHeadDim + dim]);
      const float route_score = sycl::reduce_over_group(
          item.get_group(), partial, sycl::plus<float>()) * scale_log2;
      if (dim == 0) {
        const int64_t distance = query_block > key_block
            ? query_block - key_block
            : key_block - query_block;
        const bool key_sink = key_block >= sink_start && key_block < sink_end;
        const bool exact = query_sink || key_sink || distance <= 1 ||
            route_score > threshold;
        route_ptr[(batch_head * blocks + query_block) * blocks + key_block] =
            static_cast<uint8_t>(exact);
      }
    }
  });

  submit(queue, batch * tokens * heads, [=](sycl::nd_item<1> item) {
    const int64_t group = item.get_group_linear_id();
    const int64_t dim = item.get_local_linear_id();
    const int64_t head = group % heads;
    const int64_t batch_token = group / heads;
    const int64_t token = batch_token % tokens;
    const int64_t batch_index = batch_token / tokens;
    const int64_t batch_head = batch_index * heads + head;
    const int64_t query_block = token / kBlock;
    const float q_value = static_cast<float>(
        q_ptr[batch_index * sq_b + token * sq_t + head * sq_h + dim]);

    float output_value = 0.0f;
    float denominator = 0.0f;
    float row_max = -std::numeric_limits<float>::infinity();

    for (int64_t key_block = 0; key_block < blocks; ++key_block) {
      const bool exact = route_ptr[
          (batch_head * blocks + query_block) * blocks + key_block] != 0;
      const int64_t key_start = key_block * kBlock;
      const int64_t key_length = sycl::min(kBlock, tokens - key_start);

      if (!exact) {
        const float partial = q_value * static_cast<float>(
            kc_ptr[(batch_head * blocks + key_block) * kHeadDim + dim]);
        const float score = sycl::reduce_over_group(
            item.get_group(), partial, sycl::plus<float>()) * scale_log2;
        const float new_max = sycl::fmax(row_max, score);
        const float alpha = sycl::exp2(row_max - new_max);
        const float probability = sycl::exp2(score - new_max);
        output_value = output_value * alpha + probability * static_cast<float>(
            vc_ptr[(batch_head * blocks + key_block) * kHeadDim + dim]);
        denominator = denominator * alpha +
            probability * static_cast<float>(key_length);
        row_max = new_max;
        continue;
      }

      for (int64_t key_offset = 0; key_offset < key_length; ++key_offset) {
        const int64_t key_token = key_start + key_offset;
        const float partial = q_value * static_cast<float>(
            k_ptr[batch_index * sk_b + key_token * sk_t + head * sk_h + dim]);
        const float score = sycl::reduce_over_group(
            item.get_group(), partial, sycl::plus<float>()) * scale_log2;
        const float new_max = sycl::fmax(row_max, score);
        const float alpha = sycl::exp2(row_max - new_max);
        const float probability = sycl::exp2(score - new_max);
        output_value = output_value * alpha + probability * static_cast<float>(
            v_ptr[batch_index * sv_b + key_token * sv_t + head * sv_h + dim]);
        denominator = denominator * alpha + probability;
        row_max = new_max;
      }
    }

    const int64_t output_offset =
        ((batch_index * tokens + token) * heads + head) * kHeadDim + dim;
    output_ptr[output_offset] = static_cast<bf16>(output_value / denominator);
  });

  return output;
}

}  // namespace
}  // namespace sol_attn_xpu

TORCH_LIBRARY(sol_attn_xpu, m) {
  m.def(
      "forward(Tensor q, Tensor k, Tensor v, float scale, float tau, "
      "int sink_start, int sink_end, int sink_q_start, int sink_q_end) -> Tensor");
}

TORCH_LIBRARY_IMPL(sol_attn_xpu, XPU, m) {
  m.impl("forward", &sol_attn_xpu::forward);
}
