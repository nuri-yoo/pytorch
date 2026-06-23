#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/LossCTC.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_ctc_loss_backward_native.h>
#include <ATen/ops/_ctc_loss_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/full_like.h>
#include <ATen/ops/tensor.h>
#endif

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/LossCTC_metallib.h>
#endif

namespace {

// Build the per-batch offset into the targets tensor. Targets come either as a
// 2D (batch x max_target_length) tensor or as a 1D concatenation of all targets;
// in both cases we precompute where each batch item starts, on the host, exactly
// like the CPU/CUDA implementations.
static Tensor make_tg_batch_offsets(const Tensor& targets,
                                    IntArrayRef target_lengths,
                                    int64_t batch_size,
                                    int64_t& tg_target_stride,
                                    int64_t& max_target_length) {
  auto tg_batch_offsets = at::empty({batch_size}, at::device(at::kCPU).dtype(at::kLong));
  auto tg_batch_offsets_data = tg_batch_offsets.mutable_data_ptr<int64_t>();
  max_target_length = 0;
  if (targets.dim() == 1) { // concatenated targets
    int64_t pos = 0;
    for (const auto i : c10::irange(batch_size)) {
      tg_batch_offsets_data[i] = pos;
      pos += target_lengths[i];
      max_target_length = std::max(max_target_length, target_lengths[i]);
    }
    tg_target_stride = targets.stride(0);
  } else { // batch x max_target_length
    int64_t tg_batch_stride = targets.stride(0);
    for (const auto i : c10::irange(batch_size)) {
      tg_batch_offsets_data[i] = i * tg_batch_stride;
      max_target_length = std::max(max_target_length, target_lengths[i]);
    }
    tg_target_stride = targets.stride(1);
  }
  return tg_batch_offsets;
}

static std::string ctc_kernel_suffix(const Tensor& log_probs, ScalarType target_scalar_type) {
  return mps::scalarToMetalTypeString(log_probs) + "_" + (target_scalar_type == kInt ? "int" : "long");
}

} // anonymous namespace

std::tuple<Tensor, Tensor> ctc_loss_mps(const Tensor& log_probs,
                                        const Tensor& targets,
                                        IntArrayRef input_lengths,
                                        IntArrayRef target_lengths,
                                        int64_t BLANK,
                                        bool zero_infinity) {
  (void)zero_infinity; // forward loss does not depend on zero_infinity
  TORCH_CHECK(log_probs.numel() > 0, "log_probs tensor must not be empty");
  TORCH_CHECK(log_probs.dim() == 3, "log_probs has to be a 3D tensor");
  // float and half only (Metal has no double).
  TORCH_CHECK(log_probs.scalar_type() == kFloat || log_probs.scalar_type() == kHalf,
              "ctc_loss: MPS only supports float32 or float16 log_probs, got ",
              log_probs.scalar_type());
  auto target_scalar_type = targets.scalar_type();
  TORCH_CHECK(target_scalar_type == kInt || target_scalar_type == kLong,
              "targets must be int32 or int64, got ",
              target_scalar_type);

  int64_t batch_size = log_probs.size(1);
  int64_t num_labels = log_probs.size(2);
  int64_t max_input_length = log_probs.size(0);
  TORCH_CHECK((0 <= BLANK) && (BLANK < num_labels), "blank must be in label range");
  TORCH_CHECK(input_lengths.size() == static_cast<size_t>(batch_size), "input_lengths must be of size batch_size");
  TORCH_CHECK(target_lengths.size() == static_cast<size_t>(batch_size), "target_lengths must be of size batch_size");

  int64_t tg_target_stride = 0;
  int64_t max_target_length = 0;
  auto tg_batch_offsets =
      make_tg_batch_offsets(targets, target_lengths, batch_size, tg_target_stride, max_target_length);

  for (const auto b : c10::irange(batch_size)) {
    TORCH_CHECK(input_lengths[b] >= 0, "input_lengths must be non-negative");
    TORCH_CHECK(input_lengths[b] <= max_input_length,
                "Expected input_lengths to have value at most ",
                max_input_length,
                ", but got ",
                input_lengths[b]);
    TORCH_CHECK(target_lengths[b] >= 0, "target_lengths must be non-negative");
  }

  auto input_lengths_t = at::tensor(input_lengths, targets.options().dtype(kLong)).to(log_probs.device());
  auto target_lengths_t = at::tensor(target_lengths, targets.options().dtype(kLong)).to(log_probs.device());
  tg_batch_offsets = tg_batch_offsets.to(log_probs.device());

  // DP table and loss accumulate in float (acc_t); see LossCTC.metal.
  auto acc_options = log_probs.options().dtype(kFloat);
  Tensor log_alpha = at::empty({batch_size, max_input_length, 2 * max_target_length + 1}, acc_options);
  Tensor neg_log_likelihood = at::empty({batch_size}, acc_options);

  CTCLossParams params{batch_size,
                       max_input_length,
                       max_target_length,
                       num_labels,
                       BLANK,
                       log_probs.stride(0),
                       log_probs.stride(1),
                       log_probs.stride(2),
                       log_alpha.stride(0),
                       log_alpha.stride(1),
                       log_alpha.stride(2),
                       tg_target_stride,
                       zero_infinity};

  // One threadgroup per batch item; threads cover the label axis (capped at the
  // device limit, the kernel strides over s if the axis is longer).
  const int64_t la_size = 2 * max_target_length + 1;
  using namespace mps;
  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = stream->commandEncoder();
      auto PSO = lib.getPipelineStateForFunc("ctc_loss_alpha_" + ctc_kernel_suffix(log_probs, target_scalar_type));
      uint64_t tg_threads = std::min<uint64_t>(la_size, PSO.maxTotalThreadsPerThreadgroup);
      getMPSProfiler().beginProfileKernel(PSO, "ctc_loss_alpha", {log_probs, targets});
      [computeEncoder setComputePipelineState:PSO];
      mtl_setArgs(computeEncoder,
                  log_alpha,
                  log_probs,
                  targets,
                  input_lengths_t,
                  target_lengths_t,
                  tg_batch_offsets,
                  neg_log_likelihood,
                  params);
      [computeEncoder dispatchThreadgroups:MTLSizeMake(batch_size, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(tg_threads, 1, 1)];
      getMPSProfiler().endProfileKernel(PSO);
    }
  });
  // Cast the loss back to the input dtype; log_alpha stays float for backward.
  return std::make_tuple(neg_log_likelihood.to(log_probs.scalar_type()), log_alpha);
}

Tensor ctc_loss_backward_mps(const Tensor& grad_out,
                             const Tensor& log_probs,
                             const Tensor& targets,
                             IntArrayRef input_lengths,
                             IntArrayRef target_lengths,
                             const Tensor& neg_log_likelihood,
                             const Tensor& log_alpha,
                             int64_t BLANK,
                             bool zero_infinity) {
  constexpr double neginf = -std::numeric_limits<double>::infinity();
  TORCH_CHECK(log_probs.scalar_type() == kFloat || log_probs.scalar_type() == kHalf,
              "ctc_loss_backward: MPS only supports float32 or float16 log_probs, got ",
              log_probs.scalar_type());
  auto target_scalar_type = targets.scalar_type();
  int64_t batch_size = log_probs.size(1);
  int64_t num_labels = log_probs.size(2);
  int64_t max_input_length = log_probs.size(0);

  int64_t tg_target_stride = 0;
  int64_t max_target_length = 0;
  auto tg_batch_offsets =
      make_tg_batch_offsets(targets, target_lengths, batch_size, tg_target_stride, max_target_length);
  // For 2D targets the alpha width is authoritative (targets.size(1) may be larger).
  if (targets.dim() != 1) {
    max_target_length = log_alpha.size(2) / 2;
  }

  auto input_lengths_t = at::tensor(input_lengths, targets.options().dtype(kLong)).to(log_probs.device());
  auto target_lengths_t = at::tensor(target_lengths, targets.options().dtype(kLong)).to(log_probs.device());
  tg_batch_offsets = tg_batch_offsets.to(log_probs.device());

  // Normalize to the kernel's buffer dtypes (float accumulators, input-dtype
  // grad_out) in case autograd handed us promoted tensors.
  auto log_alpha_acc = log_alpha.to(kFloat);
  auto nll_acc = neg_log_likelihood.to(kFloat);
  auto grad_out_in = grad_out.to(log_probs.scalar_type());

  Tensor log_beta = at::empty_like(log_alpha_acc, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // gradient accumulates log(sum(alpha*beta)) in float, then is rewritten in place.
  auto acc_options = log_probs.options().dtype(kFloat);
  Tensor grad = at::full_like(log_probs, neginf, acc_options, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  CTCLossParams params{batch_size,
                       max_input_length,
                       max_target_length,
                       num_labels,
                       BLANK,
                       log_probs.stride(0),
                       log_probs.stride(1),
                       log_probs.stride(2),
                       log_alpha_acc.stride(0),
                       log_alpha_acc.stride(1),
                       log_alpha_acc.stride(2),
                       tg_target_stride,
                       zero_infinity};
  // grad is (T, N, C); the kernel expects {batch, input, char} order.
  std::array<int64_t, 3> gr_strides = {grad.stride(1), grad.stride(0), grad.stride(2)};
  int64_t grad_out_batch_stride = grad_out_in.stride(0);
  const int64_t la_size = 2 * max_target_length + 1;
  const std::string suffix = ctc_kernel_suffix(log_probs, target_scalar_type);

  using namespace mps;
  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = stream->commandEncoder();

      // beta: one threadgroup per batch item, threads over the label axis.
      auto betaPSO = lib.getPipelineStateForFunc("ctc_loss_beta_" + suffix);
      uint64_t tg_threads = std::min<uint64_t>(la_size, betaPSO.maxTotalThreadsPerThreadgroup);
      getMPSProfiler().beginProfileKernel(betaPSO, "ctc_loss_beta", {log_probs, targets});
      [computeEncoder setComputePipelineState:betaPSO];
      mtl_setArgs(
          computeEncoder, log_beta, log_probs, targets, input_lengths_t, target_lengths_t, tg_batch_offsets, params);
      [computeEncoder dispatchThreadgroups:MTLSizeMake(batch_size, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(tg_threads, 1, 1)];
      getMPSProfiler().endProfileKernel(betaPSO);

      // collect (naive, race-free): parallel over (t, b), sequential over s.
      auto collectPSO = lib.getPipelineStateForFunc("ctc_loss_collect_" + suffix);
      getMPSProfiler().beginProfileKernel(collectPSO, "ctc_loss_collect", {log_probs, targets});
      [computeEncoder setComputePipelineState:collectPSO];
      mtl_setArgs(computeEncoder,
                  grad,
                  grad_out_in,
                  log_alpha_acc,
                  log_beta,
                  log_probs,
                  targets,
                  input_lengths_t,
                  target_lengths_t,
                  tg_batch_offsets,
                  nll_acc,
                  gr_strides,
                  grad_out_batch_stride,
                  params);
      [computeEncoder dispatchThreads:MTLSizeMake(max_input_length, batch_size, 1)
                threadsPerThreadgroup:MTLSizeMake(std::min<uint64_t>(max_input_length,
                                                                     collectPSO.maxTotalThreadsPerThreadgroup),
                                                  1,
                                                  1)];
      getMPSProfiler().endProfileKernel(collectPSO);
    }
  });
  // grad was accumulated in float; return it in the input dtype.
  return grad.to(log_probs.scalar_type());
}

} // namespace at::native
