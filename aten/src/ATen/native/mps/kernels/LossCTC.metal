#include <ATen/native/mps/kernels/LossCTC.h>
#include <metal_stdlib>

using namespace metal;

// CTC loss on Metal, a direct port of the CUDA forward-backward algorithm in
// aten/src/ATen/native/cuda/LossCTC.cu (Graves et al., section 4.1). All log
// quantities use the log-sum-exp trick for numerical stability.
//
// One threadgroup handles one batch item; the label axis s of the augmented
// target l' (length 2 * target_length + 1) is split across the threads. Both
// alpha and beta march along the time axis sequentially, so every timestep is
// separated by a threadgroup_barrier. log_alpha / log_beta live in device
// memory exactly as in the CUDA version, so threads only need the barrier to
// observe each other's previous-timestep writes.
//
// scalar_t is the input/output dtype (float or half); acc_t is the accumulation
// dtype, always float, used for the DP tables and the log-space gradient. This
// keeps the half path full-precision; only inputs/outputs are half.

// l' maps even indices to BLANK and odd indices to the real target label.
template <typename target_t>
static inline long get_target_prime(
    constant target_t* target,
    long offset,
    long stride,
    long idx,
    long BLANK) {
  if (idx % 2 == 0) {
    return BLANK;
  }
  return target[offset + stride * (idx / 2)];
}

template <typename scalar_t, typename acc_t, typename target_t>
kernel void ctc_loss_alpha(
    device acc_t* log_alpha [[buffer(0)]],
    constant scalar_t* log_probs [[buffer(1)]],
    constant target_t* targets [[buffer(2)]],
    constant long* input_lengths [[buffer(3)]],
    constant long* target_lengths [[buffer(4)]],
    constant long* tg_batch_offsets [[buffer(5)]],
    device acc_t* neg_log_likelihood [[buffer(6)]],
    constant CTCLossParams& p [[buffer(7)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
  const acc_t neginf = -INFINITY;
  const long b = tg_id;
  if (b >= p.batch_size) {
    return;
  }

  const long input_length = input_lengths[b];
  const long target_length = target_lengths[b];
  const long lp_batch_offset = b * p.lp_batch_stride;
  const long la_batch_offset = b * p.la_batch_stride;
  const long tg_batch_offset = tg_batch_offsets[b];
  const long la_size = 2 * p.max_target_length + 1;

  // Empty input: likelihood is 0 iff the target is also empty, else -inf.
  if (input_length == 0) {
    if (ltid == 0) {
      neg_log_likelihood[b] = (target_length == 0) ? 0 : INFINITY;
    }
    return;
  }

  // t = 0 initialization (the equations for alpha_1 above eq (6)).
  for (long s = ltid; s < la_size; s += tg_size) {
    acc_t la;
    if (s == 0) {
      la = log_probs[lp_batch_offset + p.lp_char_stride * p.BLANK];
    } else if (s == 1) {
      la = target_length == 0 ? neginf
                              : log_probs
                                    [lp_batch_offset +
                                     p.lp_char_stride *
                                         get_target_prime(
                                             targets,
                                             tg_batch_offset,
                                             p.tg_target_stride,
                                             1,
                                             p.BLANK)];
    } else {
      la = neginf;
    }
    log_alpha[la_batch_offset + p.la_target_stride * s] = la;
  }

  for (long s = ltid; s < la_size; s += tg_size) {
    // current_char and have_three only depend on s, so compute them once.
    long current_char;
    bool have_three;
    if (s < 2 * target_length + 1 && target_length > 0) {
      current_char = get_target_prime(
          targets, tg_batch_offset, p.tg_target_stride, s, p.BLANK);
      have_three = (s > 1) &&
          (get_target_prime(
               targets, tg_batch_offset, p.tg_target_stride, s - 2, p.BLANK) !=
           current_char);
    } else {
      current_char = p.BLANK;
      have_three = false;
    }
    for (long t = 1; t < p.max_input_length; t++) {
      threadgroup_barrier(mem_flags::mem_device);
      if (t < input_length && s < 2 * target_length + 1) {
        // eq (6)/(7): la1, la2, la3 are the three summands, lamax the max for
        // the log-sum-exp trick.
        acc_t la1 = log_alpha
            [la_batch_offset + p.la_input_stride * (t - 1) +
             p.la_target_stride * s];
        acc_t lamax = la1;
        acc_t la2, la3;
        if (s > 0) {
          la2 = log_alpha
              [la_batch_offset + p.la_input_stride * (t - 1) +
               p.la_target_stride * (s - 1)];
          lamax = max(lamax, la2);
        } else {
          la2 = neginf;
        }
        if (have_three) {
          la3 = log_alpha
              [la_batch_offset + p.la_input_stride * (t - 1) +
               p.la_target_stride * (s - 2)];
          lamax = max(lamax, la3);
        } else {
          la3 = neginf;
        }
        // All summands -inf: pretend lamax is 0, the result stays -inf.
        if (lamax == neginf) {
          lamax = 0;
        }
        log_alpha
            [la_batch_offset + p.la_input_stride * t + p.la_target_stride * s] =
                precise::log(
                    precise::exp(la1 - lamax) + precise::exp(la2 - lamax) +
                    precise::exp(la3 - lamax)) +
            lamax +
            log_probs
                [lp_batch_offset + t * p.lp_input_stride +
                 p.lp_char_stride * current_char];
      } else if (s < la_size) {
        log_alpha
            [la_batch_offset + p.la_input_stride * t + p.la_target_stride * s] =
                neginf;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_device);

  // eq (8): combine the two valid final states into the loss.
  if (ltid == 0) {
    acc_t l1 = log_alpha
        [la_batch_offset + p.la_input_stride * (input_length - 1) +
         p.la_target_stride * (target_length * 2)];
    acc_t l2 = target_length > 0
        ? log_alpha
              [la_batch_offset + p.la_input_stride * (input_length - 1) +
               p.la_target_stride * (target_length * 2 - 1)]
        : neginf;
    acc_t m = max(l1, l2);
    m = (m == neginf) ? 0 : m;
    acc_t log_likelihood =
        precise::log(precise::exp(l1 - m) + precise::exp(l2 - m)) + m;
    neg_log_likelihood[b] = -log_likelihood;
  }
}

// beta recursion ((10)/(11)), the time-reversed counterpart of alpha. log_beta
// is laid out like log_alpha (la_* strides are reused for it).
template <typename scalar_t, typename acc_t, typename target_t>
kernel void ctc_loss_beta(
    device acc_t* log_beta [[buffer(0)]],
    constant scalar_t* log_probs [[buffer(1)]],
    constant target_t* targets [[buffer(2)]],
    constant long* input_lengths [[buffer(3)]],
    constant long* target_lengths [[buffer(4)]],
    constant long* tg_batch_offsets [[buffer(5)]],
    constant CTCLossParams& p [[buffer(6)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
  const acc_t neginf = -INFINITY;
  const long b = tg_id;
  if (b >= p.batch_size) {
    return;
  }

  const long input_length = input_lengths[b];
  const long target_length = target_lengths[b];
  const long lp_batch_offset = b * p.lp_batch_stride;
  const long lb_batch_offset = b * p.la_batch_stride;
  const long tg_batch_offset = tg_batch_offsets[b];
  const long la_size = 2 * p.max_target_length + 1;

  if (input_length == 0) {
    return;
  }

  // beta initialization at t = input_length - 1 (before eq (10)).
  for (long s = ltid; s < la_size; s += tg_size) {
    acc_t lb;
    if (s == 2 * target_length) {
      lb = log_probs
          [lp_batch_offset + (input_length - 1) * p.lp_input_stride +
           p.lp_char_stride * p.BLANK];
    } else if (s == 2 * target_length - 1) { // false when target_length == 0
      long current = get_target_prime(
          targets, tg_batch_offset, p.tg_target_stride, s, p.BLANK);
      lb = log_probs
          [lp_batch_offset + (input_length - 1) * p.lp_input_stride +
           p.lp_char_stride * current];
    } else {
      lb = neginf;
    }
    log_beta
        [lb_batch_offset + (input_length - 1) * p.la_input_stride +
         p.la_target_stride * s] = lb;
  }

  for (long s = ltid; s < la_size; s += tg_size) {
    long current_target_prime;
    bool have_three;
    if (s < 2 * target_length + 1 && target_length > 0) {
      current_target_prime = get_target_prime(
          targets, tg_batch_offset, p.tg_target_stride, s, p.BLANK);
      have_three = (s < 2 * target_length - 1) &&
          (get_target_prime(
               targets, tg_batch_offset, p.tg_target_stride, s + 2, p.BLANK) !=
           current_target_prime);
    } else {
      current_target_prime = p.BLANK;
      have_three = false;
    }
    // Go backward in t, skipping the last timestep initialized above.
    for (long t = p.max_input_length - 2; t >= 0; t--) {
      threadgroup_barrier(mem_flags::mem_device);
      if (t < input_length - 1 && s < 2 * target_length + 1) {
        acc_t lb1 = log_beta
            [lb_batch_offset + p.la_input_stride * (t + 1) +
             p.la_target_stride * s];
        acc_t lbmax = lb1;
        acc_t lb2, lb3;
        if (s < 2 * target_length) {
          lb2 = log_beta
              [lb_batch_offset + p.la_input_stride * (t + 1) +
               p.la_target_stride * (s + 1)];
          lbmax = max(lbmax, lb2);
        } else {
          lb2 = neginf;
        }
        if (have_three) {
          lb3 = log_beta
              [lb_batch_offset + p.la_input_stride * (t + 1) +
               p.la_target_stride * (s + 2)];
          lbmax = max(lbmax, lb3);
        } else {
          lb3 = neginf;
        }
        if (lbmax == neginf) {
          lbmax = 0;
        }
        log_beta
            [lb_batch_offset + p.la_input_stride * t + p.la_target_stride * s] =
                precise::log(
                    precise::exp(lb1 - lbmax) + precise::exp(lb2 - lbmax) +
                    precise::exp(lb3 - lbmax)) +
            lbmax +
            log_probs
                [lp_batch_offset + t * p.lp_input_stride +
                 p.lp_char_stride * current_target_prime];
      } else if (
          s < la_size &&
          (((target_length == 0) && (s > 0)) || (s >= 2 * target_length + 1) ||
           (t >= input_length))) {
        log_beta
            [lb_batch_offset + p.la_input_stride * t + p.la_target_stride * s] =
                neginf;
      }
    }
  }
}

// Gradient collection, the naive variant of eq (16): parallel over (b, t) with
// a sequential loop over s, so each gradient element is owned by exactly one
// thread and no atomics are needed. The CUDA code keeps an atomic
// large-alphabet path too; MPS always uses this race-free version. gradient
// (acc_t) first accumulates log(alpha*beta), then is rewritten to the gradient.
template <typename scalar_t, typename acc_t, typename target_t>
kernel void ctc_loss_collect(
    device acc_t* gradient [[buffer(0)]],
    constant scalar_t* grad_out [[buffer(1)]],
    constant acc_t* log_alpha [[buffer(2)]],
    constant acc_t* log_beta [[buffer(3)]],
    constant scalar_t* log_probs [[buffer(4)]],
    constant target_t* targets [[buffer(5)]],
    constant long* input_lengths [[buffer(6)]],
    constant long* target_lengths [[buffer(7)]],
    constant long* tg_batch_offsets [[buffer(8)]],
    constant acc_t* neg_log_likelihood [[buffer(9)]],
    constant long* gr_strides [[buffer(10)]], // {batch, input, char}
    constant long& grad_out_batch_stride [[buffer(11)]],
    constant CTCLossParams& p [[buffer(12)]],
    uint2 tid [[thread_position_in_grid]]) {
  const acc_t neginf = -INFINITY;
  const long t = tid.x;
  const long b = tid.y;
  if (t >= p.max_input_length || b >= p.batch_size) {
    return;
  }

  const long input_length = input_lengths[b];
  const long target_length = target_lengths[b];
  const long gr_batch_offset = b * gr_strides[0];
  const long lp_batch_offset = b * p.lp_batch_stride;
  const long la_batch_offset = b * p.la_batch_stride;
  const long tg_batch_offset = tg_batch_offsets[b];

  // collected[b, t, l'[s]] "log+=" log_alpha[t, s] + log_beta[t, s].
  for (long s = 0; s < 2 * p.max_target_length + 1; s++) {
    if (s < 2 * target_length + 1) { // if target_length == 0 then only s == 0
      long current_target_prime = get_target_prime(
          targets, tg_batch_offset, p.tg_target_stride, s, p.BLANK);
      acc_t log_alpha_beta = log_alpha
                                 [la_batch_offset + p.la_input_stride * t +
                                  p.la_target_stride * s] +
          log_beta[la_batch_offset + p.la_input_stride * t +
                   p.la_target_stride * s];
      long idx = gr_batch_offset + t * gr_strides[1] +
          gr_strides[2] * current_target_prime;
      acc_t lcab = gradient[idx];
      if (lcab == neginf) {
        gradient[idx] = log_alpha_beta;
      } else {
        acc_t m = max(lcab, log_alpha_beta);
        gradient[idx] =
            precise::log(
                precise::exp(lcab - m) + precise::exp(log_alpha_beta - m)) +
            m;
      }
    }
  }

  acc_t nll = neg_log_likelihood[b];
  acc_t gr = grad_out[b * grad_out_batch_stride];

  for (long c = 0; c < p.num_labels; c++) {
    long idx = gr_batch_offset + t * gr_strides[1] + gr_strides[2] * c;
    if (t < input_length && (!p.zero_infinity || nll != INFINITY)) {
      acc_t lp = log_probs
          [lp_batch_offset + t * p.lp_input_stride + p.lp_char_stride * c];
      gradient[idx] =
          (precise::exp(lp) - precise::exp(gradient[idx] + nll - lp)) * gr;
    } else {
      gradient[idx] = 0;
    }
  }
}

#define INSTANTIATE_CTC(DTYPE, ADTYPE, TDTYPE)                                \
  template [[host_name("ctc_loss_alpha_" #DTYPE "_" #TDTYPE)]] kernel void    \
  ctc_loss_alpha<DTYPE, ADTYPE, TDTYPE>(                                      \
      device ADTYPE*,                                                         \
      constant DTYPE*,                                                        \
      constant TDTYPE*,                                                       \
      constant long*,                                                         \
      constant long*,                                                         \
      constant long*,                                                         \
      device ADTYPE*,                                                         \
      constant CTCLossParams&,                                                \
      uint,                                                                   \
      uint,                                                                   \
      uint);                                                                  \
  template [[host_name("ctc_loss_beta_" #DTYPE "_" #TDTYPE)]] kernel void     \
  ctc_loss_beta<DTYPE, ADTYPE, TDTYPE>(                                       \
      device ADTYPE*,                                                         \
      constant DTYPE*,                                                        \
      constant TDTYPE*,                                                       \
      constant long*,                                                         \
      constant long*,                                                         \
      constant long*,                                                         \
      constant CTCLossParams&,                                                \
      uint,                                                                   \
      uint,                                                                   \
      uint);                                                                  \
  template [[host_name("ctc_loss_collect_" #DTYPE "_" #TDTYPE)]] kernel void  \
  ctc_loss_collect<DTYPE, ADTYPE, TDTYPE>(                                    \
      device ADTYPE*,                                                         \
      constant DTYPE*,                                                        \
      constant ADTYPE*,                                                       \
      constant ADTYPE*,                                                       \
      constant DTYPE*,                                                        \
      constant TDTYPE*,                                                       \
      constant long*,                                                         \
      constant long*,                                                         \
      constant long*,                                                         \
      constant ADTYPE*,                                                       \
      constant long*,                                                         \
      constant long&,                                                         \
      constant CTCLossParams&,                                                \
      uint2);

// (scalar_t, acc_t, target_t); acc_t is float for both, so half accumulates in float.
INSTANTIATE_CTC(float, float, int);
INSTANTIATE_CTC(float, float, long);
INSTANTIATE_CTC(half, float, int);
INSTANTIATE_CTC(half, float, long);
