#pragma once

// Strides and sizes shared by the CTC alpha/beta/collect kernels. Targets are
// passed as the augmented sequence l' (BLANK l_0 BLANK l_1 ... BLANK), so the
// label axis runs over 2 * target_length + 1 entries.
struct CTCLossParams {
  long batch_size;
  long max_input_length;
  long max_target_length;
  long num_labels;
  long BLANK;
  long lp_input_stride;
  long lp_batch_stride;
  long lp_char_stride;
  long la_batch_stride;
  long la_input_stride;
  long la_target_stride;
  long tg_target_stride;
  bool zero_infinity;
};
