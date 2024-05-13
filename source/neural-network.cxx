/*
Copyright (C) 2024 Gregory Teicher

Author: Gregory Teicher

This file is part of Marigold.

Marigold is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Marigold is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Marigold.
If not, see <https://www.gnu.org/licenses/>.
*/

#include <float.h>
#include <limits.h>
#include <stdalign.h>
#include <stdatomic.h>
#include <stdint.h>

static_assert(sizeof(float) == 4);
static_assert(sizeof(double) == 8);

int32_t constexpr channels_rgb = 3;
int32_t constexpr channels_rgba = 4;

extern "C"
{
  extern uint8_t *__heap_base;
  // extern uint8_t __heap_base[];

  auto heap_base() -> void *
  {
    return (void *)&__heap_base;
  }

  auto malloc() -> void *
  {
    return nullptr;
  }

  auto free(void *) -> void
  {
  }

  auto zero(float *x, int32_t size) -> void
  {
    for (int32_t i = 0; i < size; ++i)
    {
      x[i] = 0.0f;
    }
  }

  auto accumulate(float const *x, float *y, int32_t size) -> void
  {
    for (int32_t i = 0; i < size; ++i)
    {
      y[i] += x[i];
    }
  }

  auto merge(float const *x_1, float const *x_2, float *y, int32_t size) -> void
  {
    for (int32_t i = 0; i < size; ++i)
    {
      y[i] = x_1[i] + x_2[i];
    }
  }

  auto matrix_multiply_accumulating(float const *a,
                                    float const *b,
                                    float *c,
                                    int32_t m,
                                    int32_t k,
                                    int32_t n) -> void;
  auto matrix_multiply_accumulating_transpose_a(float const *a,
                                                float const *b,
                                                float *c,
                                                int32_t m,
                                                int32_t k,
                                                int32_t n) -> void;
  auto matrix_multiply_accumulating_transpose_b(float const *a,
                                                float const *b,
                                                float *c,
                                                int32_t m,
                                                int32_t k,
                                                int32_t n) -> void;
  auto matrix_multiply_accumulating_transpose_b_alt(float const *a,
                                                    float const *b,
                                                    float *c,
                                                    float *d,
                                                    int32_t m,
                                                    int32_t k,
                                                    int32_t n) -> void;
  auto matrix_multiply_accumulating_blocked(float const *a,
                                            float const *b,
                                            float *c,
                                            int32_t m,
                                            int32_t k,
                                            int32_t n) -> void;
  auto matrix_multiply_accumulating_blocked_transpose_a(float const *a,
                                                        float const *b,
                                                        float *c,
                                                        int32_t m,
                                                        int32_t k,
                                                        int32_t n) -> void;
  auto matrix_multiply_accumulating_blocked_transpose_b(float const *a,
                                                        float const *b,
                                                        float *c,
                                                        int32_t m,
                                                        int32_t k,
                                                        int32_t n) -> void;
  auto matrix_multiply_accumulating_blocked_transpose_b_alt(float const *a,
                                                            float const *b,
                                                            float *c,
                                                            float *d,
                                                            int32_t m,
                                                            int32_t k,
                                                            int32_t n) -> void;

  auto pad(float const *x, float *x_pad, int32_t height, int32_t width, int32_t channels) -> void;
  auto unpad(float const *x_pad, float *x, int32_t height, int32_t width, int32_t channels) -> void;

  auto hard_swish_forward(float const *x, float *y, int32_t size) -> void;
  auto hard_swish_backward(float const *d_y, float *d_x, float const *x, int32_t size) -> void;

  auto pixel_shuffle_forward(float const *x, float *y, int32_t x_h, int32_t x_w, int32_t x_c) -> void;
  auto pixel_shuffle_backward(float const *d_y, float *d_x, int32_t x_h, int32_t x_w, int32_t x_c) -> void;

  auto instance_normalization_forward(float const *x,
                                      float *y,
                                      float const *gamma,
                                      float const *beta,
                                      float *sample_mean,
                                      float *sample_std_dev,
                                      float epsilon,
                                      int32_t x_h,
                                      int32_t x_w,
                                      int32_t x_c) -> void;
  auto instance_normalization_backward(float const *d_y,
                                       float *d_x,
                                       float *d_gamma,
                                       float *d_beta,
                                       float const *gamma,
                                       float const *sample_mean,
                                       float const *sample_std_dev,
                                       float const *x,
                                       float *sum_1,
                                       [[maybe_unused]] float *sum_2,
                                       int32_t x_h,
                                       int32_t x_w,
                                       int32_t x_c) -> void;

  auto pointwise_convolution_forward(float const *in,
                                     float *out,
                                     float const *kernel,
                                     float const *bias,
                                     int32_t height,
                                     int32_t width,
                                     int32_t channels_in,
                                     int32_t channels_out) -> void;
  auto pointwise_convolution_forward_test_1(float const *in,
                                            float *out,
                                            float const *kernel,
                                            float const *bias,
                                            float *kernel_,
                                            int32_t height,
                                            int32_t width,
                                            int32_t channels_in,
                                            int32_t channels_out) -> void;
  auto pointwise_convolution_forward_test_2(float const *in,
                                            float *out,
                                            float const *kernel,
                                            float const *bias,
                                            float *kernel_,
                                            int32_t height,
                                            int32_t width,
                                            int32_t channels_in,
                                            int32_t channels_out) -> void;
  auto pointwise_convolution_forward_test_3(float const *in,
                                            float *out,
                                            float const *kernel,
                                            float const *bias,
                                            float *kernel_,
                                            int32_t height,
                                            int32_t width,
                                            int32_t channels_in,
                                            int32_t channels_out) -> void;
  auto pointwise_convolution_forward_test_4(float const *in,
                                            float *out,
                                            float const *kernel,
                                            float const *bias,
                                            float *kernel_,
                                            int32_t height,
                                            int32_t width,
                                            int32_t channels_in,
                                            int32_t channels_out) -> void;
  auto pointwise_convolution_forward_test_5(float const *in,
                                            float *out,
                                            float const *kernel,
                                            float const *bias,
                                            float *kernel_,
                                            int32_t height,
                                            int32_t width,
                                            int32_t channels_in,
                                            int32_t channels_out) -> void;
  auto pointwise_convolution_backward(float const *d_out,
                                      float *d_in,
                                      float *d_kernel,
                                      float *d_bias,
                                      float const *in,
                                      float const *kernel,
                                      [[maybe_unused]] float *kernel_buffer,
                                      int32_t height,
                                      int32_t width,
                                      int32_t channels_in,
                                      int32_t channels_out) -> void;
  auto pointwise_convolution_backward_test_1(float const *d_out,
                                             float *d_in,
                                             float *d_kernel,
                                             float *d_bias,
                                             float const *in,
                                             float const *kernel,
                                             [[maybe_unused]] float *kernel_buffer,
                                             int32_t height,
                                             int32_t width,
                                             int32_t channels_in,
                                             int32_t channels_out) -> void;
  auto pointwise_convolution_backward_test_2(float const *d_out,
                                             float *d_in,
                                             float *d_kernel,
                                             float *d_bias,
                                             float const *in,
                                             float const *kernel,
                                             [[maybe_unused]] float *kernel_buffer,
                                             int32_t height,
                                             int32_t width,
                                             int32_t channels_in,
                                             int32_t channels_out) -> void;
  auto pointwise_convolution_backward_test_3(float const *d_out,
                                             float *d_in,
                                             float *d_kernel,
                                             float *d_bias,
                                             float const *in,
                                             float const *kernel,
                                             [[maybe_unused]] float *kernel_buffer,
                                             int32_t height,
                                             int32_t width,
                                             int32_t channels_in,
                                             int32_t channels_out) -> void;
  auto pointwise_convolution_backward_d_out_test_1(float const *d_out,
                                                   float *d_in,
                                                   float *d_kernel,
                                                   float *d_bias,
                                                   float const *in,
                                                   float const *kernel,
                                                   [[maybe_unused]] float *kernel_buffer,
                                                   int32_t height,
                                                   int32_t width,
                                                   int32_t channels_in,
                                                   int32_t channels_out) -> void;
  auto pointwise_convolution_backward_d_out_test_2(float const *d_out,
                                                   float *d_in,
                                                   float *d_kernel,
                                                   float *d_bias,
                                                   float const *in,
                                                   float const *kernel,
                                                   [[maybe_unused]] float *kernel_buffer,
                                                   int32_t height,
                                                   int32_t width,
                                                   int32_t channels_in,
                                                   int32_t channels_out) -> void;
  auto pointwise_convolution_backward_d_out_test_3(float const *d_out,
                                                   float *d_in,
                                                   float *d_kernel,
                                                   float *d_bias,
                                                   float const *in,
                                                   float const *kernel,
                                                   [[maybe_unused]] float *kernel_buffer,
                                                   int32_t height,
                                                   int32_t width,
                                                   int32_t channels_in,
                                                   int32_t channels_out) -> void;

  auto depthwise_convolution_forward(float const *x,
                                     float *y,
                                     float const *k,
                                     float const *b,
                                     float *x_pad,
                                     int32_t height,
                                     int32_t width,
                                     int32_t channels) -> void;
  auto depthwise_convolution_forward_unpadded(float const *x,
                                              float *y,
                                              float const *k,
                                              float const *b,
                                              // float * x_pad, // temp
                                              int32_t height,
                                              int32_t width,
                                              int32_t channels) -> void;
  auto depthwise_convolution_forward_unpadded_(float const *x,
                                               float *y,
                                               float const *k,
                                               float const *b,
                                               //  float * x_pad, // temp
                                               int32_t height,
                                               int32_t width,
                                               int32_t channels) -> void;
  auto depthwise_convolution_backward(float const *d_y,
                                      float *d_x,
                                      float *d_k,
                                      float *d_b,
                                      float const *k,
                                      float const *x_pad,
                                      float *d_x_pad,
                                      int32_t height,
                                      int32_t width,
                                      int32_t channels) -> void;
  auto depthwise_convolution_backward_unpadded(float const *d_y,
                                               float *d_x,
                                               float *d_k,
                                               float *d_b,
                                               float const *k,
                                               //  float const * x_pad,
                                               float const *x,
                                               //  float * d_x_pad,
                                               int32_t height,
                                               int32_t width,
                                               int32_t channels) -> void;
  auto depthwise_convolution_backward_unpadded_(float const *d_y,
                                                float *d_x,
                                                float *d_k,
                                                float *d_b,
                                                float const *k,
                                                //  float const * x_pad,
                                                float const *x,
                                                //  float * d_x_pad,
                                                int32_t height,
                                                int32_t width,
                                                int32_t channels) -> void;

  auto patchified_convolution_forward(float const *x,
                                      float *y,
                                      float const *k,
                                      float const *b,
                                      int32_t x_h,
                                      int32_t x_w,
                                      int32_t k_n) -> void;
  auto patchified_convolution_backward(float const *d_y,
                                       [[maybe_unused]] float *d_x,
                                       float *d_k,
                                       float *d_b,
                                       float const *x,
                                       [[maybe_unused]] float const *k,
                                       int32_t x_h,
                                       int32_t x_w,
                                       int32_t k_n) -> void;

  auto patchified_convolution_im2row_forward(float const *x,
                                             float *y,
                                             float const *k,
                                             float const *b,
                                             float *rows,
                                             int32_t x_h,
                                             int32_t x_w,
                                             int32_t k_n) -> void;
  auto patchified_convolution_im2row_forward_(float const *x,
                                              float *y,
                                              float const *k,
                                              float const *b,
                                              float *rows,
                                              int32_t x_h,
                                              int32_t x_w,
                                              int32_t k_n) -> void;
  auto patchified_convolution_im2row_backward(float const *d_y,
                                              [[maybe_unused]] float *d_x,
                                              float *d_k,
                                              float *d_b,
                                              [[maybe_unused]] float const *x,
                                              [[maybe_unused]] float const *k,
                                              float const *rows,
                                              int32_t x_h,
                                              int32_t x_w,
                                              int32_t k_n) -> void;

  auto update_parameters(float const *gradients,
                         float *parameters,
                         float *m,
                         float *v,
                         int32_t size,
                         float beta1,
                         float beta2,
                         float epsilon,
                         [[maybe_unused]] float schedule_multiplier,
                         float learning_rate,
                         [[maybe_unused]] float weight_decay,
                         int32_t t) -> void;

  auto resize_bilinear_rgba_to_rgb(uint8_t const *x, float *y, int32_t xHeight, int32_t xWidth, int32_t yHeight, int32_t yWidth) -> void;

  auto rotate_bilinear(float const *original, float *rotated, int32_t height, int32_t width, float cosTheta, float sinTheta, float padValue) -> void;

  auto draw_gaussians(float *data, int32_t height, int32_t width, int32_t channels, float const *coords, float sigma) -> void;
  // auto draw_gaussians_fast(float * data, int32_t height, int32_t width, int32_t channels, float const * coords, float sigma) -> void;

  auto mean_squared_error_forward(float const *x_pred, float const *x_true, int32_t size) -> float;
  auto mean_squared_error_backward(float d_y, float *d_x, float const *x_pred, float const *x_true, int32_t size) -> void;

  auto flip_horizontal(float *x, int32_t height, int32_t width) -> void;
  auto flip_vertical(float *x, int32_t height, int32_t width) -> void;
  auto brightness_adjustment(float *x, int32_t height, int32_t width, float brightness_adjustment) -> void;

  extern auto exp(float base) -> float;
  extern auto log(float base) -> float;
  extern auto pow(float base, int32_t exp) -> float;

  auto _start() -> void
  {
  }
}

namespace
{
  template <typename T>
  constexpr auto min(T a, T b) -> T
  {
    return a <= b ? a : b;
  }

  template <typename T>
  auto max(T a, T b) -> T
  {
    return a >= b ? a : b;
  }

  template <typename T>
  constexpr auto square(T x) -> T
  {
    return x * x;
  }

  template <typename T>
  struct View1D
  {
    View1D(T *data, int32_t i_0) : data(data), shape{i_0}
    {
    }

    auto operator()(int32_t i_0) const -> T &
    {
      return data[i_0];
    }

    T *data = nullptr;
    int32_t shape[1] = {};
  };

  template <typename T>
  struct View2D
  {
    View2D(T *data, int32_t i_0, int32_t i_1) : data(data), shape{i_0, i_1}
    {
    }

    auto operator()(int32_t i_0, int32_t i_1) const -> T &
    {
      return data[i_0 * shape[1] + i_1];
    }

    T *data = nullptr;
    int32_t shape[2] = {};
  };

  template <typename T>
  struct View3D
  {
    View3D(T *data, int32_t i_0, int32_t i_1, int32_t i_2) : data(data), shape{i_0, i_1, i_2}
    {
    }

    auto operator()(int32_t i_0, int32_t i_1, int32_t i_2) const -> T &
    {
      return data[i_0 * shape[1] * shape[2] + i_1 * shape[2] + i_2];
    }

    T *data = nullptr;
    int32_t shape[3] = {};
  };

  template <typename T>
  struct View4D
  {
    View4D(T *data, int32_t i_0, int32_t i_1, int32_t i_2, int32_t i_3) : data(data), shape{i_0, i_1, i_2, i_3}
    {
    }

    auto operator()(int32_t i_0, int32_t i_1, int32_t i_2, int32_t i_3) const -> T &
    {
      return data[i_0 * shape[1] * shape[2] * shape[3] + i_1 * shape[2] * shape[3] + i_2 * shape[3] + i_3];
    }

    T *data = nullptr;
    int32_t shape[4] = {};
  };

  template <typename T>
  struct View5D
  {
    View5D(T *data, int32_t i_0, int32_t i_1, int32_t i_2, int32_t i_3, int32_t i_4) : data(data), shape{i_0, i_1, i_2, i_3, i_4}
    {
    }

    auto operator()(int32_t i_0, int32_t i_1, int32_t i_2, int32_t i_3, int32_t i_4) const -> T &
    {
      return data[i_0 * shape[1] * shape[2] * shape[3] * shape[4] + i_1 * shape[2] * shape[3] * shape[4] + i_2 * shape[3] * shape[4] + i_3 * shape[4] + i_4];
    }

    T *data = nullptr;
    int32_t shape[5] = {};
  };
}

auto matrix_multiply_accumulating(float const *a,
                                  float const *b,
                                  float *c,
                                  int32_t m,
                                  int32_t k,
                                  int32_t n) -> void
{
  for (int32_t i_m = 0; i_m < m; ++i_m)
  {
    for (int32_t i_k = 0; i_k < k; ++i_k)
    {
      float temp = a[i_m * k + i_k];
      for (int32_t i_n = 0; i_n < n; ++i_n)
      {
        c[i_m * n + i_n] += temp * b[i_k * n + i_n];
      }
    }
  }
}

auto matrix_multiply_accumulating_transpose_a(float const *a,
                                              float const *b,
                                              float *c,
                                              int32_t m,
                                              int32_t k,
                                              int32_t n) -> void
{
  for (int32_t i_m = 0; i_m < m; ++i_m)
  {
    for (int32_t i_k = 0; i_k < k; ++i_k)
    {
      float temp = a[i_k * m + i_m];
      for (int32_t i_n = 0; i_n < n; ++i_n)
      {
        c[i_m * n + i_n] += temp * b[i_k * n + i_n];
      }
    }
  }
}

auto matrix_multiply_accumulating_transpose_b(float const *a,
                                              float const *b,
                                              float *c,
                                              int32_t m,
                                              int32_t k,
                                              int32_t n) -> void
{
  for (int32_t i_m = 0; i_m < m; ++i_m)
  {
    for (int32_t i_k = 0; i_k < k; ++i_k)
    {
      float temp = a[i_m * k + i_k];
      for (int32_t i_n = 0; i_n < n; ++i_n)
      {
        c[i_m * n + i_n] += temp * b[i_n * k + i_k];
      }
    }
  }
}

auto matrix_multiply_accumulating_transpose_b_alt(float const *a,
                                                  float const *b,
                                                  float *c,
                                                  float *d,
                                                  int32_t m,
                                                  int32_t k,
                                                  int32_t n) -> void
{
  for (int32_t i_k = 0; i_k < k; ++i_k)
  {
    for (int32_t i_n = 0; i_n < n; ++i_n)
    {
      d[i_k * n + i_n] = b[i_n * k + i_k];
    }
  }

  for (int32_t i_m = 0; i_m < m; ++i_m)
  {
    for (int32_t i_k = 0; i_k < k; ++i_k)
    {
      float temp = a[i_m * k + i_k];
      for (int32_t i_n = 0; i_n < n; ++i_n)
      {
        c[i_m * n + i_n] += temp * d[i_k * n + i_n];
      }
    }
  }
}

auto matrix_multiply_accumulating_blocked(float const *a,
                                          float const *b,
                                          float *c,
                                          int32_t m,
                                          int32_t k,
                                          int32_t n) -> void
{
  int32_t constexpr block_size = 16;
  for (int32_t i_k = 0; i_k < k; i_k += block_size)
  {
    for (int32_t i_m = 0; i_m < m; ++i_m)
    {
      for (int32_t i_kk = i_k; i_kk < min(i_k + block_size, k); ++i_kk)
      {
        float temp = a[i_m * k + i_kk];
        for (int32_t i_n = 0; i_n < n; ++i_n)
        {
          c[i_m * n + i_n] += temp * b[i_kk * n + i_n];
        }
      }
    }
  }
}

auto matrix_multiply_accumulating_blocked_transpose_a(float const *a,
                                                      float const *b,
                                                      float *c,
                                                      int32_t m,
                                                      int32_t k,
                                                      int32_t n) -> void
{
  int32_t constexpr block_size = 16;
  for (int32_t i_k = 0; i_k < k; i_k += block_size)
  {
    for (int32_t i_m = 0; i_m < m; ++i_m)
    {
      for (int32_t i_kk = i_k; i_kk < min(i_k + block_size, k); ++i_kk)
      {
        float temp = a[i_kk * m + i_m];
        for (int32_t i_n = 0; i_n < n; ++i_n)
        {
          c[i_m * n + i_n] += temp * b[i_kk * n + i_n];
        }
      }
    }
  }
}

auto matrix_multiply_accumulating_blocked_transpose_b(float const *a,
                                                      float const *b,
                                                      float *c,
                                                      int32_t m,
                                                      int32_t k,
                                                      int32_t n) -> void
{
  int32_t constexpr block_size = 16;
  for (int32_t i_k = 0; i_k < k; i_k += block_size)
  {
    for (int32_t i_m = 0; i_m < m; ++i_m)
    {
      for (int32_t i_kk = i_k; i_kk < min(i_k + block_size, k); ++i_kk)
      {
        float temp = a[i_m * k + i_kk];
        for (int32_t i_n = 0; i_n < n; ++i_n)
        {
          c[i_m * n + i_n] += temp * b[i_n * k + i_kk];
        }
      }
    }
  }
}

auto matrix_multiply_accumulating_blocked_transpose_b_alt(float const *a,
                                                          float const *b,
                                                          float *c,
                                                          float *d,
                                                          int32_t m,
                                                          int32_t k,
                                                          int32_t n) -> void
{
  for (int32_t i_k = 0; i_k < k; ++i_k)
  {
    for (int32_t i_n = 0; i_n < n; ++i_n)
    {
      d[i_k * n + i_n] = b[i_n * k + i_k];
    }
  }

  int32_t constexpr block_size = 16;
  for (int32_t i_k = 0; i_k < k; i_k += block_size)
  {
    for (int32_t i_m = 0; i_m < m; ++i_m)
    {
      for (int32_t i_kk = i_k; i_kk < min(i_k + block_size, k); ++i_kk)
      {
        float temp = a[i_m * k + i_kk];
        for (int32_t i_n = 0; i_n < n; ++i_n)
        {
          c[i_m * n + i_n] += temp * d[i_kk * n + i_n];
        }
      }
    }
  }
}

auto pad(float const *x, float *x_pad, int32_t height, int32_t width, int32_t channels) -> void
{
  // int32_t constexpr padding = 3;
  // int32_t constexpr padding = 2;
  int32_t constexpr padding = 1;

  int32_t padded_height = height + (2 * padding);
  int32_t padded_width = width + (2 * padding);

  View3D x_view = {x, height, width, channels};
  View3D x_pad_view = {x_pad, padded_height, padded_width, channels};

  for (int32_t h = 0; h < padding; ++h)
  {
    for (int32_t w = 0; w < padded_width; ++w)
    {
      for (int32_t c = 0; c < channels; ++c)
      {
        x_pad_view(h, w, c) = 0.0f;
      }
    }
  }

  for (int32_t h = padded_height - padding; h < padded_height; ++h)
  {
    for (int32_t w = 0; w < padded_width; ++w)
    {
      for (int32_t c = 0; c < channels; ++c)
      {
        x_pad_view(h, w, c) = 0.0f;
      }
    }
  }

  for (int32_t h = padding; h < padded_height - padding; ++h)
  {
    for (int32_t w = 0; w < padding; ++w)
    {
      for (int32_t c = 0; c < channels; ++c)
      {
        x_pad_view(h, w, c) = 0.0f;
      }
    }
  }

  for (int32_t h = padding; h < padded_height - padding; ++h)
  {
    for (int32_t w = padded_width - padding; w < padded_width; ++w)
    {
      for (int32_t c = 0; c < channels; ++c)
      {
        x_pad_view(h, w, c) = 0.0f;
      }
    }
  }

  for (int32_t h = 0; h < height; ++h)
  {
    for (int32_t w = 0; w < width; ++w)
    {
      for (int32_t c = 0; c < channels; ++c)
      {
        x_pad_view(padding + h, padding + w, c) = x_view(h, w, c);
      }
    }
  }
}

auto unpad(float const *x_pad, float *x, int32_t height, int32_t width, int32_t channels) -> void
{
  // int32_t constexpr padding = 3;
  // int32_t constexpr padding = 2;
  int32_t constexpr padding = 1;

  int32_t padded_height = height + (2 * padding);
  int32_t padded_width = width + (2 * padding);

  View3D x_view = {x, height, width, channels};
  View3D x_pad_view = {x_pad, padded_height, padded_width, channels};

  for (int32_t h = 0; h < height; ++h)
  {
    for (int32_t w = 0; w < width; ++w)
    {
      for (int32_t c = 0; c < channels; ++c)
      {
        x_view(h, w, c) += x_pad_view(padding + h, padding + w, c);
      }
    }
  }
}

auto hard_swish_forward(float const *x, float *y, int32_t size) -> void
{
  for (int32_t i = 0; i < size; ++i)
  {
    if (x[i] <= -3.0f)
    {
      y[i] = 0.0f;
    }
    else if (x[i] >= 3.0f)
    {
      y[i] = x[i];
    }
    else
    {
      y[i] = x[i] * (x[i] + 3.0f) * (1.0f / 6.0f);
    }
  }
}

auto hard_swish_backward(float const *d_y, float *d_x, float const *x, int32_t size) -> void
{
  for (int32_t i = 0; i < size; ++i)
  {
    float temp;
    if (x[i] <= -3.0f)
    {
      temp = 0.0f;
    }
    else if (x[i] >= 3.0f)
    {
      temp = 1.0f;
    }
    else
    {
      temp = (2.0f * x[i] + 3.0f) * (1.0f / 6.0f);
    }
    d_x[i] += temp * d_y[i];
  }
}

auto pixel_shuffle_forward(float const *x, float *y, int32_t x_h, int32_t x_w, int32_t x_c) -> void
{
  int32_t constexpr scale = 4;

  for (int32_t h = 0; h < x_h * scale; ++h)
  {
    for (int32_t w = 0; w < x_w * scale; ++w)
    {
      for (int32_t c = 0; c < x_c / square(scale); ++c)
      {
        int32_t x_i = (h / scale) * x_w * x_c;
        x_i += (w / scale) * x_c;
        x_i += c * scale * scale + (h % scale) * scale + (w % scale);

        int32_t y_i = h * (x_w * scale) * (x_c / square(scale));
        y_i += w * (x_c / square(scale));
        y_i += c;

        y[y_i] = x[x_i];
      }
    }
  }
}

auto pixel_shuffle_backward(float const *d_y, float *d_x, int32_t x_h, int32_t x_w, int32_t x_c) -> void
{
  int32_t constexpr scale = 4;

  for (int32_t h = 0; h < x_h * scale; ++h)
  {
    for (int32_t w = 0; w < x_w * scale; ++w)
    {
      for (int32_t c = 0; c < x_c / square(scale); ++c)
      {
        int32_t d_y_i = h * (x_w * scale) * (x_c / square(scale));
        d_y_i += w * (x_c / square(scale));
        d_y_i += c;

        int32_t d_x_i = (h / scale) * x_w * x_c;
        d_x_i += (w / scale) * x_c;
        d_x_i += c * scale * scale + (h % scale) * scale + (w % scale);

        d_x[d_x_i] = d_y[d_y_i];
      }
    }
  }
}

auto instance_normalization_forward(float const *x,
                                    float *y,
                                    float const *gamma,
                                    float const *beta,
                                    float *sample_mean,
                                    float *sample_std_dev,
                                    float epsilon,
                                    int32_t x_h,
                                    int32_t x_w,
                                    int32_t x_c) -> void
{
  int32_t num = x_h * x_w;

  for (int32_t c = 0; c < x_c; ++c)
  {
    sample_mean[c] = 0.0f;
    sample_std_dev[c] = 0.0f;
  }

  for (int32_t h = 0; h < x_h; ++h)
  {
    for (int32_t w = 0; w < x_w; ++w)
    {
      for (int32_t c = 0; c < x_c; ++c)
      {
        int32_t i = h * x_w * x_c + w * x_c + c;

        sample_mean[c] += x[i];
      }
    }
  }

  for (int32_t c = 0; c < x_c; ++c)
  {
    sample_mean[c] /= num;
  }

  for (int32_t h = 0; h < x_h; ++h)
  {
    for (int32_t w = 0; w < x_w; ++w)
    {
      for (int32_t c = 0; c < x_c; ++c)
      {
        int32_t i = h * x_w * x_c + w * x_c + c;

        sample_std_dev[c] += square(x[i] - sample_mean[c]);
      }
    }
  }

  for (int32_t c = 0; c < x_c; ++c)
  {
    sample_std_dev[c] /= num;
  }

  for (int32_t c = 0; c < x_c; ++c)
  {
    sample_std_dev[c] = __builtin_sqrtf(sample_std_dev[c] + epsilon);
  }

  for (int32_t h = 0; h < x_h; ++h)
  {
    for (int32_t w = 0; w < x_w; ++w)
    {
      for (int32_t c = 0; c < x_c; ++c)
      {
        int32_t i = h * x_w * x_c + w * x_c + c;

        y[i] = x[i];
        y[i] -= sample_mean[c];
        y[i] /= sample_std_dev[c];
        y[i] *= gamma[c];
        y[i] += beta[c];
      }
    }
  }
}

auto instance_normalization_backward(float const *d_y,
                                     float *d_x,
                                     float *d_gamma,
                                     float *d_beta,
                                     float const *gamma,
                                     float const *sample_mean,
                                     float const *sample_std_dev,
                                     float const *x,
                                     float *sum_1,
                                     [[maybe_unused]] float *sum_2,
                                     //  float * d_sample_mean,
                                     //  float * d_sample_std_dev,
                                     int32_t x_h,
                                     int32_t x_w,
                                     int32_t x_c) -> void
{
  int32_t num = x_h * x_w;

  for (int32_t h = 0; h < x_h; ++h)
  {
    for (int32_t w = 0; w < x_w; ++w)
    {
      for (int32_t c = 0; c < x_c; ++c)
      {
        int32_t i = h * x_w * x_c + w * x_c + c;

        d_beta[c] += d_y[i];
      }
    }
  }

  for (int32_t h = 0; h < x_h; ++h)
  {
    for (int32_t w = 0; w < x_w; ++w)
    {
      for (int32_t c = 0; c < x_c; ++c)
      {
        int32_t i = h * x_w * x_c + w * x_c + c;

        d_gamma[c] += d_y[i] * ((x[i] - sample_mean[c]) / sample_std_dev[c]);
      }
    }
  }

  //

  for (int32_t h = 0; h < x_h; ++h)
  {
    for (int32_t w = 0; w < x_w; ++w)
    {
      for (int32_t c = 0; c < x_c; ++c)
      {
        int32_t i = h * x_w * x_c + w * x_c + c;

        sum_1[c] += d_y[i] * gamma[c];
        sum_2[c] += d_y[i] * gamma[c] * ((x[i] - sample_mean[c]) / sample_std_dev[c]);
      }
    }
  }

  for (int32_t h = 0; h < x_h; ++h)
  {
    for (int32_t w = 0; w < x_w; ++w)
    {
      for (int32_t c = 0; c < x_c; ++c)
      {
        int32_t i = h * x_w * x_c + w * x_c + c;

        d_x[i] += (d_y[i] * gamma[c] - ((x[i] - sample_mean[c]) / sample_std_dev[c]) * sum_2[c] / num - sum_1[c] / num) / sample_std_dev[c];
      }
    }
  }
}

auto pointwise_convolution_forward(float const *in,
                                   float *out,
                                   float const *kernel,
                                   float const *bias,
                                   int32_t height,
                                   int32_t width,
                                   int32_t channels_in,
                                   int32_t channels_out) -> void
{
  for (int32_t h = 0; h < height; ++h)
  {
    for (int32_t w = 0; w < width; ++w)
    {
      for (int32_t c = 0; c < channels_out; ++c)
      {
        out[h * width * channels_out + w * channels_out + c] = bias[c];
      }
    }
  }

  matrix_multiply_accumulating(in, kernel, out, height * width, channels_in, channels_out);
  // matrix_multiply_accumulating_blocked(in, kernel, out, height * width, channels_in, channels_out);
}

auto matrix_multiply_accumulating_test_1(float const *a,
                                         float const *b,
                                         float *c,
                                         float *b_,
                                         int32_t m,
                                         int32_t k,
                                         int32_t n) -> void
{
  /*
  A: m x k
  B: k * n
  C: m x n
  */

  for (int32_t i_m = 0; i_m < m; ++i_m)
  {
    for (int32_t i_k = 0; i_k < k; ++i_k)
    {
      float temp = a[i_m * k + i_k];
      for (int32_t i_n = 0; i_n < n; ++i_n)
      {
        c[i_m * n + i_n] += temp * b[i_k * n + i_n];
      }
    }
  }
}

auto matrix_multiply_accumulating_test_2(float const *a,
                                         float const *b,
                                         float *c,
                                         float *b_,
                                         int32_t m,
                                         int32_t k,
                                         int32_t n) -> void
{
  /*
  A: m x k
  B: k * n
  C: m x n
  */

  for (int32_t i_m = 0; i_m < m; i_m += 4)
  {
    for (int32_t i_k = 0; i_k < k; ++i_k)
    {
      float a_0 = a[(i_m + 0) * k + i_k];
      float a_1 = a[(i_m + 1) * k + i_k];
      float a_2 = a[(i_m + 2) * k + i_k];
      float a_3 = a[(i_m + 3) * k + i_k];
      for (int32_t i_n = 0; i_n < n; ++i_n)
      {
        c[(i_m + 0) * n + i_n] += a_0 * b[i_k * n + i_n];
        c[(i_m + 1) * n + i_n] += a_1 * b[i_k * n + i_n];
        c[(i_m + 2) * n + i_n] += a_2 * b[i_k * n + i_n];
        c[(i_m + 3) * n + i_n] += a_3 * b[i_k * n + i_n];
      }
    }
  }
}

// GOOD!!!
auto matrix_multiply_accumulating_test_3(float const *a,
                                         float const *b,
                                         float *c,
                                         float *b_,
                                         int32_t m,
                                         int32_t k,
                                         int32_t n) -> void
{
  /*
  A: m x k
  B: k * n
  C: m x n
  */

  for (int32_t i_m = 0; i_m < m; ++i_m)
  {
    for (int32_t i_k = 0; i_k < k; i_k += 4)
    {
      float a_0 = a[i_m * k + (i_k + 0)];
      float a_1 = a[i_m * k + (i_k + 1)];
      float a_2 = a[i_m * k + (i_k + 2)];
      float a_3 = a[i_m * k + (i_k + 3)];
      for (int32_t i_n = 0; i_n < n; ++i_n)
      {
        c[i_m * n + i_n] += a_0 * b[(i_k + 0) * n + i_n];
        c[i_m * n + i_n] += a_1 * b[(i_k + 1) * n + i_n];
        c[i_m * n + i_n] += a_2 * b[(i_k + 2) * n + i_n];
        c[i_m * n + i_n] += a_3 * b[(i_k + 3) * n + i_n];
      }
    }
  }
}

auto matrix_multiply_accumulating_test_4(float const *a,
                                         float const *b,
                                         float *c,
                                         float *b_,
                                         int32_t m,
                                         int32_t k,
                                         int32_t n) -> void
{
  /*
  A: m x k
  B: k * n
  C: m x n
  */

  // for (int32_t i_m = 0; i_m < m; i_m += 4)
  // {
  //   for (int32_t i_k = 0; i_k < k; ++i_k)
  //   {
  //     float a_0 = a[(i_m + 0) * k + i_k];
  //     float a_1 = a[(i_m + 1) * k + i_k];
  //     float a_2 = a[(i_m + 2) * k + i_k];
  //     float a_3 = a[(i_m + 3) * k + i_k];
  //     for (int32_t i_n = 0; i_n < n; i_n += 4)
  //     {
  //       c[(i_m + 0) * n + (i_n + 0)] += a_0 * b[i_k * n + (i_n + 0)];
  //       c[(i_m + 1) * n + (i_n + 0)] += a_1 * b[i_k * n + (i_n + 0)];
  //       c[(i_m + 2) * n + (i_n + 0)] += a_2 * b[i_k * n + (i_n + 0)];
  //       c[(i_m + 3) * n + (i_n + 0)] += a_3 * b[i_k * n + (i_n + 0)];

  //       c[(i_m + 0) * n + (i_n + 1)] += a_0 * b[i_k * n + (i_n + 1)];
  //       c[(i_m + 1) * n + (i_n + 1)] += a_1 * b[i_k * n + (i_n + 1)];
  //       c[(i_m + 2) * n + (i_n + 1)] += a_2 * b[i_k * n + (i_n + 1)];
  //       c[(i_m + 3) * n + (i_n + 1)] += a_3 * b[i_k * n + (i_n + 1)];

  //       c[(i_m + 0) * n + (i_n + 2)] += a_0 * b[i_k * n + (i_n + 2)];
  //       c[(i_m + 1) * n + (i_n + 2)] += a_1 * b[i_k * n + (i_n + 2)];
  //       c[(i_m + 2) * n + (i_n + 2)] += a_2 * b[i_k * n + (i_n + 2)];
  //       c[(i_m + 3) * n + (i_n + 2)] += a_3 * b[i_k * n + (i_n + 2)];

  //       c[(i_m + 0) * n + (i_n + 3)] += a_0 * b[i_k * n + (i_n + 3)];
  //       c[(i_m + 1) * n + (i_n + 3)] += a_1 * b[i_k * n + (i_n + 3)];
  //       c[(i_m + 2) * n + (i_n + 3)] += a_2 * b[i_k * n + (i_n + 3)];
  //       c[(i_m + 3) * n + (i_n + 3)] += a_3 * b[i_k * n + (i_n + 3)];
  //     }
  //   }
  // }

  // for (int32_t i_m = 0; i_m < m; i_m += 4)
  // {
  //   for (int32_t i_n = 0; i_n < n; i_n += 4)
  //   {
  //     for (int32_t i_k = 0; i_k < k; ++i_k)
  //     {
  //       float a_0 = a[(i_m + 0) * k + i_k];
  //       float a_1 = a[(i_m + 1) * k + i_k];
  //       float a_2 = a[(i_m + 2) * k + i_k];
  //       float a_3 = a[(i_m + 3) * k + i_k];
  //       for (int32_t i_n_ = i_n; i_n_ < i_n + 4; ++i_n_)
  //       {
  //         c[(i_m + 0) * n + i_n_] += a_0 * b[i_k * n + i_n_];
  //         c[(i_m + 1) * n + i_n_] += a_1 * b[i_k * n + i_n_];
  //         c[(i_m + 2) * n + i_n_] += a_2 * b[i_k * n + i_n_];
  //         c[(i_m + 3) * n + i_n_] += a_3 * b[i_k * n + i_n_];
  //       }
  //     }
  //   }
  // }

  // for (int32_t i_k = 0; i_k < k; ++i_k)
  // {
  //   for (int32_t i_n = 0; i_n < n; ++i_n)
  //   {
  //     b_[i_n * k + i_k] = b[i_k * n + i_n];
  //   }
  // }

  // for (int32_t i_m = 0; i_m < m; ++i_m)
  // {
  //   for (int32_t i_k = 0; i_k < k; i_k += 4)
  //   {
  //     float a_0 = a[i_m * k + (i_k + 0)];
  //     float a_1 = a[i_m * k + (i_k + 1)];
  //     float a_2 = a[i_m * k + (i_k + 2)];
  //     float a_3 = a[i_m * k + (i_k + 3)];
  //     for (int32_t i_n = 0; i_n < n; ++i_n)
  //     {
  //       c[i_m * n + i_n] += a_0 * b_[i_n * k + (i_k + 0)];
  //       c[i_m * n + i_n] += a_1 * b_[i_n * k + (i_k + 1)];
  //       c[i_m * n + i_n] += a_2 * b_[i_n * k + (i_k + 2)];
  //       c[i_m * n + i_n] += a_3 * b_[i_n * k + (i_k + 3)];
  //     }
  //   }
  // }

  // for (int32_t i_m = 0; i_m < m; ++i_m)
  // {
  //   for (int32_t i_n = 0; i_n < n; ++i_n)
  //   {
  //     for (int32_t i_k = 0; i_k < k; i_k += 4)
  //     {
  //       c[i_m * n + i_n] += a[i_m * k + i_k] * b_[i_n * k + i_k];
  //     }
  //   }
  // }

  // for (int32_t i_m = 0; i_m < m; i_m += 4)
  // {
  //   for (int32_t i_k = 0; i_k < k; ++i_k)
  //   {
  //     float a_0 = a[(i_m + 0) * k + i_k];
  //     float a_1 = a[(i_m + 1) * k + i_k];
  //     float a_2 = a[(i_m + 2) * k + i_k];
  //     float a_3 = a[(i_m + 3) * k + i_k];
  //     for (int32_t i_n = 0; i_n < n; ++i_n)
  //     {
  //       c[(i_m + 0) * n + i_n] += a_0 * b[i_n * k + i_k];
  //       c[(i_m + 1) * n + i_n] += a_1 * b[i_n * k + i_k];
  //       c[(i_m + 2) * n + i_n] += a_2 * b[i_n * k + i_k];
  //       c[(i_m + 3) * n + i_n] += a_3 * b[i_n * k + i_k];
  //     }
  //   }
  // }
}

auto matrix_multiply_accumulating_test_5(float const *a,
                                         float const *b,
                                         float *c,
                                         float *b_,
                                         int32_t m,
                                         int32_t k,
                                         int32_t n) -> void
{
  /*
  A: m x k
  B: k * n
  C: m x n
  */

  int32_t i = 0;
  for (int32_t i_n = 0; i_n < n; i_n += 4)
  {
    for (int32_t i_k = 0; i_k < k; ++i_k)
    {
      b_[i + 0] = b[i_k * n + (i_n + 0)];
      b_[i + 1] = b[i_k * n + (i_n + 1)];
      b_[i + 2] = b[i_k * n + (i_n + 2)];
      b_[i + 3] = b[i_k * n + (i_n + 3)];
      i += 4;
    }
  }

  // for (int32_t i_m = 0; i_m < m; i_m += 4)
  // {
  //   int32_t j = 0;
  //   for (int32_t i_n = 0; i_n < n; i_n += 4)
  //   {
  //     for (int32_t i_k = 0; i_k < k; i_k += 4)
  //     {
  //       c[(i_m + 0) * n + (i_n + 0)] += a[(i_m + 0) * k + (i_k + 0)] * b_[j + 0];
  //       c[(i_m + 1) * n + (i_n + 0)] += a[(i_m + 1) * k + (i_k + 0)] * b_[j + 0];
  //       c[(i_m + 2) * n + (i_n + 0)] += a[(i_m + 2) * k + (i_k + 0)] * b_[j + 0];
  //       c[(i_m + 3) * n + (i_n + 0)] += a[(i_m + 3) * k + (i_k + 0)] * b_[j + 0];

  //       c[(i_m + 0) * n + (i_n + 1)] += a[(i_m + 0) * k + (i_k + 0)] * b_[j + 1];
  //       c[(i_m + 1) * n + (i_n + 1)] += a[(i_m + 1) * k + (i_k + 0)] * b_[j + 1];
  //       c[(i_m + 2) * n + (i_n + 1)] += a[(i_m + 2) * k + (i_k + 0)] * b_[j + 1];
  //       c[(i_m + 3) * n + (i_n + 1)] += a[(i_m + 3) * k + (i_k + 0)] * b_[j + 1];

  //       c[(i_m + 0) * n + (i_n + 2)] += a[(i_m + 0) * k + (i_k + 0)] * b_[j + 2];
  //       c[(i_m + 1) * n + (i_n + 2)] += a[(i_m + 1) * k + (i_k + 0)] * b_[j + 2];
  //       c[(i_m + 2) * n + (i_n + 2)] += a[(i_m + 2) * k + (i_k + 0)] * b_[j + 2];
  //       c[(i_m + 3) * n + (i_n + 2)] += a[(i_m + 3) * k + (i_k + 0)] * b_[j + 2];

  //       c[(i_m + 0) * n + (i_n + 3)] += a[(i_m + 0) * k + (i_k + 0)] * b_[j + 3];
  //       c[(i_m + 1) * n + (i_n + 3)] += a[(i_m + 1) * k + (i_k + 0)] * b_[j + 3];
  //       c[(i_m + 2) * n + (i_n + 3)] += a[(i_m + 2) * k + (i_k + 0)] * b_[j + 3];
  //       c[(i_m + 3) * n + (i_n + 3)] += a[(i_m + 3) * k + (i_k + 0)] * b_[j + 3];

  //       //

  //       c[(i_m + 0) * n + (i_n + 0)] += a[(i_m + 0) * k + (i_k + 1)] * b_[j + 4];
  //       c[(i_m + 1) * n + (i_n + 0)] += a[(i_m + 1) * k + (i_k + 1)] * b_[j + 4];
  //       c[(i_m + 2) * n + (i_n + 0)] += a[(i_m + 2) * k + (i_k + 1)] * b_[j + 4];
  //       c[(i_m + 3) * n + (i_n + 0)] += a[(i_m + 3) * k + (i_k + 1)] * b_[j + 4];

  //       c[(i_m + 0) * n + (i_n + 1)] += a[(i_m + 0) * k + (i_k + 1)] * b_[j + 5];
  //       c[(i_m + 1) * n + (i_n + 1)] += a[(i_m + 1) * k + (i_k + 1)] * b_[j + 5];
  //       c[(i_m + 2) * n + (i_n + 1)] += a[(i_m + 2) * k + (i_k + 1)] * b_[j + 5];
  //       c[(i_m + 3) * n + (i_n + 1)] += a[(i_m + 3) * k + (i_k + 1)] * b_[j + 5];

  //       c[(i_m + 0) * n + (i_n + 2)] += a[(i_m + 0) * k + (i_k + 1)] * b_[j + 6];
  //       c[(i_m + 1) * n + (i_n + 2)] += a[(i_m + 1) * k + (i_k + 1)] * b_[j + 6];
  //       c[(i_m + 2) * n + (i_n + 2)] += a[(i_m + 2) * k + (i_k + 1)] * b_[j + 6];
  //       c[(i_m + 3) * n + (i_n + 2)] += a[(i_m + 3) * k + (i_k + 1)] * b_[j + 6];

  //       c[(i_m + 0) * n + (i_n + 3)] += a[(i_m + 0) * k + (i_k + 1)] * b_[j + 7];
  //       c[(i_m + 1) * n + (i_n + 3)] += a[(i_m + 1) * k + (i_k + 1)] * b_[j + 7];
  //       c[(i_m + 2) * n + (i_n + 3)] += a[(i_m + 2) * k + (i_k + 1)] * b_[j + 7];
  //       c[(i_m + 3) * n + (i_n + 3)] += a[(i_m + 3) * k + (i_k + 1)] * b_[j + 7];

  //       //

  //       c[(i_m + 0) * n + (i_n + 0)] += a[(i_m + 0) * k + (i_k + 2)] * b_[j + 8];
  //       c[(i_m + 1) * n + (i_n + 0)] += a[(i_m + 1) * k + (i_k + 2)] * b_[j + 8];
  //       c[(i_m + 2) * n + (i_n + 0)] += a[(i_m + 2) * k + (i_k + 2)] * b_[j + 8];
  //       c[(i_m + 3) * n + (i_n + 0)] += a[(i_m + 3) * k + (i_k + 2)] * b_[j + 8];

  //       c[(i_m + 0) * n + (i_n + 1)] += a[(i_m + 0) * k + (i_k + 2)] * b_[j + 9];
  //       c[(i_m + 1) * n + (i_n + 1)] += a[(i_m + 1) * k + (i_k + 2)] * b_[j + 9];
  //       c[(i_m + 2) * n + (i_n + 1)] += a[(i_m + 2) * k + (i_k + 2)] * b_[j + 9];
  //       c[(i_m + 3) * n + (i_n + 1)] += a[(i_m + 3) * k + (i_k + 2)] * b_[j + 9];

  //       c[(i_m + 0) * n + (i_n + 2)] += a[(i_m + 0) * k + (i_k + 2)] * b_[j + 10];
  //       c[(i_m + 1) * n + (i_n + 2)] += a[(i_m + 1) * k + (i_k + 2)] * b_[j + 10];
  //       c[(i_m + 2) * n + (i_n + 2)] += a[(i_m + 2) * k + (i_k + 2)] * b_[j + 10];
  //       c[(i_m + 3) * n + (i_n + 2)] += a[(i_m + 3) * k + (i_k + 2)] * b_[j + 10];

  //       c[(i_m + 0) * n + (i_n + 3)] += a[(i_m + 0) * k + (i_k + 2)] * b_[j + 11];
  //       c[(i_m + 1) * n + (i_n + 3)] += a[(i_m + 1) * k + (i_k + 2)] * b_[j + 11];
  //       c[(i_m + 2) * n + (i_n + 3)] += a[(i_m + 2) * k + (i_k + 2)] * b_[j + 11];
  //       c[(i_m + 3) * n + (i_n + 3)] += a[(i_m + 3) * k + (i_k + 2)] * b_[j + 11];

  //       //

  //       c[(i_m + 0) * n + (i_n + 0)] += a[(i_m + 0) * k + (i_k + 3)] * b_[j + 12];
  //       c[(i_m + 1) * n + (i_n + 0)] += a[(i_m + 1) * k + (i_k + 3)] * b_[j + 12];
  //       c[(i_m + 2) * n + (i_n + 0)] += a[(i_m + 2) * k + (i_k + 3)] * b_[j + 12];
  //       c[(i_m + 3) * n + (i_n + 0)] += a[(i_m + 3) * k + (i_k + 3)] * b_[j + 12];

  //       c[(i_m + 0) * n + (i_n + 1)] += a[(i_m + 0) * k + (i_k + 3)] * b_[j + 13];
  //       c[(i_m + 1) * n + (i_n + 1)] += a[(i_m + 1) * k + (i_k + 3)] * b_[j + 13];
  //       c[(i_m + 2) * n + (i_n + 1)] += a[(i_m + 2) * k + (i_k + 3)] * b_[j + 13];
  //       c[(i_m + 3) * n + (i_n + 1)] += a[(i_m + 3) * k + (i_k + 3)] * b_[j + 13];

  //       c[(i_m + 0) * n + (i_n + 2)] += a[(i_m + 0) * k + (i_k + 3)] * b_[j + 14];
  //       c[(i_m + 1) * n + (i_n + 2)] += a[(i_m + 1) * k + (i_k + 3)] * b_[j + 14];
  //       c[(i_m + 2) * n + (i_n + 2)] += a[(i_m + 2) * k + (i_k + 3)] * b_[j + 14];
  //       c[(i_m + 3) * n + (i_n + 2)] += a[(i_m + 3) * k + (i_k + 3)] * b_[j + 14];

  //       c[(i_m + 0) * n + (i_n + 3)] += a[(i_m + 0) * k + (i_k + 3)] * b_[j + 15];
  //       c[(i_m + 1) * n + (i_n + 3)] += a[(i_m + 1) * k + (i_k + 3)] * b_[j + 15];
  //       c[(i_m + 2) * n + (i_n + 3)] += a[(i_m + 2) * k + (i_k + 3)] * b_[j + 15];
  //       c[(i_m + 3) * n + (i_n + 3)] += a[(i_m + 3) * k + (i_k + 3)] * b_[j + 15];

  //       j += 16;
  //     }
  //   }
  // }

  // for (int32_t i_m = 0; i_m < m; i_m += 4)
  // {
  //   int32_t j = 0;
  //   for (int32_t i_k = 0; i_k < k; i_k += 4)
  //   {
  //     for (int32_t i_n = 0; i_n < n; i_n += 4)
  //     {
  //       c[(i_m + 0) * n + (i_n + 0)] += a[(i_m + 0) * k + (i_k + 0)] * b_[j + 0];
  //       c[(i_m + 1) * n + (i_n + 0)] += a[(i_m + 1) * k + (i_k + 0)] * b_[j + 0];
  //       c[(i_m + 2) * n + (i_n + 0)] += a[(i_m + 2) * k + (i_k + 0)] * b_[j + 0];
  //       c[(i_m + 3) * n + (i_n + 0)] += a[(i_m + 3) * k + (i_k + 0)] * b_[j + 0];

  //       c[(i_m + 0) * n + (i_n + 1)] += a[(i_m + 0) * k + (i_k + 0)] * b_[j + 1];
  //       c[(i_m + 1) * n + (i_n + 1)] += a[(i_m + 1) * k + (i_k + 0)] * b_[j + 1];
  //       c[(i_m + 2) * n + (i_n + 1)] += a[(i_m + 2) * k + (i_k + 0)] * b_[j + 1];
  //       c[(i_m + 3) * n + (i_n + 1)] += a[(i_m + 3) * k + (i_k + 0)] * b_[j + 1];

  //       c[(i_m + 0) * n + (i_n + 2)] += a[(i_m + 0) * k + (i_k + 0)] * b_[j + 2];
  //       c[(i_m + 1) * n + (i_n + 2)] += a[(i_m + 1) * k + (i_k + 0)] * b_[j + 2];
  //       c[(i_m + 2) * n + (i_n + 2)] += a[(i_m + 2) * k + (i_k + 0)] * b_[j + 2];
  //       c[(i_m + 3) * n + (i_n + 2)] += a[(i_m + 3) * k + (i_k + 0)] * b_[j + 2];

  //       c[(i_m + 0) * n + (i_n + 3)] += a[(i_m + 0) * k + (i_k + 0)] * b_[j + 3];
  //       c[(i_m + 1) * n + (i_n + 3)] += a[(i_m + 1) * k + (i_k + 0)] * b_[j + 3];
  //       c[(i_m + 2) * n + (i_n + 3)] += a[(i_m + 2) * k + (i_k + 0)] * b_[j + 3];
  //       c[(i_m + 3) * n + (i_n + 3)] += a[(i_m + 3) * k + (i_k + 0)] * b_[j + 3];

  //       //

  //       c[(i_m + 0) * n + (i_n + 0)] += a[(i_m + 0) * k + (i_k + 1)] * b_[j + 4];
  //       c[(i_m + 1) * n + (i_n + 0)] += a[(i_m + 1) * k + (i_k + 1)] * b_[j + 4];
  //       c[(i_m + 2) * n + (i_n + 0)] += a[(i_m + 2) * k + (i_k + 1)] * b_[j + 4];
  //       c[(i_m + 3) * n + (i_n + 0)] += a[(i_m + 3) * k + (i_k + 1)] * b_[j + 4];

  //       c[(i_m + 0) * n + (i_n + 1)] += a[(i_m + 0) * k + (i_k + 1)] * b_[j + 5];
  //       c[(i_m + 1) * n + (i_n + 1)] += a[(i_m + 1) * k + (i_k + 1)] * b_[j + 5];
  //       c[(i_m + 2) * n + (i_n + 1)] += a[(i_m + 2) * k + (i_k + 1)] * b_[j + 5];
  //       c[(i_m + 3) * n + (i_n + 1)] += a[(i_m + 3) * k + (i_k + 1)] * b_[j + 5];

  //       c[(i_m + 0) * n + (i_n + 2)] += a[(i_m + 0) * k + (i_k + 1)] * b_[j + 6];
  //       c[(i_m + 1) * n + (i_n + 2)] += a[(i_m + 1) * k + (i_k + 1)] * b_[j + 6];
  //       c[(i_m + 2) * n + (i_n + 2)] += a[(i_m + 2) * k + (i_k + 1)] * b_[j + 6];
  //       c[(i_m + 3) * n + (i_n + 2)] += a[(i_m + 3) * k + (i_k + 1)] * b_[j + 6];

  //       c[(i_m + 0) * n + (i_n + 3)] += a[(i_m + 0) * k + (i_k + 1)] * b_[j + 7];
  //       c[(i_m + 1) * n + (i_n + 3)] += a[(i_m + 1) * k + (i_k + 1)] * b_[j + 7];
  //       c[(i_m + 2) * n + (i_n + 3)] += a[(i_m + 2) * k + (i_k + 1)] * b_[j + 7];
  //       c[(i_m + 3) * n + (i_n + 3)] += a[(i_m + 3) * k + (i_k + 1)] * b_[j + 7];

  //       //

  //       c[(i_m + 0) * n + (i_n + 0)] += a[(i_m + 0) * k + (i_k + 2)] * b_[j + 8];
  //       c[(i_m + 1) * n + (i_n + 0)] += a[(i_m + 1) * k + (i_k + 2)] * b_[j + 8];
  //       c[(i_m + 2) * n + (i_n + 0)] += a[(i_m + 2) * k + (i_k + 2)] * b_[j + 8];
  //       c[(i_m + 3) * n + (i_n + 0)] += a[(i_m + 3) * k + (i_k + 2)] * b_[j + 8];

  //       c[(i_m + 0) * n + (i_n + 1)] += a[(i_m + 0) * k + (i_k + 2)] * b_[j + 9];
  //       c[(i_m + 1) * n + (i_n + 1)] += a[(i_m + 1) * k + (i_k + 2)] * b_[j + 9];
  //       c[(i_m + 2) * n + (i_n + 1)] += a[(i_m + 2) * k + (i_k + 2)] * b_[j + 9];
  //       c[(i_m + 3) * n + (i_n + 1)] += a[(i_m + 3) * k + (i_k + 2)] * b_[j + 9];

  //       c[(i_m + 0) * n + (i_n + 2)] += a[(i_m + 0) * k + (i_k + 2)] * b_[j + 10];
  //       c[(i_m + 1) * n + (i_n + 2)] += a[(i_m + 1) * k + (i_k + 2)] * b_[j + 10];
  //       c[(i_m + 2) * n + (i_n + 2)] += a[(i_m + 2) * k + (i_k + 2)] * b_[j + 10];
  //       c[(i_m + 3) * n + (i_n + 2)] += a[(i_m + 3) * k + (i_k + 2)] * b_[j + 10];

  //       c[(i_m + 0) * n + (i_n + 3)] += a[(i_m + 0) * k + (i_k + 2)] * b_[j + 11];
  //       c[(i_m + 1) * n + (i_n + 3)] += a[(i_m + 1) * k + (i_k + 2)] * b_[j + 11];
  //       c[(i_m + 2) * n + (i_n + 3)] += a[(i_m + 2) * k + (i_k + 2)] * b_[j + 11];
  //       c[(i_m + 3) * n + (i_n + 3)] += a[(i_m + 3) * k + (i_k + 2)] * b_[j + 11];

  //       //

  //       c[(i_m + 0) * n + (i_n + 0)] += a[(i_m + 0) * k + (i_k + 3)] * b_[j + 12];
  //       c[(i_m + 1) * n + (i_n + 0)] += a[(i_m + 1) * k + (i_k + 3)] * b_[j + 12];
  //       c[(i_m + 2) * n + (i_n + 0)] += a[(i_m + 2) * k + (i_k + 3)] * b_[j + 12];
  //       c[(i_m + 3) * n + (i_n + 0)] += a[(i_m + 3) * k + (i_k + 3)] * b_[j + 12];

  //       c[(i_m + 0) * n + (i_n + 1)] += a[(i_m + 0) * k + (i_k + 3)] * b_[j + 13];
  //       c[(i_m + 1) * n + (i_n + 1)] += a[(i_m + 1) * k + (i_k + 3)] * b_[j + 13];
  //       c[(i_m + 2) * n + (i_n + 1)] += a[(i_m + 2) * k + (i_k + 3)] * b_[j + 13];
  //       c[(i_m + 3) * n + (i_n + 1)] += a[(i_m + 3) * k + (i_k + 3)] * b_[j + 13];

  //       c[(i_m + 0) * n + (i_n + 2)] += a[(i_m + 0) * k + (i_k + 3)] * b_[j + 14];
  //       c[(i_m + 1) * n + (i_n + 2)] += a[(i_m + 1) * k + (i_k + 3)] * b_[j + 14];
  //       c[(i_m + 2) * n + (i_n + 2)] += a[(i_m + 2) * k + (i_k + 3)] * b_[j + 14];
  //       c[(i_m + 3) * n + (i_n + 2)] += a[(i_m + 3) * k + (i_k + 3)] * b_[j + 14];

  //       c[(i_m + 0) * n + (i_n + 3)] += a[(i_m + 0) * k + (i_k + 3)] * b_[j + 15];
  //       c[(i_m + 1) * n + (i_n + 3)] += a[(i_m + 1) * k + (i_k + 3)] * b_[j + 15];
  //       c[(i_m + 2) * n + (i_n + 3)] += a[(i_m + 2) * k + (i_k + 3)] * b_[j + 15];
  //       c[(i_m + 3) * n + (i_n + 3)] += a[(i_m + 3) * k + (i_k + 3)] * b_[j + 15];

  //       j += 16;
  //     }
  //   }
  // }

  for (int32_t i_m = 0; i_m < m; i_m += 4)
  {
    int32_t j = -1;
    for (int32_t i_k = 0; i_k < k; i_k += 4)
    {
      float a_0 = a[(i_m + 0) * k + (i_k + 0)];
      float a_1 = a[(i_m + 1) * k + (i_k + 0)];
      float a_2 = a[(i_m + 2) * k + (i_k + 0)];
      float a_3 = a[(i_m + 3) * k + (i_k + 0)];

      float a_4 = a[(i_m + 0) * k + (i_k + 1)];
      float a_5 = a[(i_m + 1) * k + (i_k + 1)];
      float a_6 = a[(i_m + 2) * k + (i_k + 1)];
      float a_7 = a[(i_m + 3) * k + (i_k + 1)];

      float a_8 = a[(i_m + 0) * k + (i_k + 2)];
      float a_9 = a[(i_m + 1) * k + (i_k + 2)];
      float a_10 = a[(i_m + 2) * k + (i_k + 2)];
      float a_11 = a[(i_m + 3) * k + (i_k + 2)];

      float a_12 = a[(i_m + 0) * k + (i_k + 3)];
      float a_13 = a[(i_m + 1) * k + (i_k + 3)];
      float a_14 = a[(i_m + 2) * k + (i_k + 3)];
      float a_15 = a[(i_m + 3) * k + (i_k + 3)];

      for (int32_t i_n = 0; i_n < n; i_n += 4)
      {
        c[(i_m + 0) * n + (i_n + 0)] += a_0 * b_[++j];
        c[(i_m + 1) * n + (i_n + 0)] += a_1 * b_[j];
        c[(i_m + 2) * n + (i_n + 0)] += a_2 * b_[j];
        c[(i_m + 3) * n + (i_n + 0)] += a_3 * b_[j];

        c[(i_m + 0) * n + (i_n + 1)] += a_0 * b_[++j];
        c[(i_m + 1) * n + (i_n + 1)] += a_1 * b_[j];
        c[(i_m + 2) * n + (i_n + 1)] += a_2 * b_[j];
        c[(i_m + 3) * n + (i_n + 1)] += a_3 * b_[j];

        c[(i_m + 0) * n + (i_n + 2)] += a_0 * b_[++j];
        c[(i_m + 1) * n + (i_n + 2)] += a_1 * b_[j];
        c[(i_m + 2) * n + (i_n + 2)] += a_2 * b_[j];
        c[(i_m + 3) * n + (i_n + 2)] += a_3 * b_[j];

        c[(i_m + 0) * n + (i_n + 3)] += a_0 * b_[++j];
        c[(i_m + 1) * n + (i_n + 3)] += a_1 * b_[j];
        c[(i_m + 2) * n + (i_n + 3)] += a_2 * b_[j];
        c[(i_m + 3) * n + (i_n + 3)] += a_3 * b_[j];

        //

        c[(i_m + 0) * n + (i_n + 0)] += a_4 * b_[++j];
        c[(i_m + 1) * n + (i_n + 0)] += a_5 * b_[j];
        c[(i_m + 2) * n + (i_n + 0)] += a_6 * b_[j];
        c[(i_m + 3) * n + (i_n + 0)] += a_7 * b_[j];

        c[(i_m + 0) * n + (i_n + 1)] += a_4 * b_[++j];
        c[(i_m + 1) * n + (i_n + 1)] += a_5 * b_[j];
        c[(i_m + 2) * n + (i_n + 1)] += a_6 * b_[j];
        c[(i_m + 3) * n + (i_n + 1)] += a_7 * b_[j];

        c[(i_m + 0) * n + (i_n + 2)] += a_4 * b_[++j];
        c[(i_m + 1) * n + (i_n + 2)] += a_5 * b_[j];
        c[(i_m + 2) * n + (i_n + 2)] += a_6 * b_[j];
        c[(i_m + 3) * n + (i_n + 2)] += a_7 * b_[j];

        c[(i_m + 0) * n + (i_n + 3)] += a_4 * b_[++j];
        c[(i_m + 1) * n + (i_n + 3)] += a_5 * b_[j];
        c[(i_m + 2) * n + (i_n + 3)] += a_6 * b_[j];
        c[(i_m + 3) * n + (i_n + 3)] += a_7 * b_[j];

        //

        c[(i_m + 0) * n + (i_n + 0)] += a_8 * b_[++j];
        c[(i_m + 1) * n + (i_n + 0)] += a_9 * b_[j];
        c[(i_m + 2) * n + (i_n + 0)] += a_10 * b_[j];
        c[(i_m + 3) * n + (i_n + 0)] += a_11 * b_[j];

        c[(i_m + 0) * n + (i_n + 1)] += a_8 * b_[++j];
        c[(i_m + 1) * n + (i_n + 1)] += a_9 * b_[j];
        c[(i_m + 2) * n + (i_n + 1)] += a_10 * b_[j];
        c[(i_m + 3) * n + (i_n + 1)] += a_11 * b_[j];

        c[(i_m + 0) * n + (i_n + 2)] += a_8 * b_[++j];
        c[(i_m + 1) * n + (i_n + 2)] += a_9 * b_[j];
        c[(i_m + 2) * n + (i_n + 2)] += a_10 * b_[j];
        c[(i_m + 3) * n + (i_n + 2)] += a_11 * b_[j];

        c[(i_m + 0) * n + (i_n + 3)] += a_8 * b_[++j];
        c[(i_m + 1) * n + (i_n + 3)] += a_9 * b_[j];
        c[(i_m + 2) * n + (i_n + 3)] += a_10 * b_[j];
        c[(i_m + 3) * n + (i_n + 3)] += a_11 * b_[j];

        //

        c[(i_m + 0) * n + (i_n + 0)] += a_12 * b_[++j];
        c[(i_m + 1) * n + (i_n + 0)] += a_13 * b_[j];
        c[(i_m + 2) * n + (i_n + 0)] += a_14 * b_[j];
        c[(i_m + 3) * n + (i_n + 0)] += a_15 * b_[j];

        c[(i_m + 0) * n + (i_n + 1)] += a_12 * b_[++j];
        c[(i_m + 1) * n + (i_n + 1)] += a_13 * b_[j];
        c[(i_m + 2) * n + (i_n + 1)] += a_14 * b_[j];
        c[(i_m + 3) * n + (i_n + 1)] += a_15 * b_[j];

        c[(i_m + 0) * n + (i_n + 2)] += a_12 * b_[++j];
        c[(i_m + 1) * n + (i_n + 2)] += a_13 * b_[j];
        c[(i_m + 2) * n + (i_n + 2)] += a_14 * b_[j];
        c[(i_m + 3) * n + (i_n + 2)] += a_15 * b_[j];

        c[(i_m + 0) * n + (i_n + 3)] += a_12 * b_[++j];
        c[(i_m + 1) * n + (i_n + 3)] += a_13 * b_[j];
        c[(i_m + 2) * n + (i_n + 3)] += a_14 * b_[j];
        c[(i_m + 3) * n + (i_n + 3)] += a_15 * b_[j];

        // j += 16;
      }
    }
  }
}

auto pointwise_convolution_forward_test_1(float const *in,
                                          float *out,
                                          float const *kernel,
                                          float const *bias,
                                          float *kernel_,
                                          int32_t height,
                                          int32_t width,
                                          int32_t channels_in,
                                          int32_t channels_out) -> void
{
  for (int32_t h = 0; h < height; ++h)
  {
    for (int32_t w = 0; w < width; ++w)
    {
      for (int32_t c = 0; c < channels_out; ++c)
      {
        out[h * width * channels_out + w * channels_out + c] = bias[c];
      }
    }
  }

  matrix_multiply_accumulating_test_1(in, kernel, out, kernel_, height * width, channels_in, channels_out);
}

auto pointwise_convolution_forward_test_2(float const *in,
                                          float *out,
                                          float const *kernel,
                                          float const *bias,
                                          float *kernel_,
                                          int32_t height,
                                          int32_t width,
                                          int32_t channels_in,
                                          int32_t channels_out) -> void
{
  for (int32_t h = 0; h < height; ++h)
  {
    for (int32_t w = 0; w < width; ++w)
    {
      for (int32_t c = 0; c < channels_out; ++c)
      {
        out[h * width * channels_out + w * channels_out + c] = bias[c];
      }
    }
  }

  matrix_multiply_accumulating_test_2(in, kernel, out, kernel_, height * width, channels_in, channels_out);
}

auto pointwise_convolution_forward_test_3(float const *in,
                                          float *out,
                                          float const *kernel,
                                          float const *bias,
                                          float *kernel_,
                                          int32_t height,
                                          int32_t width,
                                          int32_t channels_in,
                                          int32_t channels_out) -> void
{
  for (int32_t h = 0; h < height; ++h)
  {
    for (int32_t w = 0; w < width; ++w)
    {
      for (int32_t c = 0; c < channels_out; ++c)
      {
        out[h * width * channels_out + w * channels_out + c] = bias[c];
      }
    }
  }

  matrix_multiply_accumulating_test_3(in, kernel, out, kernel_, height * width, channels_in, channels_out);
}

auto pointwise_convolution_forward_test_4(float const *in,
                                          float *out,
                                          float const *kernel,
                                          float const *bias,
                                          float *kernel_,
                                          int32_t height,
                                          int32_t width,
                                          int32_t channels_in,
                                          int32_t channels_out) -> void
{
  for (int32_t h = 0; h < height; ++h)
  {
    for (int32_t w = 0; w < width; ++w)
    {
      for (int32_t c = 0; c < channels_out; ++c)
      {
        out[h * width * channels_out + w * channels_out + c] = bias[c];
      }
    }
  }

  matrix_multiply_accumulating_test_4(in, kernel, out, kernel_, height * width, channels_in, channels_out);
}

auto pointwise_convolution_forward_test_5(float const *in,
                                          float *out,
                                          float const *kernel,
                                          float const *bias,
                                          float *kernel_,
                                          int32_t height,
                                          int32_t width,
                                          int32_t channels_in,
                                          int32_t channels_out) -> void
{
  for (int32_t h = 0; h < height; ++h)
  {
    for (int32_t w = 0; w < width; ++w)
    {
      for (int32_t c = 0; c < channels_out; ++c)
      {
        out[h * width * channels_out + w * channels_out + c] = bias[c];
      }
    }
  }

  matrix_multiply_accumulating_test_5(in, kernel, out, kernel_, height * width, channels_in, channels_out);
}

// auto matrix_multiply_accumulating_transpose_a(float const * a,
//                                               float const * b,
//                                               float * c,
//                                               int32_t m,
//                                               int32_t k,
//                                               int32_t n) -> void;
// auto matrix_multiply_accumulating_transpose_b(float const * a,
//                                               float const * b,
//                                               float * c,
//                                               int32_t m,
//                                               int32_t k,
//                                               int32_t n) -> void;

auto matrix_multiply_accumulating_transpose_a_test_1(float const *a,
                                                     float const *b,
                                                     float *c,
                                                     int32_t m,
                                                     int32_t k,
                                                     int32_t n) -> void
{
  for (int32_t i_m = 0; i_m < m; ++i_m)
  {
    for (int32_t i_k = 0; i_k < k; ++i_k)
    {
      float temp = a[i_k * m + i_m];
      for (int32_t i_n = 0; i_n < n; ++i_n)
      {
        c[i_m * n + i_n] += temp * b[i_k * n + i_n];
      }
    }
  }
}

// GOOD!!!
auto matrix_multiply_accumulating_transpose_a_test_2(float const *a,
                                                     float const *b,
                                                     float *c,
                                                     int32_t m,
                                                     int32_t k,
                                                     int32_t n) -> void
{
  for (int32_t i_k = 0; i_k < k; i_k += 4)
  {
    for (int32_t i_m = 0; i_m < m; ++i_m)
    {
      float a_0 = a[(i_k + 0) * m + i_m];
      float a_1 = a[(i_k + 1) * m + i_m];
      float a_2 = a[(i_k + 2) * m + i_m];
      float a_3 = a[(i_k + 3) * m + i_m];
      for (int32_t i_n = 0; i_n < n; ++i_n)
      {
        c[i_m * n + i_n] += a_0 * b[(i_k + 0) * n + i_n];
        c[i_m * n + i_n] += a_1 * b[(i_k + 1) * n + i_n];
        c[i_m * n + i_n] += a_2 * b[(i_k + 2) * n + i_n];
        c[i_m * n + i_n] += a_3 * b[(i_k + 3) * n + i_n];
      }
    }
  }
}

//

auto matrix_multiply_accumulating_transpose_b_test_1(float const *a,
                                                     float const *b,
                                                     float *c,
                                                     int32_t m,
                                                     int32_t k,
                                                     int32_t n) -> void
{
  for (int32_t i_m = 0; i_m < m; ++i_m)
  {
    for (int32_t i_k = 0; i_k < k; ++i_k)
    {
      float temp = a[i_m * k + i_k];
      for (int32_t i_n = 0; i_n < n; ++i_n)
      {
        c[i_m * n + i_n] += temp * b[i_n * k + i_k];
      }
    }
  }
}

auto matrix_multiply_accumulating_transpose_b_alt_test_1(float const *a,
                                                         float const *b,
                                                         float *c,
                                                         float *d,
                                                         int32_t m,
                                                         int32_t k,
                                                         int32_t n) -> void
{
  for (int32_t i_k = 0; i_k < k; ++i_k)
  {
    for (int32_t i_n = 0; i_n < n; ++i_n)
    {
      d[i_k * n + i_n] = b[i_n * k + i_k];
    }
  }

  for (int32_t i_m = 0; i_m < m; ++i_m)
  {
    for (int32_t i_k = 0; i_k < k; ++i_k)
    {
      float temp = a[i_m * k + i_k];
      for (int32_t i_n = 0; i_n < n; ++i_n)
      {
        c[i_m * n + i_n] += temp * d[i_k * n + i_n];
      }
    }
  }
}

auto matrix_multiply_accumulating_transpose_b_test_2(float const *a,
                                                     float const *b,
                                                     float *c,
                                                     //  float * d,
                                                     int32_t m,
                                                     int32_t k,
                                                     int32_t n) -> void
{
  // for (int32_t i_m = 0; i_m < m; ++i_m)
  // {
  //   for (int32_t i_k = 0; i_k < k; i_k += 4)
  //   {
  //     float a_0 = a[i_m * k + (i_k + 0)];
  //     float a_1 = a[i_m * k + (i_k + 1)];
  //     float a_2 = a[i_m * k + (i_k + 2)];
  //     float a_3 = a[i_m * k + (i_k + 3)];
  //     for (int32_t i_n = 0; i_n < n; ++i_n)
  //     {
  //       c[i_m * n + i_n] += a_0 * b[i_n * k + (i_k + 0)];
  //       c[i_m * n + i_n] += a_1 * b[i_n * k + (i_k + 1)];
  //       c[i_m * n + i_n] += a_2 * b[i_n * k + (i_k + 2)];
  //       c[i_m * n + i_n] += a_3 * b[i_n * k + (i_k + 3)];
  //     }
  //   }
  // }

  // for (int32_t i_m = 0; i_m < m; ++i_m)
  // {
  //   // float a_0 = a[i_m * k + (i_k + 0)];
  //   // float a_1 = a[i_m * k + (i_k + 1)];
  //   // float a_2 = a[i_m * k + (i_k + 2)];
  //   // float a_3 = a[i_m * k + (i_k + 3)];
  //   for (int32_t i_n = 0; i_n < n; ++i_n)
  //   {
  //     for (int32_t i_k = 0; i_k < k; i_k += 4)
  //     {
  //       c[i_m * n + i_n] += a[i_m * k + (i_k + 0)] * b[i_n * k + (i_k + 0)];
  //       c[i_m * n + i_n] += a[i_m * k + (i_k + 1)] * b[i_n * k + (i_k + 1)];
  //       c[i_m * n + i_n] += a[i_m * k + (i_k + 2)] * b[i_n * k + (i_k + 2)];
  //       c[i_m * n + i_n] += a[i_m * k + (i_k + 3)] * b[i_n * k + (i_k + 3)];
  //     }
  //   }
  // }

  for (int32_t i_m = 0; i_m < m; i_m += 4)
  {
    for (int32_t i_k = 0; i_k < k; ++i_k)
    {
      float a_0 = a[(i_m + 0) * k + i_k];
      float a_1 = a[(i_m + 1) * k + i_k];
      float a_2 = a[(i_m + 2) * k + i_k];
      float a_3 = a[(i_m + 3) * k + i_k];
      for (int32_t i_n = 0; i_n < n; ++i_n)
      {
        c[(i_m + 0) * n + i_n] += a_0 * b[i_n * k + i_k];
        c[(i_m + 1) * n + i_n] += a_1 * b[i_n * k + i_k];
        c[(i_m + 2) * n + i_n] += a_2 * b[i_n * k + i_k];
        c[(i_m + 3) * n + i_n] += a_3 * b[i_n * k + i_k];
      }
    }
  }
}

auto matrix_multiply_accumulating_transpose_b_alt_test_2(float const *a,
                                                         float const *b,
                                                         float *c,
                                                         float *d,
                                                         int32_t m,
                                                         int32_t k,
                                                         int32_t n) -> void
{
  for (int32_t i_k = 0; i_k < k; ++i_k)
  {
    for (int32_t i_n = 0; i_n < n; ++i_n)
    {
      d[i_k * n + i_n] = b[i_n * k + i_k];
    }
  }

  for (int32_t i_m = 0; i_m < m; ++i_m)
  {
    for (int32_t i_k = 0; i_k < k; i_k += 4)
    {
      float a_0 = a[i_m * k + (i_k + 0)];
      float a_1 = a[i_m * k + (i_k + 1)];
      float a_2 = a[i_m * k + (i_k + 2)];
      float a_3 = a[i_m * k + (i_k + 3)];
      for (int32_t i_n = 0; i_n < n; ++i_n)
      {
        c[i_m * n + i_n] += a_0 * d[(i_k + 0) * n + i_n];
        c[i_m * n + i_n] += a_1 * d[(i_k + 1) * n + i_n];
        c[i_m * n + i_n] += a_2 * d[(i_k + 2) * n + i_n];
        c[i_m * n + i_n] += a_3 * d[(i_k + 3) * n + i_n];
      }
    }
  }
}

auto matrix_multiply_accumulating_transpose_b_alt_test_3(float const *a,
                                                         float const *b,
                                                         float *c,
                                                         float *d,
                                                         int32_t m,
                                                         int32_t k,
                                                         int32_t n) -> void
{
  // for (int32_t i_m = 0; i_m < m; ++i_m)
  // {
  //   for (int32_t i_n = 0; i_n < n; ++i_n)
  //   {
  //     for (int32_t i_k = 0; i_k < k; i_k += 4)
  //     {
  //       c[i_m * n + i_n] += a[i_m * k + i_k] * b[i_n * k + i_k];
  //     }
  //   }
  // }

  // for (int32_t i_m = 0; i_m < m; ++i_m)
  // {
  //   for (int32_t i_n = 0; i_n < n; ++i_n)
  //   {
  //     for (int32_t i_k = 0; i_k < k; ++i_k)
  //     {
  //       c[i_m * n + i_n] += a[i_m * k + i_k] * b[i_n * k + i_k];
  //     }
  //   }
  // }

  // for (int32_t i_k = 0; i_k < k; ++i_k)
  // {
  //   for (int32_t i_m = 0; i_m < m; i_m += 4)
  //   {
  //     for (int32_t i_n = 0; i_n < n; i_n += 4)
  //     {
  //       c[i_m * n + i_n] += a[(i_m + 0) * k + i_k] * b[(i_n + 0) * k + i_k];
  //       c[i_m * n + i_n] += a[(i_m + 0) * k + i_k] * b[(i_n + 1) * k + i_k];
  //       c[i_m * n + i_n] += a[(i_m + 0) * k + i_k] * b[(i_n + 2) * k + i_k];
  //       c[i_m * n + i_n] += a[(i_m + 0) * k + i_k] * b[(i_n + 3) * k + i_k];

  //       c[i_m * n + i_n] += a[(i_m + 1) * k + i_k] * b[(i_n + 0) * k + i_k];
  //       c[i_m * n + i_n] += a[(i_m + 1) * k + i_k] * b[(i_n + 1) * k + i_k];
  //       c[i_m * n + i_n] += a[(i_m + 1) * k + i_k] * b[(i_n + 2) * k + i_k];
  //       c[i_m * n + i_n] += a[(i_m + 1) * k + i_k] * b[(i_n + 3) * k + i_k];

  //       c[i_m * n + i_n] += a[(i_m + 2) * k + i_k] * b[(i_n + 0) * k + i_k];
  //       c[i_m * n + i_n] += a[(i_m + 2) * k + i_k] * b[(i_n + 1) * k + i_k];
  //       c[i_m * n + i_n] += a[(i_m + 2) * k + i_k] * b[(i_n + 2) * k + i_k];
  //       c[i_m * n + i_n] += a[(i_m + 2) * k + i_k] * b[(i_n + 3) * k + i_k];

  //       c[i_m * n + i_n] += a[(i_m + 3) * k + i_k] * b[(i_n + 0) * k + i_k];
  //       c[i_m * n + i_n] += a[(i_m + 3) * k + i_k] * b[(i_n + 1) * k + i_k];
  //       c[i_m * n + i_n] += a[(i_m + 3) * k + i_k] * b[(i_n + 2) * k + i_k];
  //       c[i_m * n + i_n] += a[(i_m + 3) * k + i_k] * b[(i_n + 3) * k + i_k];
  //     }
  //   }
  // }

  // for (int32_t i_k = 0; i_k < k; ++i_k)
  // {
  //   for (int32_t i_m = 0; i_m < m; ++i_m)
  //   {
  //     for (int32_t i_n = 0; i_n < n; ++i_n)
  //     {
  //       c[i_m * n + i_n] += a[i_m * k + i_k] * b[i_n * k + i_k];
  //     }
  //   }
  // }

  // for (int32_t i_k = 0; i_k < k; ++i_k)
  // {
  //   for (int32_t i_m = 0; i_m < m; ++i_m)
  //   {
  //     for (int32_t i_n = 0; i_n < n; ++i_n)
  //     {
  //       // c[i_m * n + i_n] += a[i_m * k + i_k] * b[i_n * k + i_k];
  //       c[i_n * m + i_m] += a[i_k * m + i_m] * b[i_k * n + i_n];
  //     }
  //   }
  // }
}

//

auto pointwise_convolution_backward(float const *d_out,
                                    float *d_in,
                                    float *d_kernel,
                                    float *d_bias,
                                    float const *in,
                                    float const *kernel,
                                    [[maybe_unused]] float *kernel_buffer,
                                    int32_t height,
                                    int32_t width,
                                    int32_t channels_in,
                                    int32_t channels_out) -> void
{
  // matrix_multiply_accumulating_transpose_b(d_out, kernel, d_in, height * width, channels_out, channels_in);
  // matrix_multiply_accumulating_blocked_transpose_b(d_out, kernel, d_in, height * width, channels_out, channels_in);
  matrix_multiply_accumulating_transpose_b_alt(d_out, kernel, d_in, kernel_buffer, height * width, channels_out, channels_in);
  // matrix_multiply_accumulating_blocked_transpose_b_alt(d_out, kernel, d_in, kernel_buffer, height * width, channels_out, channels_in);

  matrix_multiply_accumulating_transpose_a(in, d_out, d_kernel, channels_in, height * width, channels_out);
  // matrix_multiply_accumulating_blocked_transpose_a(in, d_out, d_kernel, channels_in, height * width, channels_out);

  for (int32_t h = 0; h < height; ++h)
  {
    for (int32_t w = 0; w < width; ++w)
    {
      for (int32_t c = 0; c < channels_out; ++c)
      {
        d_bias[c] += d_out[h * width * channels_out + w * channels_out + c];
      }
    }
  }
}

auto pointwise_convolution_backward_test_1(float const *d_out,
                                           float *d_in,
                                           float *d_kernel,
                                           float *d_bias,
                                           float const *in,
                                           float const *kernel,
                                           [[maybe_unused]] float *kernel_buffer,
                                           int32_t height,
                                           int32_t width,
                                           int32_t channels_in,
                                           int32_t channels_out) -> void
{
  matrix_multiply_accumulating_transpose_b_alt_test_1(d_out, kernel, d_in, kernel_buffer, height * width, channels_out, channels_in);

  // matrix_multiply_accumulating_transpose_a_test_1(in, d_out, d_kernel, channels_in, height * width, channels_out);

  // for (int32_t h = 0; h < height; ++h)
  // {
  //   for (int32_t w = 0; w < width; ++w)
  //   {
  //     for (int32_t c = 0; c < channels_out; ++c)
  //     {
  //       d_bias[c] += d_out[h * width * channels_out + w * channels_out + c];
  //     }
  //   }
  // }
}

auto pointwise_convolution_backward_test_2(float const *d_out,
                                           float *d_in,
                                           float *d_kernel,
                                           float *d_bias,
                                           float const *in,
                                           float const *kernel,
                                           [[maybe_unused]] float *kernel_buffer,
                                           int32_t height,
                                           int32_t width,
                                           int32_t channels_in,
                                           int32_t channels_out) -> void
{
  matrix_multiply_accumulating_transpose_b_alt_test_2(d_out, kernel, d_in, kernel_buffer, height * width, channels_out, channels_in);

  matrix_multiply_accumulating_transpose_a_test_2(in, d_out, d_kernel, channels_in, height * width, channels_out);

  for (int32_t h = 0; h < height; ++h)
  {
    for (int32_t w = 0; w < width; ++w)
    {
      for (int32_t c = 0; c < channels_out; ++c)
      {
        d_bias[c] += d_out[h * width * channels_out + w * channels_out + c];
      }
    }
  }
}

auto pointwise_convolution_backward_test_3(float const *d_out,
                                           float *d_in,
                                           float *d_kernel,
                                           float *d_bias,
                                           float const *in,
                                           float const *kernel,
                                           [[maybe_unused]] float *kernel_buffer,
                                           int32_t height,
                                           int32_t width,
                                           int32_t channels_in,
                                           int32_t channels_out) -> void
{
  matrix_multiply_accumulating_transpose_b_alt_test_3(d_out, kernel, d_in, kernel_buffer, height * width, channels_out, channels_in);

  // matrix_multiply_accumulating_transpose_a_test_2(in, d_out, d_kernel, channels_in, height * width, channels_out);

  // for (int32_t h = 0; h < height; ++h)
  // {
  //   for (int32_t w = 0; w < width; ++w)
  //   {
  //     for (int32_t c = 0; c < channels_out; ++c)
  //     {
  //       d_bias[c] += d_out[h * width * channels_out + w * channels_out + c];
  //     }
  //   }
  // }
}

auto pointwise_convolution_backward_d_out_test_1(float const *d_out,
                                                 float *d_in,
                                                 float *d_kernel,
                                                 float *d_bias,
                                                 float const *in,
                                                 float const *kernel,
                                                 [[maybe_unused]] float *kernel_buffer,
                                                 int32_t height,
                                                 int32_t width,
                                                 int32_t channels_in,
                                                 int32_t channels_out) -> void
{
  // // matrix_multiply_accumulating_transpose_b_test_1(d_out, kernel, d_in, height * width, channels_out, channels_in);
  // // matrix_multiply_accumulating_blocked_transpose_b(d_out, kernel, d_in, height * width, channels_out, channels_in);
  matrix_multiply_accumulating_transpose_b_alt_test_1(d_out, kernel, d_in, kernel_buffer, height * width, channels_out, channels_in);
  // // matrix_multiply_accumulating_blocked_transpose_b_alt(d_out, kernel, d_in, kernel_buffer, height * width, channels_out, channels_in);

  // matrix_multiply_accumulating_transpose_a_test_1(in, d_out, d_kernel, channels_in, height * width, channels_out);
  // matrix_multiply_accumulating_blocked_transpose_a(in, d_out, d_kernel, channels_in, height * width, channels_out);

  // for (int32_t h = 0; h < height; ++h)
  // {
  //   for (int32_t w = 0; w < width; ++w)
  //   {
  //     for (int32_t c = 0; c < channels_out; ++c)
  //     {
  //       d_bias[c] += d_out[h * width * channels_out + w * channels_out + c];
  //     }
  //   }
  // }
}

auto pointwise_convolution_backward_d_out_test_2(float const *d_out,
                                                 float *d_in,
                                                 float *d_kernel,
                                                 float *d_bias,
                                                 float const *in,
                                                 float const *kernel,
                                                 [[maybe_unused]] float *kernel_buffer,
                                                 int32_t height,
                                                 int32_t width,
                                                 int32_t channels_in,
                                                 int32_t channels_out) -> void
{
  // matrix_multiply_accumulating_transpose_b_test_2(d_out, kernel, d_in, height * width, channels_out, channels_in);
  // // matrix_multiply_accumulating_blocked_transpose_b(d_out, kernel, d_in, height * width, channels_out, channels_in);
  matrix_multiply_accumulating_transpose_b_alt_test_2(d_out, kernel, d_in, kernel_buffer, height * width, channels_out, channels_in);
  // // matrix_multiply_accumulating_blocked_transpose_b_alt(d_out, kernel, d_in, kernel_buffer, height * width, channels_out, channels_in);

  // matrix_multiply_accumulating_transpose_a_test_2(in, d_out, d_kernel, channels_in, height * width, channels_out);
  // matrix_multiply_accumulating_blocked_transpose_a(in, d_out, d_kernel, channels_in, height * width, channels_out);

  // for (int32_t h = 0; h < height; ++h)
  // {
  //   for (int32_t w = 0; w < width; ++w)
  //   {
  //     for (int32_t c = 0; c < channels_out; ++c)
  //     {
  //       d_bias[c] += d_out[h * width * channels_out + w * channels_out + c];
  //     }
  //   }
  // }
}

auto depthwise_convolution_forward(float const *x,
                                   float *y,
                                   float const *k,
                                   float const *b,
                                   float *x_pad,
                                   int32_t height,
                                   int32_t width,
                                   int32_t channels) -> void
{
  // int32_t constexpr kernel_height = 7;
  // int32_t constexpr kernel_width = 7;
  // int32_t constexpr kernel_height = 5;
  // int32_t constexpr kernel_width = 5;
  int32_t constexpr kernel_height = 3;
  int32_t constexpr kernel_width = 3;

  // int32_t constexpr padding = 3;
  // int32_t constexpr padding = 2;
  int32_t constexpr padding = 1;
  int32_t padded_height = height + (2 * padding);
  int32_t padded_width = width + (2 * padding);

  pad(x, x_pad, height, width, channels);

  for (int32_t h = 0; h < height; ++h)
  {
    for (int32_t w = 0; w < width; ++w)
    {
      for (int32_t c = 0; c < channels; ++c)
      {
        y[h * width * channels + w * channels + c] = b[c];
      }
    }
  }

  View3D x_pad_view = {x_pad, padded_height, padded_width, channels};
  View3D k_view = {k, kernel_height, kernel_width, channels};
  View3D y_view = {y, height, width, channels};

  for (int32_t xh = 0; xh < height; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height; ++kh)
    {
      for (int32_t xw = 0; xw < width; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            y_view(xh, xw, xc) += x_pad_view(xh + kh, xw + kw, xc) * k_view(kh, kw, xc);
          }
        }
      }
    }
  }
}

auto depthwise_convolution_forward_unpadded(float const *x,
                                            float *y,
                                            float const *k,
                                            float const *b,
                                            // float * x_pad, // temp
                                            int32_t height,
                                            int32_t width,
                                            int32_t channels) -> void
{
  int32_t constexpr kernel_height = 3;
  int32_t constexpr kernel_width = 3;

  int32_t constexpr padding = 1;

  // pad(x, x_pad, height, width, channels); // temp

  for (int32_t h = 0; h < height; ++h)
  {
    for (int32_t w = 0; w < width; ++w)
    {
      for (int32_t c = 0; c < channels; ++c)
      {
        y[h * width * channels + w * channels + c] = b[c];
      }
    }
  }

  View3D x_view = {x, height, width, channels};
  View3D k_view = {k, kernel_height, kernel_width, channels};
  View3D y_view = {y, height, width, channels};

  //

  // for (int32_t xh = 0; xh < height; ++xh)
  // {
  //   for (int32_t xw = 0; xw < width; ++xw)
  //   {
  //     for (int32_t xc = 0; xc < channels; ++xc)
  //     {
  //       y_view(xh, xw, xc) = 0.0f;
  //     }
  //   }
  // }

  //

  // top left
  for (int32_t xh = 0; xh < 1; ++xh)
  {
    for (int32_t kh = 1; kh < kernel_height; ++kh)
    {
      for (int32_t xw = 0; xw < 1; ++xw)
      {
        for (int32_t kw = 1; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            y_view(xh, xw, xc) += x_view(xh + kh - 1, xw + kw - 1, xc) * k_view(kh, kw, xc);
          }
        }
      }
    }
  }

  // top middle
  for (int32_t xh = 0; xh < 1; ++xh)
  {
    for (int32_t kh = 1; kh < kernel_height; ++kh)
    {
      for (int32_t xw = 1; xw < width - 1; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            y_view(xh, xw, xc) += x_view(xh + kh - 1, xw + kw - 1, xc) * k_view(kh, kw, xc);
          }
        }
      }
    }
  }

  // top right
  for (int32_t xh = 0; xh < 1; ++xh)
  {
    for (int32_t kh = 1; kh < kernel_height; ++kh)
    {
      for (int32_t xw = width - 1; xw < width; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width - 1; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            y_view(xh, xw, xc) += x_view(xh + kh - 1, xw + kw - 1, xc) * k_view(kh, kw, xc);
          }
        }
      }
    }
  }

  // middle left
  for (int32_t xh = 1; xh < height - 1; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height; ++kh)
    {
      for (int32_t xw = 0; xw < 1; ++xw)
      {
        for (int32_t kw = 1; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            y_view(xh, xw, xc) += x_view(xh + kh - 1, xw + kw - 1, xc) * k_view(kh, kw, xc);
          }
        }
      }
    }
  }

  // middle middle
  for (int32_t xh = 1; xh < height - 1; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height; ++kh)
    {
      for (int32_t xw = 1; xw < width - 1; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            y_view(xh, xw, xc) += x_view(xh + kh - 1, xw + kw - 1, xc) * k_view(kh, kw, xc);
          }
        }
      }
    }
  }

  // middle right
  for (int32_t xh = 1; xh < height - 1; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height; ++kh)
    {
      for (int32_t xw = width - 1; xw < width; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width - 1; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            y_view(xh, xw, xc) += x_view(xh + kh - 1, xw + kw - 1, xc) * k_view(kh, kw, xc);
          }
        }
      }
    }
  }

  // bottom left
  for (int32_t xh = height - 1; xh < height; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height - 1; ++kh)
    {
      for (int32_t xw = 0; xw < 1; ++xw)
      {
        for (int32_t kw = 1; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            y_view(xh, xw, xc) += x_view(xh + kh - 1, xw + kw - 1, xc) * k_view(kh, kw, xc);
          }
        }
      }
    }
  }

  // bottom middle
  for (int32_t xh = height - 1; xh < height; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height - 1; ++kh)
    {
      for (int32_t xw = 1; xw < width - 1; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            y_view(xh, xw, xc) += x_view(xh + kh - 1, xw + kw - 1, xc) * k_view(kh, kw, xc);
          }
        }
      }
    }
  }

  // bottom right
  for (int32_t xh = height - 1; xh < height; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height - 1; ++kh)
    {
      for (int32_t xw = width - 1; xw < width; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width - 1; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            y_view(xh, xw, xc) += x_view(xh + kh - 1, xw + kw - 1, xc) * k_view(kh, kw, xc);
          }
        }
      }
    }
  }
}

auto depthwise_convolution_forward_unpadded_(float const *x,
                                             float *y,
                                             float const *k,
                                             float const *b,
                                             // float * x_pad, // temp
                                             int32_t height,
                                             int32_t width,
                                             int32_t channels) -> void
{
  int32_t constexpr kernel_height = 3;
  int32_t constexpr kernel_width = 3;

  int32_t constexpr padding = 1;

  // pad(x, x_pad, height, width, channels); // temp

  for (int32_t h = 0; h < height; ++h)
  {
    for (int32_t w = 0; w < width; ++w)
    {
      for (int32_t c = 0; c < channels; ++c)
      {
        // y[h * width * channels + w * channels + c] = b[c];
        y[h * width * channels + w * channels + c] = 0.0f;
      }
    }
  }

  View3D x_view = {x, height, width, channels};
  View3D k_view = {k, kernel_height, kernel_width, channels};
  View3D y_view = {y, height, width, channels};

  //

  // for (int32_t xh = 0; xh < height; ++xh)
  // {
  //   for (int32_t xw = 0; xw < width; ++xw)
  //   {
  //     for (int32_t xc = 0; xc < channels; ++xc)
  //     {
  //       y_view(xh, xw, xc) = 0.0f;
  //     }
  //   }
  // }

  //

  // top left
  for (int32_t xh = 0; xh < 1; ++xh)
  {
    for (int32_t kh = 1; kh < kernel_height; ++kh)
    {
      for (int32_t xw = 0; xw < 1; ++xw)
      {
        for (int32_t kw = 1; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            y_view(xh, xw, xc) += x_view(xh + kh - 1, xw + kw - 1, xc) * k_view(kh, kw, xc);
          }
        }
      }

      for (int32_t xw = 1; xw < width - 1; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            y_view(xh, xw, xc) += x_view(xh + kh - 1, xw + kw - 1, xc) * k_view(kh, kw, xc);
          }
        }
      }

      for (int32_t xw = width - 1; xw < width; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width - 1; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            y_view(xh, xw, xc) += x_view(xh + kh - 1, xw + kw - 1, xc) * k_view(kh, kw, xc);
          }
        }
      }
    }
  }

  // middle left
  for (int32_t xh = 1; xh < height - 1; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height; ++kh)
    {
      for (int32_t xw = 0; xw < 1; ++xw)
      {
        for (int32_t kw = 1; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            y_view(xh, xw, xc) += x_view(xh + kh - 1, xw + kw - 1, xc) * k_view(kh, kw, xc);
          }
        }
      }

      for (int32_t xw = 1; xw < width - 1; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            y_view(xh, xw, xc) += x_view(xh + kh - 1, xw + kw - 1, xc) * k_view(kh, kw, xc);
          }
        }
      }

      for (int32_t xw = width - 1; xw < width; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width - 1; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            y_view(xh, xw, xc) += x_view(xh + kh - 1, xw + kw - 1, xc) * k_view(kh, kw, xc);
          }
        }
      }
    }
  }

  // bottom left
  for (int32_t xh = height - 1; xh < height; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height - 1; ++kh)
    {
      for (int32_t xw = 0; xw < 1; ++xw)
      {
        for (int32_t kw = 1; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            y_view(xh, xw, xc) += x_view(xh + kh - 1, xw + kw - 1, xc) * k_view(kh, kw, xc);
          }
        }
      }

      for (int32_t xw = 1; xw < width - 1; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            y_view(xh, xw, xc) += x_view(xh + kh - 1, xw + kw - 1, xc) * k_view(kh, kw, xc);
          }
        }
      }

      for (int32_t xw = width - 1; xw < width; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width - 1; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            y_view(xh, xw, xc) += x_view(xh + kh - 1, xw + kw - 1, xc) * k_view(kh, kw, xc);
          }
        }
      }
    }
  }
}

auto depthwise_convolution_backward(float const *d_y,
                                    float *d_x,
                                    float *d_k,
                                    float *d_b,
                                    float const *k,
                                    float const *x_pad,
                                    float *d_x_pad,
                                    int32_t height,
                                    int32_t width,
                                    int32_t channels) -> void
{
  // int32_t constexpr kernel_height = 7;
  // int32_t constexpr kernel_width = 7;
  // int32_t constexpr kernel_height = 5;
  // int32_t constexpr kernel_width = 5;
  int32_t constexpr kernel_height = 3;
  int32_t constexpr kernel_width = 3;

  // int32_t constexpr padding = 3;
  // int32_t constexpr padding = 2;
  int32_t constexpr padding = 1;
  [[maybe_unused]] int32_t padded_height = height + (2 * padding);
  int32_t padded_width = width + (2 * padding);

  for (int32_t xh = 0; xh < height; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height; ++kh)
    {
      for (int32_t xw = 0; xw < width; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            int32_t d_x_pad_i = (xh + kh) * padded_width * channels + (xw + kw) * channels + xc;

            int32_t k_i = kh * kernel_width * channels + kw * channels + xc;

            int32_t d_y_i = xh * width * channels + xw * channels + xc;

            d_x_pad[d_x_pad_i] += k[k_i] * d_y[d_y_i];
          }
        }
      }
    }
  }

  for (int32_t xh = 0; xh < height; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height; ++kh)
    {
      for (int32_t xw = 0; xw < width; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            int32_t d_k_i = kh * kernel_width * channels + kw * channels + xc;

            int32_t x_pad_i = (xh + kh) * padded_width * channels + (xw + kw) * channels + xc;

            int32_t d_y_i = xh * width * channels + xw * channels + xc;

            d_k[d_k_i] += x_pad[x_pad_i] * d_y[d_y_i];
          }
        }
      }
    }
  }

  unpad(d_x_pad, d_x, height, width, channels);

  // for (int32_t h = 0; h < height; ++h)
  // {
  //   for (int32_t w = 0; w < width; ++w)
  //   {
  //     for (int32_t c = 0; c < channels; ++c)
  //     {
  //       d_b[c] += d_y[h * width * channels + w * channels + c];
  //     }
  //   }
  // }
}

auto depthwise_convolution_backward_unpadded(float const *d_y,
                                             float *d_x,
                                             float *d_k,
                                             float *d_b,
                                             float const *k,
                                             //  float const * x_pad,
                                             float const *x,
                                             //  float * d_x_pad,
                                             int32_t height,
                                             int32_t width,
                                             int32_t channels) -> void
{
  // int32_t constexpr kernel_height = 7;
  // int32_t constexpr kernel_width = 7;
  // int32_t constexpr kernel_height = 5;
  // int32_t constexpr kernel_width = 5;
  int32_t constexpr kernel_height = 3;
  int32_t constexpr kernel_width = 3;

  // int32_t constexpr padding = 3;
  // int32_t constexpr padding = 2;
  int32_t constexpr padding = 1;
  // [[maybe_unused]] int32_t padded_height = height + (2 * padding);
  // int32_t padded_width = width + (2 * padding);

  // for (int32_t xh = 0; xh < height; ++xh)
  // {
  //   for (int32_t kh = 0; kh < kernel_height; ++kh)
  //   {
  //     for (int32_t xw = 0; xw < width; ++xw)
  //     {
  //       for (int32_t kw = 0; kw < kernel_width; ++kw)
  //       {
  //         for (int32_t xc = 0; xc < channels; ++xc)
  //         {
  //           int32_t d_x_pad_i = (xh + kh) * padded_width * channels + (xw + kw) * channels + xc;

  //           int32_t k_i = kh * kernel_width * channels + kw * channels + xc;

  //           int32_t d_y_i = xh * width * channels + xw * channels + xc;

  //           d_x_pad[d_x_pad_i] += k[k_i] * d_y[d_y_i];
  //         }
  //       }
  //     }
  //   }
  // }

  // for (int32_t xh = 0; xh < height; ++xh)
  // {
  //   for (int32_t kh = 0; kh < kernel_height; ++kh)
  //   {
  //     for (int32_t xw = 0; xw < width; ++xw)
  //     {
  //       for (int32_t kw = 0; kw < kernel_width; ++kw)
  //       {
  //         for (int32_t xc = 0; xc < channels; ++xc)
  //         {
  //           int32_t d_k_i = kh * kernel_width * channels + kw * channels + xc;

  //           int32_t x_pad_i = (xh + kh) * padded_width * channels + (xw + kw) * channels + xc;

  //           int32_t d_y_i = xh * width * channels + xw * channels + xc;

  //           d_k[d_k_i] += x_pad[x_pad_i] * d_y[d_y_i];
  //         }
  //       }
  //     }
  //   }
  // }

  // unpad(d_x_pad, d_x, height, width, channels);

  // for (int32_t h = 0; h < height; ++h)
  // {
  //   for (int32_t w = 0; w < width; ++w)
  //   {
  //     for (int32_t c = 0; c < channels; ++c)
  //     {
  //       d_b[c] += d_y[h * width * channels + w * channels + c];
  //     }
  //   }
  // }

  View3D x_view = {x, height, width, channels};
  View3D k_view = {k, kernel_height, kernel_width, channels};
  View3D d_y_view = {d_y, height, width, channels};
  View3D d_k_view = {d_k, kernel_height, kernel_width, channels};
  View3D d_x_view = {d_x, height, width, channels};

  //

  // for (int32_t xh = 0; xh < height; ++xh)
  // {
  //   for (int32_t xw = 0; xw < width; ++xw)
  //   {
  //     for (int32_t xc = 0; xc < channels; ++xc)
  //     {
  //       y_view(xh, xw, xc) = 0.0f;
  //     }
  //   }
  // }

  //

  // top left
  for (int32_t xh = 0; xh < 1; ++xh)
  {
    for (int32_t kh = 1; kh < kernel_height; ++kh)
    {
      for (int32_t xw = 0; xw < 1; ++xw)
      {
        for (int32_t kw = 1; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            d_x_view(xh + kh - 1, xw + kw - 1, xc) += k_view(kh, kw, xc) * d_y_view(xh, xw, xc);
          }
        }
      }
    }
  }

  // top middle
  for (int32_t xh = 0; xh < 1; ++xh)
  {
    for (int32_t kh = 1; kh < kernel_height; ++kh)
    {
      for (int32_t xw = 1; xw < width - 1; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            d_x_view(xh + kh - 1, xw + kw - 1, xc) += k_view(kh, kw, xc) * d_y_view(xh, xw, xc);
          }
        }
      }
    }
  }

  // top right
  for (int32_t xh = 0; xh < 1; ++xh)
  {
    for (int32_t kh = 1; kh < kernel_height; ++kh)
    {
      for (int32_t xw = width - 1; xw < width; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width - 1; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            d_x_view(xh + kh - 1, xw + kw - 1, xc) += k_view(kh, kw, xc) * d_y_view(xh, xw, xc);
          }
        }
      }
    }
  }

  // middle left
  for (int32_t xh = 1; xh < height - 1; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height; ++kh)
    {
      for (int32_t xw = 0; xw < 1; ++xw)
      {
        for (int32_t kw = 1; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            d_x_view(xh + kh - 1, xw + kw - 1, xc) += k_view(kh, kw, xc) * d_y_view(xh, xw, xc);
          }
        }
      }
    }
  }

  // middle middle
  for (int32_t xh = 1; xh < height - 1; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height; ++kh)
    {
      for (int32_t xw = 1; xw < width - 1; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            d_x_view(xh + kh - 1, xw + kw - 1, xc) += k_view(kh, kw, xc) * d_y_view(xh, xw, xc);
          }
        }
      }
    }
  }

  // middle right
  for (int32_t xh = 1; xh < height - 1; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height; ++kh)
    {
      for (int32_t xw = width - 1; xw < width; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width - 1; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            d_x_view(xh + kh - 1, xw + kw - 1, xc) += k_view(kh, kw, xc) * d_y_view(xh, xw, xc);
          }
        }
      }
    }
  }

  // bottom left
  for (int32_t xh = height - 1; xh < height; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height - 1; ++kh)
    {
      for (int32_t xw = 0; xw < 1; ++xw)
      {
        for (int32_t kw = 1; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            d_x_view(xh + kh - 1, xw + kw - 1, xc) += k_view(kh, kw, xc) * d_y_view(xh, xw, xc);
          }
        }
      }
    }
  }

  // bottom middle
  for (int32_t xh = height - 1; xh < height; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height - 1; ++kh)
    {
      for (int32_t xw = 1; xw < width - 1; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            d_x_view(xh + kh - 1, xw + kw - 1, xc) += k_view(kh, kw, xc) * d_y_view(xh, xw, xc);
          }
        }
      }
    }
  }

  // bottom right
  for (int32_t xh = height - 1; xh < height; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height - 1; ++kh)
    {
      for (int32_t xw = width - 1; xw < width; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width - 1; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            d_x_view(xh + kh - 1, xw + kw - 1, xc) += k_view(kh, kw, xc) * d_y_view(xh, xw, xc);
          }
        }
      }
    }
  }

  //
  //
  //

  // top left
  for (int32_t xh = 0; xh < 1; ++xh)
  {
    for (int32_t kh = 1; kh < kernel_height; ++kh)
    {
      for (int32_t xw = 0; xw < 1; ++xw)
      {
        for (int32_t kw = 1; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            d_k_view(kh, kw, xc) += x_view(xh + kh - 1, xw + kw - 1, xc) * d_y_view(xh, xw, xc);
          }
        }
      }
    }
  }

  // top middle
  for (int32_t xh = 0; xh < 1; ++xh)
  {
    for (int32_t kh = 1; kh < kernel_height; ++kh)
    {
      for (int32_t xw = 1; xw < width - 1; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            d_k_view(kh, kw, xc) += x_view(xh + kh - 1, xw + kw - 1, xc) * d_y_view(xh, xw, xc);
          }
        }
      }
    }
  }

  // top right
  for (int32_t xh = 0; xh < 1; ++xh)
  {
    for (int32_t kh = 1; kh < kernel_height; ++kh)
    {
      for (int32_t xw = width - 1; xw < width; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width - 1; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            d_k_view(kh, kw, xc) += x_view(xh + kh - 1, xw + kw - 1, xc) * d_y_view(xh, xw, xc);
          }
        }
      }
    }
  }

  // middle left
  for (int32_t xh = 1; xh < height - 1; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height; ++kh)
    {
      for (int32_t xw = 0; xw < 1; ++xw)
      {
        for (int32_t kw = 1; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            d_k_view(kh, kw, xc) += x_view(xh + kh - 1, xw + kw - 1, xc) * d_y_view(xh, xw, xc);
          }
        }
      }
    }
  }

  // middle middle
  for (int32_t xh = 1; xh < height - 1; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height; ++kh)
    {
      for (int32_t xw = 1; xw < width - 1; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            d_k_view(kh, kw, xc) += x_view(xh + kh - 1, xw + kw - 1, xc) * d_y_view(xh, xw, xc);
          }
        }
      }
    }
  }

  // middle right
  for (int32_t xh = 1; xh < height - 1; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height; ++kh)
    {
      for (int32_t xw = width - 1; xw < width; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width - 1; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            d_k_view(kh, kw, xc) += x_view(xh + kh - 1, xw + kw - 1, xc) * d_y_view(xh, xw, xc);
          }
        }
      }
    }
  }

  // bottom left
  for (int32_t xh = height - 1; xh < height; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height - 1; ++kh)
    {
      for (int32_t xw = 0; xw < 1; ++xw)
      {
        for (int32_t kw = 1; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            d_k_view(kh, kw, xc) += x_view(xh + kh - 1, xw + kw - 1, xc) * d_y_view(xh, xw, xc);
          }
        }
      }
    }
  }

  // bottom middle
  for (int32_t xh = height - 1; xh < height; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height - 1; ++kh)
    {
      for (int32_t xw = 1; xw < width - 1; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            d_k_view(kh, kw, xc) += x_view(xh + kh - 1, xw + kw - 1, xc) * d_y_view(xh, xw, xc);
          }
        }
      }
    }
  }

  // bottom right
  for (int32_t xh = height - 1; xh < height; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height - 1; ++kh)
    {
      for (int32_t xw = width - 1; xw < width; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width - 1; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            d_k_view(kh, kw, xc) += x_view(xh + kh - 1, xw + kw - 1, xc) * d_y_view(xh, xw, xc);
          }
        }
      }
    }
  }
}

auto patchified_convolution_forward(float const *x,
                                    float *y,
                                    float const *k,
                                    float const *b,
                                    int32_t x_h,
                                    int32_t x_w,
                                    int32_t k_n) -> void
{
  int32_t constexpr x_c = 3;
  int32_t constexpr kernel_size = 8;
  int32_t constexpr stride = kernel_size;
  // int32_t constexpr k_n = 64;

  int32_t height_out = x_h / stride;
  int32_t width_out = x_w / stride;

  for (int32_t h = 0; h < height_out; ++h)
  {
    for (int32_t w = 0; w < width_out; ++w)
    {
      for (int32_t c = 0; c < k_n; ++c)
      {
        // y[h * width_out * k_n + w * k_n + c] = b[c];
        y[h * width_out * k_n + w * k_n + c] = 0.0f;
      }
    }
  }

  // for (int32_t xh = 0; xh < x_h; xh += stride)
  // {
  //   for (int32_t kh = 0; kh < kernel_size; ++kh)
  //   {
  //     for (int32_t xw = 0; xw < x_w; xw += stride)
  //     {
  //       for (int32_t kw = 0; kw < kernel_size; ++kw)
  //       {
  //         for (int32_t kn = 0; kn < k_n; ++kn)
  //         {
  //           for (int32_t xc = 0; xc < x_c; ++xc)
  //           {
  //             int32_t y_i = (xh / stride) * width_out * k_n + (xw / stride) * k_n + kn;

  //             int32_t k_i = kh * kernel_size * x_c * k_n + kw * x_c * k_n + xc * k_n + kn;

  //             int32_t x_i = (xh + kh) * x_w * x_c + (xw + kw) * x_c + xc;

  //             y[y_i] += k[k_i] * x[x_i];
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  // for (int32_t kh = 0; kh < kernel_size; ++kh)
  // {
  //   for (int32_t xh = 0; xh < x_h; xh += stride)
  //   {
  //     for (int32_t kw = 0; kw < kernel_size; ++kw)
  //     {
  //       for (int32_t xw = 0; xw < x_w; xw += stride)
  //       {
  //         for (int32_t kn = 0; kn < k_n; ++kn)
  //         {
  //           for (int32_t xc = 0; xc < x_c; ++xc)
  //           {
  //             int32_t y_i = (xh / stride) * width_out * k_n + (xw / stride) * k_n + kn;

  //             int32_t k_i = kh * kernel_size * x_c * k_n + kw * x_c * k_n + xc * k_n + kn;

  //             int32_t x_i = (xh + kh) * x_w * x_c + (xw + kw) * x_c + xc;

  //             y[y_i] += k[k_i] * x[x_i];
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  // View3D x_view = {x, x_h, x_w, x_c};
  // // View4D k_view = {k, kernel_size, kernel_size, k_n, x_c};
  // View4D k_view = {k, kernel_size, kernel_size, x_c, k_n};
  // View3D y_view = {y, height_out, width_out, k_n};

  // for (int32_t xh = 0; xh < height_out; ++xh)
  // {
  //   for (int32_t xw = 0; xw < width_out; ++xw)
  //   {
  //     for (int32_t kh = 0; kh < kernel_size; ++kh)
  //     {
  //       for (int32_t kw = 0; kw < kernel_size; ++kw)
  //       {
  //         for (int32_t kn = 0; kn < k_n; ++kn)
  //         {
  //           for (int32_t xc = 0; xc < x_c; ++xc)
  //           {
  //             // y_view(xh, xw, kn) += k_view(kh, kw, kn, xc) * x_view(stride * xh + kh, stride * xw + kw, xc);
  //             y_view(xh, xw, kn) += k_view(kh, kw, xc, kn) * x_view(stride * xh + kh, stride * xw + kw, xc);
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  View3D x_view = {x, x_h, x_w, x_c};
  View4D k_view = {k, kernel_size, kernel_size, x_c, k_n};
  View3D y_view = {y, height_out, width_out, k_n};

  for (int32_t xh = 0; xh < height_out; ++xh)
  {
    for (int32_t xw = 0; xw < width_out; ++xw)
    {
      // for (int32_t kn = 0; kn < k_n; ++kn)
      // {
      //   y_view(xh, xw, kn) = 0.0f;
      // }

      for (int32_t kh = 0; kh < kernel_size; ++kh)
      {
        for (int32_t kw = 0; kw < kernel_size; ++kw)
        {
          for (int32_t kn = 0; kn < k_n; ++kn)
          {
            for (int32_t xc = 0; xc < x_c; ++xc)
            {
              y_view(xh, xw, kn) += k_view(kh, kw, xc, kn) * x_view(stride * xh + kh, stride * xw + kw, xc);
            }
          }
        }
      }
    }
  }

  // for (int32_t xh = 0; xh < x_h; xh += stride)
  // {
  //   for (int32_t kh = 0; kh < kernel_size; ++kh)
  //   {
  //     for (int32_t xw = 0; xw < x_w; xw += stride)
  //     {
  //       for (int32_t kw = 0; kw < kernel_size; ++kw)
  //       {
  //         for (int32_t kn = 0; kn < k_n; ++kn)
  //         {
  //           for (int32_t xc = 0; xc < x_c; ++xc)
  //           {
  //             int32_t y_i = (xh / stride) * width_out * k_n + (xw / stride) * k_n + kn;

  //             // int32_t k_i = kh * kernel_size * x_c * k_n + kw * x_c * k_n + xc * k_n + kn;
  //             int32_t k_i = kh * kernel_size * k_n * x_c + kw * k_n * x_c + kn * x_c + xc;

  //             int32_t x_i = (xh + kh) * x_w * x_c + (xw + kw) * x_c + xc;

  //             y[y_i] += k[k_i] * x[x_i];
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  // for (int32_t xh = 0; xh < x_h; xh += stride)
  // {
  //   for (int32_t kh = 0; kh < kernel_size; ++kh)
  //   {
  //     for (int32_t xw = 0; xw < x_w; xw += stride)
  //     {
  //       for (int32_t kw = 0; kw < kernel_size; ++kw)
  //       {
  //         for (int32_t xc = 0; xc < x_c; ++xc)
  //         {
  //           for (int32_t kn = 0; kn < k_n; ++kn)
  //           {
  //             int32_t y_i = (xh / stride) * width_out * k_n + (xw / stride) * k_n + kn;

  //             int32_t k_i = kh * kernel_size * x_c * k_n + kw * x_c * k_n + xc * k_n + kn;

  //             int32_t x_i = (xh + kh) * x_w * x_c + (xw + kw) * x_c + xc;

  //             y[y_i] += k[k_i] * x[x_i];
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
}

auto patchified_convolution_backward(float const *d_y,
                                     [[maybe_unused]] float *d_x,
                                     float *d_k,
                                     float *d_b,
                                     float const *x,
                                     [[maybe_unused]] float const *k,
                                     int32_t x_h,
                                     int32_t x_w,
                                     int32_t k_n) -> void
{
  int32_t constexpr x_c = 3;
  int32_t constexpr kernel_size = 8;
  int32_t constexpr stride = kernel_size;

  int32_t height_out = x_h / stride;
  int32_t width_out = x_w / stride;

  // for (int32_t xh = 0; xh < x_h; xh += stride)
  // {
  //   for (int32_t kh = 0; kh < kernel_size; ++kh)
  //   {
  //     for (int32_t xw = 0; xw < x_w; xw += stride)
  //     {
  //       for (int32_t kw = 0; kw < kernel_size; ++kw)
  //       {
  //         for (int32_t kn = 0; kn < k_n; ++kn)
  //         {
  //           for (int32_t xc = 0; xc < x_c; ++xc)
  //           {
  //             int32_t d_x_i = (xh + kh) * x_w * x_c + (xw + kw) * x_c + xc;

  //             int32_t k_i = kh * kernel_size * x_c * k_n + kw * x_c * k_n + xc * k_n + kn;

  //             int32_t d_y_i = (xh / stride) * width_out * k_n + (xw / stride) * k_n + kn;

  //             d_x[d_x_i] += k[k_i] * d_y[d_y_i];
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  // for (int32_t xh = 0; xh < x_h; xh += stride)
  // {
  //   for (int32_t kh = 0; kh < kernel_size; ++kh)
  //   {
  //     for (int32_t xw = 0; xw < x_w; xw += stride)
  //     {
  //       for (int32_t kw = 0; kw < kernel_size; ++kw)
  //       {
  //         for (int32_t kn = 0; kn < k_n; ++kn)
  //         {
  //           int32_t d_y_i = (xh / stride) * width_out * k_n + (xw / stride) * k_n + kn;

  //           for (int32_t xc = 0; xc < x_c; ++xc)
  //           {
  //             int32_t d_k_i = kh * kernel_size * x_c * k_n + kw * x_c * k_n + xc * k_n + kn;

  //             int32_t x_i = (xh + kh) * x_w * x_c + (xw + kw) * x_c + xc;

  //             d_k[d_k_i] += x[x_i] * d_y[d_y_i];
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  // View3D x_view = {x, x_h, x_w, x_c};
  // View4D d_k_view = {d_k, kernel_size, kernel_size, x_c, k_n};
  // View3D d_y_view = {d_y, height_out, width_out, k_n};

  // for (int32_t kh = 0; kh < kernel_size; ++kh)
  // {
  //   for (int32_t xh = 0; xh < height_out; ++xh)
  //   {
  //     for (int32_t kw = 0; kw < kernel_size; ++kw)
  //     {
  //       for (int32_t xw = 0; xw < width_out; ++xw)
  //       {
  //         for (int32_t kn = 0; kn < k_n; ++kn)
  //         {
  //           for (int32_t xc = 0; xc < x_c; ++xc)
  //           {
  //             d_k_view(kh, kw, xc, kn) += x_view(stride * xh + kh, stride * xw + kw, xc) * d_y_view(xh, xw, kn);
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  View3D x_view = {x, x_h, x_w, x_c};
  View4D d_k_view = {d_k, kernel_size, kernel_size, x_c, k_n};
  View3D d_y_view = {d_y, height_out, width_out, k_n};

  for (int32_t xh = 0; xh < height_out; ++xh)
  {
    for (int32_t xw = 0; xw < width_out; ++xw)
    {
      for (int32_t kh = 0; kh < kernel_size; ++kh)
      {
        for (int32_t kw = 0; kw < kernel_size; ++kw)
        {
          for (int32_t kn = 0; kn < k_n; ++kn)
          {
            for (int32_t xc = 0; xc < x_c; ++xc)
            {
              d_k_view(kh, kw, xc, kn) += x_view(stride * xh + kh, stride * xw + kw, xc) * d_y_view(xh, xw, kn);
            }
          }
        }
      }
    }
  }

  // for (int32_t h = 0; h < height_out; ++h)
  // {
  //   for (int32_t w = 0; w < width_out; ++w)
  //   {
  //     for (int32_t c = 0; c < k_n; ++c)
  //     {
  //       d_b[c] += d_y[h * width_out * k_n + w * k_n + c];
  //     }
  //   }
  // }
}

// auto patchified_convolution_im2row_forward(float const * x,
//                                            float * y,
//                                            float const * k,
//                                            float const * b,
//                                            float * rows,
//                                            int32_t x_h,
//                                            int32_t x_w,
//                                            int32_t k_n) -> void
// {
//   int32_t constexpr x_c = 3;
//   int32_t constexpr kernel_size = 8;
//   int32_t constexpr stride = kernel_size;

//   int32_t height_out = x_h / stride;
//   int32_t width_out = x_w / stride;

//   for (int32_t h = 0; h < height_out; ++h)
//   {
//     for (int32_t w = 0; w < width_out; ++w)
//     {
//       for (int32_t c = 0; c < k_n; ++c)
//       {
//         y[h * width_out * k_n + w * k_n + c] = b[c];
//       }
//     }
//   }

//   View3D x_view = {x, x_h, x_w, x_c};
//   View5D rows_view = {rows, height_out, width_out, kernel_size, kernel_size, x_c};
//   View4D k_view = {k, kernel_size, kernel_size, x_c, k_n};

//   for (int32_t xh = 0; xh < height_out; ++xh)
//   {
//     for (int32_t xw = 0; xw < width_out; ++xw)
//     {
//       for (int32_t kh = 0; kh < kernel_size; ++kh)
//       {
//         for (int32_t kw = 0; kw < kernel_size; ++kw)
//         {
//           for (int32_t xc = 0; xc < x_c; ++xc)
//           {
//             rows_view(xh, xw, kh, kw, xc) = x_view(stride * xh + kh, stride * xw + kw, xc);
//           }
//         }
//       }
//     }
//   }

//   matrix_multiply_accumulating(rows,
//                                k,
//                                y,
//                                height_out * width_out,
//                                kernel_size * kernel_size * x_c,
//                                k_n);
// }

auto patchified_convolution_im2row_forward(float const *x,
                                           float *y,
                                           float const *k,
                                           float const *b,
                                           float *rows,
                                           int32_t x_h,
                                           int32_t x_w,
                                           int32_t k_n) -> void
{
  int32_t constexpr x_c = 3;
  int32_t constexpr kernel_size = 8;
  int32_t constexpr stride = kernel_size;

  int32_t height_out = x_h / stride;
  int32_t width_out = x_w / stride;

  View3D x_view = {x, x_h, x_w, x_c};
  View5D rows_view = {rows, height_out, width_out, kernel_size, kernel_size, x_c};
  View4D k_view = {k, kernel_size, kernel_size, x_c, k_n};

  for (int32_t xh = 0; xh < height_out; ++xh)
  {
    for (int32_t xw = 0; xw < width_out; ++xw)
    {
      for (int32_t kh = 0; kh < kernel_size; ++kh)
      {
        for (int32_t kw = 0; kw < kernel_size; ++kw)
        {
          for (int32_t xc = 0; xc < x_c; ++xc)
          {
            rows_view(xh, xw, kh, kw, xc) = x_view(stride * xh + kh, stride * xw + kw, xc);
          }
        }
      }
    }
  }

  for (int32_t h = 0; h < height_out; ++h)
  {
    for (int32_t w = 0; w < width_out; ++w)
    {
      for (int32_t k = 0; k < k_n; ++k)
      {
        // y[h * width_out * k_n + w * k_n + c] = b[c];
        y[h * width_out * k_n + w * k_n + k] = 0.0f;
      }
    }
  }

  // int32_t i = 0;
  // for (int32_t xh = 0; xh < x_h; xh += stride)
  // {
  //   for (int32_t xw = 0; xw < x_w; xw += stride)
  //   {
  //     int i_ = i;
  //     for (int32_t kh = 0; kh < kernel_size; ++kh)
  //     {
  //       for (int32_t kw = 0; kw < kernel_size; ++kw)
  //       {
  //         for (int32_t xc = 0; xc < x_c; ++xc)
  //         {
  //           //   // rows_view(xh, xw, kh, kw, xc) = x_view(stride * xh + kh, stride * xw + kw, xc);
  //           //   // rows[i] = x_view(xh + kh, xw + kw, xc);
  //           //   rows[i] = x[(xh + kh) * x_w * x_c + (xw + kw) * x_c + xc];
  //           //   // rows[i] = x[(xh + kh) * x_w * x_c + (xw + kw) * x_c + 0];
  //           //   // rows[i] = x[(xh + kh) * x_w * x_c + (xw + kw) * x_c + 1];
  //           //   // rows[i] = x[(xh + kh) * x_w * x_c + (xw + kw) * x_c + 2];
  //           //   // rows_view(xh, xw, kh, kw, xc) = x_view(i_h, i_w, xc);

  //           //   // rows[i] = x[(xh + kh) * x_w * x_c + (xw + kw) * x_c + 0];
  //           //   // rows[i] = x[(xh + kh) * x_w * x_c + (xw + kw) * x_c + 1];
  //           //   // rows[i] = x[(xh + kh) * x_w * x_c + (xw + kw) * x_c + 2];

  //           rows[i] = x[(xh + kh) * x_w * x_c + (xw + kw) * x_c + xc];
  //           ++i;
  //         }

  //         // rows[i++] = x[(xh + kh) * x_w * x_c + (xw + kw) * x_c + 0];
  //         // rows[i++] = x[(xh + kh) * x_w * x_c + (xw + kw) * x_c + 1];
  //         // rows[i++] = x[(xh + kh) * x_w * x_c + (xw + kw) * x_c + 2];
  //       }
  //     }

  //     matrix_multiply_accumulating_test_3(rows + i_,
  //                                         k,
  //                                         y + (xh / stride) * width_out * k_n + (xw / stride) * k_n,
  //                                         nullptr,
  //                                         1, // height_out * width_out,
  //                                         kernel_size * kernel_size * x_c,
  //                                         k_n);
  //   }
  // }

  // // matrix_multiply_accumulating_test_2(rows,
  // //                                     k,
  // //                                     y,
  // //                                     nullptr,
  // //                                     height_out * width_out,
  // //                                     kernel_size * kernel_size * x_c,
  // //                                     k_n);
  matrix_multiply_accumulating_test_3(rows,
                                      k,
                                      y,
                                      nullptr,
                                      height_out * width_out,
                                      kernel_size * kernel_size * x_c,
                                      k_n);
  // // matrix_multiply_accumulating_test_4(rows,
  // //                                     k,
  // //                                     y,
  // //                                     nullptr,
  // //                                     height_out * width_out,
  // //                                     kernel_size * kernel_size * x_c,
  // //                                     k_n);
  // // matrix_multiply_accumulating_test_5(rows,
  // //                                     k,
  // //                                     y,
  // //                                     nullptr,
  // //                                     height_out * width_out,
  // //                                     kernel_size * kernel_size * x_c,
  // //                                     k_n);
}

auto patchified_convolution_im2row_forward_(float const *x,
                                            float *y,
                                            float const *k,
                                            float const *b,
                                            float *rows,
                                            int32_t x_h,
                                            int32_t x_w,
                                            int32_t k_n) -> void
{
  int32_t constexpr x_c = 3;
  int32_t constexpr kernel_size = 8;
  int32_t constexpr stride = kernel_size;

  int32_t height_out = x_h / stride;
  int32_t width_out = x_w / stride;

  View3D x_view = {x, x_h, x_w, x_c};
  View5D rows_view = {rows, height_out, width_out, kernel_size, kernel_size, x_c};
  View4D k_view = {k, kernel_size, kernel_size, x_c, k_n};

  // for (int32_t xh = 0; xh < height_out; ++xh)
  // {
  //   for (int32_t xw = 0; xw < width_out; ++xw)
  //   {
  //     for (int32_t kh = 0; kh < kernel_size; ++kh)
  //     {
  //       for (int32_t kw = 0; kw < kernel_size; ++kw)
  //       {
  //         for (int32_t xc = 0; xc < x_c; ++xc)
  //         {
  //           rows_view(xh, xw, kh, kw, xc) = x_view(stride * xh + kh, stride * xw + kw, xc);
  //         }
  //       }
  //     }
  //   }
  // }

  for (int32_t h = 0; h < height_out; ++h)
  {
    for (int32_t w = 0; w < width_out; ++w)
    {
      for (int32_t c = 0; c < k_n; ++c)
      {
        // y[h * width_out * k_n + w * k_n + c] = b[c];
        y[h * width_out * k_n + w * k_n + c] = 0.0f;
      }
    }
  }

  int32_t i = 0;
  for (int32_t xh = 0; xh < x_h; xh += stride)
  {
    for (int32_t xw = 0; xw < x_w; xw += stride)
    {
      int i_ = i;
      for (int32_t kh = 0; kh < kernel_size; ++kh)
      {
        for (int32_t kw = 0; kw < kernel_size; ++kw)
        {
          for (int32_t xc = 0; xc < x_c; ++xc)
          {
            //   // rows_view(xh, xw, kh, kw, xc) = x_view(stride * xh + kh, stride * xw + kw, xc);
            //   // rows[i] = x_view(xh + kh, xw + kw, xc);
            //   rows[i] = x[(xh + kh) * x_w * x_c + (xw + kw) * x_c + xc];
            //   // rows[i] = x[(xh + kh) * x_w * x_c + (xw + kw) * x_c + 0];
            //   // rows[i] = x[(xh + kh) * x_w * x_c + (xw + kw) * x_c + 1];
            //   // rows[i] = x[(xh + kh) * x_w * x_c + (xw + kw) * x_c + 2];
            //   // rows_view(xh, xw, kh, kw, xc) = x_view(i_h, i_w, xc);

            //   // rows[i] = x[(xh + kh) * x_w * x_c + (xw + kw) * x_c + 0];
            //   // rows[i] = x[(xh + kh) * x_w * x_c + (xw + kw) * x_c + 1];
            //   // rows[i] = x[(xh + kh) * x_w * x_c + (xw + kw) * x_c + 2];

            rows[i] = x[(xh + kh) * x_w * x_c + (xw + kw) * x_c + xc];
            ++i;
          }

          // rows[i++] = x[(xh + kh) * x_w * x_c + (xw + kw) * x_c + 0];
          // rows[i++] = x[(xh + kh) * x_w * x_c + (xw + kw) * x_c + 1];
          // rows[i++] = x[(xh + kh) * x_w * x_c + (xw + kw) * x_c + 2];
        }
      }

      matrix_multiply_accumulating_test_3(rows + i_,
                                          k,
                                          y + (xh / stride) * width_out * k_n + (xw / stride) * k_n,
                                          nullptr,
                                          1, // height_out * width_out,
                                          kernel_size * kernel_size * x_c,
                                          k_n);
    }
  }

  // // matrix_multiply_accumulating_test_2(rows,
  // //                                     k,
  // //                                     y,
  // //                                     nullptr,
  // //                                     height_out * width_out,
  // //                                     kernel_size * kernel_size * x_c,
  // //                                     k_n);
  // matrix_multiply_accumulating_test_3(rows,
  //                                     k,
  //                                     y,
  //                                     nullptr,
  //                                     height_out * width_out,
  //                                     kernel_size * kernel_size * x_c,
  //                                     k_n);
  // // matrix_multiply_accumulating_test_4(rows,
  // //                                     k,
  // //                                     y,
  // //                                     nullptr,
  // //                                     height_out * width_out,
  // //                                     kernel_size * kernel_size * x_c,
  // //                                     k_n);
  // // matrix_multiply_accumulating_test_5(rows,
  // //                                     k,
  // //                                     y,
  // //                                     nullptr,
  // //                                     height_out * width_out,
  // //                                     kernel_size * kernel_size * x_c,
  // //                                     k_n);
}

auto patchified_convolution_im2row_backward(float const *d_y,
                                            [[maybe_unused]] float *d_x,
                                            float *d_k,
                                            float *d_b,
                                            [[maybe_unused]] float const *x,
                                            [[maybe_unused]] float const *k,
                                            float const *rows,
                                            int32_t x_h,
                                            int32_t x_w,
                                            int32_t k_n) -> void
{
  int32_t constexpr x_c = 3;
  int32_t constexpr kernel_size = 8;
  int32_t constexpr stride = kernel_size;

  int32_t height_out = x_h / stride;
  int32_t width_out = x_w / stride;

  // for (int32_t xh = 0; xh < x_h; xh += stride)
  // {
  //   for (int32_t kh = 0; kh < kernel_size; ++kh)
  //   {
  //     for (int32_t xw = 0; xw < x_w; xw += stride)
  //     {
  //       for (int32_t kw = 0; kw < kernel_size; ++kw)
  //       {
  //         for (int32_t xc = 0; xc < x_c; ++xc)
  //         {
  //           for (int32_t kn = 0; kn < k_n; ++kn)
  //           {
  //             int32_t d_x_i = (xh + kh) * x_w * x_c + (xw + kw) * x_c + xc;

  //             int32_t k_i = kh * kernel_size * x_c * k_n + kw * x_c * k_n + xc * k_n + kn;

  //             int32_t d_y_i = (xh / stride) * width_out * k_n + (xw / stride) * k_n + kn;

  //             d_x[d_x_i] += k[k_i] * d_y[d_y_i];
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  // matrix_multiply_accumulating_transpose_a(rows,
  //                                          d_y,
  //                                          d_k,
  //                                          kernel_size * kernel_size * x_c,
  //                                          height_out * width_out,
  //                                          k_n);
  matrix_multiply_accumulating_transpose_a_test_2(rows,
                                                  d_y,
                                                  d_k,
                                                  kernel_size * kernel_size * x_c,
                                                  height_out * width_out,
                                                  k_n);

  // for (int32_t h = 0; h < height_out; ++h)
  // {
  //   for (int32_t w = 0; w < width_out; ++w)
  //   {
  //     for (int32_t c = 0; c < k_n; ++c)
  //     {
  //       d_b[c] += d_y[h * width_out * k_n + w * k_n + c];
  //     }
  //   }
  // }
}

auto update_parameters(float const *gradients,
                       float *parameters,
                       float *m,
                       float *v,
                       int32_t size,
                       float beta1,
                       float beta2,
                       float epsilon,
                       [[maybe_unused]] float schedule_multiplier,
                       float learning_rate,
                       [[maybe_unused]] float weight_decay,
                       int32_t t) -> void
{
  constexpr float b1 = 0.9;
  constexpr float b2 = 0.999;

  float b1_val = pow(b1, t);
  float b2_val = pow(b2, t);

  for (int32_t i = 0; i < size; ++i)
  {
    m[i] = beta1 * m[i] + (1.0f - beta1) * gradients[i];
    v[i] = beta2 * v[i] + (1.0f - beta2) * square(gradients[i]);

    float m_ = m[i] / (1.0f - b1_val);
    float v_ = v[i] / (1.0f - b2_val);

    // parameters[i] = parameters[i] - (learning_rate * m_ / (__builtin_sqrtf(v_) + epsilon) + weight_decay * parameters[i]);
    parameters[i] = parameters[i] - (learning_rate * m_ / (__builtin_sqrtf(v_) + epsilon));
  }
}

auto resize_bilinear_rgba_to_rgb(uint8_t const *x, float *y, int32_t xHeight, int32_t xWidth, int32_t yHeight, int32_t yWidth) -> void
{
  float hRatio = (xHeight - 1.0f) / (yHeight - 1.0f);
  float wRatio = (xWidth - 1.0f) / (yWidth - 1.0f);

  for (int32_t h = 0; h < yHeight; ++h)
  {
    int32_t hLow = max(0.0f, __builtin_floorf(hRatio * h));
    int32_t hHigh = min(xHeight - 1.0f, __builtin_ceilf(hRatio * h));

    float yWeight = (hRatio * h) - hLow;

    for (int32_t w = 0; w < yWidth; ++w)
    {
      int32_t wLow = max(0.0f, __builtin_floorf(wRatio * w));
      int32_t wHigh = min(xWidth - 1.0f, __builtin_ceilf(wRatio * w));

      float xWeight = (wRatio * w) - wLow;

      for (int32_t c = 0; c < channels_rgb; ++c)
      {
        float a_ = x[hLow * xWidth * channels_rgba + wLow * channels_rgba + c] / 255.0f;
        float b_ = x[hLow * xWidth * channels_rgba + wHigh * channels_rgba + c] / 255.0f;
        float c_ = x[hHigh * xWidth * channels_rgba + wLow * channels_rgba + c] / 255.0f;
        float d_ = x[hHigh * xWidth * channels_rgba + wHigh * channels_rgba + c] / 255.0f;

        float pixel = 0.0f;
        pixel += a_ * (1.0f - xWeight) * (1.0f - yWeight);
        pixel += b_ * xWeight * (1.0f - yWeight);
        pixel += c_ * (1.0f - xWeight) * yWeight;
        pixel += d_ * xWeight * yWeight;

        y[h * yWidth * channels_rgb + w * channels_rgb + c] = pixel;
      }
    }
  }
}

auto rotate_bilinear(float const *original, float *rotated, int32_t height, int32_t width, float cosTheta, float sinTheta, float padValue) -> void
{
  // float cosTheta = __builtin_cosf(theta);
  // float sinTheta = __builtin_sinf(theta);

  float rotationMatrix[4] = {cosTheta, sinTheta, -sinTheta, cosTheta};

  float mean = 0.0f;
  for (int32_t i = 0; i < height * width * channels_rgb; ++i)
  {
    mean += original[i] / (height * width * channels_rgb);
  }

  padValue = mean;

  for (int32_t y = 0; y < height; ++y)
  {
    for (int32_t x = 0; x < width; ++x)
    {
      for (int32_t c = 0; c < channels_rgb; ++c)
      {
        float yF = y;
        float xF = x;

        yF -= height / 2.0f;
        xF -= width / 2.0f;

        float yPrime = rotationMatrix[0] * yF + rotationMatrix[1] * xF;
        float xPrime = rotationMatrix[2] * yF + rotationMatrix[3] * xF;

        yPrime += height / 2.0f;
        xPrime += width / 2.0f;

        if ((yPrime >= 0) && (yPrime < height - 1) && (xPrime >= 0) && (xPrime < width - 1))
        {
          int32_t yLow = __builtin_floorf(yPrime);
          int32_t xLow = __builtin_floorf(xPrime);
          int32_t yHigh = __builtin_ceilf(yPrime);
          int32_t xHigh = __builtin_ceilf(xPrime);

          float yWeight = yPrime - yLow;
          float xWeight = xPrime - xLow;

          float a_ = original[yLow * width * channels_rgb + xLow * channels_rgb + c];
          float b_ = original[yLow * width * channels_rgb + xHigh * channels_rgb + c];
          float c_ = original[yHigh * width * channels_rgb + xLow * channels_rgb + c];
          float d_ = original[yHigh * width * channels_rgb + xHigh * channels_rgb + c];

          float pixel = 0.0f;
          pixel += a_ * (1.0f - yWeight) * (1.0f - xWeight);
          pixel += b_ * (1.0f - yWeight) * xWeight;
          pixel += c_ * yWeight * (1.0f - xWeight);
          pixel += d_ * yWeight * xWeight;

          // rotated[y * width * channels_rgb + x * channels_rgb + c] = Math.min(255, Math.max(0, Math.round(pixel)));
          rotated[y * width * channels_rgb + x * channels_rgb + c] = pixel;
        }
        else
        {
          rotated[y * width * channels_rgb + x * channels_rgb + c] = padValue;
        }
      }
    }
  }
}

// auto fast_approx_exp(float x) -> float
// {
//   int32_t y[2];
//   y[0] = 0;
//   y[1] = (1048576 / ln_2) * static_cast<f64>(x) + (1072693248 - 60801);
//   f64 d;
//   __builtin_memcpy(&d, &y[0], 8);
//   return static_cast<float>(d);
// }

// auto fast_approx_exp(double x) -> double
// {
//   int32_t y[2];
//   y[0] = 0;
//   y[1] = (1048576 / ln_2) * x + (1072693248 - 60801);
//   f64 d;
//   __builtin_memcpy(&d, &y[0], 8);
//   return d;
// }

// auto fast_approx_exp(float x) -> float
// {
//   // int32_t y[2];
//   // y[0] = 0;
//   // y[1] = (1048576 / ln_2) * x + (1072693248 - 60801);
//   // f64 d;
//   // __builtin_memcpy(&d, &y[0], 8);
//   // return d;

//   union
//   {
//     float f;
//     int i;
//   } y;
//   y.i = (int)(x * 0xb5645f + 0x3f7893f5);
//   return y.f;
// }

auto mean(float const *x, int32_t size) -> float
{
  float result = 0.0f;

  for (int32_t i = 0; i < size; ++i)
  {
    result += x[i];
  }
  result /= size;
  return result;
}

auto std_dev(float const *x, int32_t size, float mean) -> float
{
  float result = 0.0f;

  for (int32_t i = 0; i < size; ++i)
  {
    result += square(x[i] - mean);
  }
  result /= size;
  result = __builtin_sqrtf(result);
  // result = __builtin_sqrtf(result + epsilon);
  return result;
}

auto draw_gaussians(float *data, int32_t height, int32_t width, int32_t channels, float const *coords, float sigma) -> void
{
  float center_y = height / 2.0f;
  float center_x = width / 2.0f;

  for (int32_t h = 0; h < height; ++h)
  {
    for (int32_t w = 0; w < width; ++w)
    {
      float left_top = square(w + 0.5f - center_x);
      float right_top = square(h + 0.5f - center_y);
      float inner = (left_top + right_top) / (2.0f * square(sigma));

      data[h * width + w] = exp(-inner);
    }
  }

  float m = mean(data, height * width);
  float s = std_dev(data, height * width, m);

  for (int32_t h = 0; h < height; ++h)
  {
    for (int32_t w = 0; w < width; ++w)
    {
      for (int32_t c = 0; c < channels; ++c)
      {
        float left_top = square(w + 0.5f - coords[c * 2 + 1]);
        float right_top = square(h + 0.5f - coords[c * 2 + 0]);
        float inner = (left_top + right_top) / (2.0f * square(sigma));

        data[h * width * channels + w * channels + c] = exp(-inner);
        data[h * width * channels + w * channels + c] -= m;
        data[h * width * channels + w * channels + c] /= s;
      }
    }
  }
}

// auto draw_gaussians_fast(float * data, int32_t height, int32_t width, int32_t channels, float const * coords, float sigma) -> void
// {
//   float center_y = height / 2.0f;
//   float center_x = width / 2.0f;

//   for (int32_t h = 0; h < height; ++h)
//   {
//     for (int32_t w = 0; w < width; ++w)
//     {
//       float left_top = square(w + 0.5f - center_x);
//       float right_top = square(h + 0.5f - center_y);
//       float inner = (left_top + right_top) / (2.0f * square(sigma));

//       data[h * width + w] = fast_approx_exp(-inner);
//     }
//   }

//   float m = mean(data, height * width);
//   float s = std_dev(data, height * width, m);

//   for (int32_t h = 0; h < height; ++h)
//   {
//     for (int32_t w = 0; w < width; ++w)
//     {
//       for (int32_t c = 0; c < channels; ++c)
//       {
//         float left_top = square(w + 0.5f - coords[c * 2 + 1]);
//         float right_top = square(h + 0.5f - coords[c * 2 + 0]);
//         float inner = (left_top + right_top) / (2.0f * square(sigma));

//         data[h * width * channels + w * channels + c] = fast_approx_exp(-inner);
//         data[h * width * channels + w * channels + c] -= m;
//         data[h * width * channels + w * channels + c] /= s;
//       }
//     }
//   }
// }

auto mean_squared_error_forward(float const *x_pred, float const *x_true, int32_t size) -> float
{
  // float result = 0.0f;
  double result = 0.0;
  for (int32_t i = 0; i < size; ++i)
  {
    result += square(x_pred[i] - x_true[i]);
    // result += square(x_pred[i] - x_true[i]) / size;
  }
  result /= size;
  // return result;
  return static_cast<float>(result);
}

auto mean_squared_error_backward(float d_y, float *d_x, float const *x_pred, float const *x_true, int32_t size) -> void
{
  for (int32_t i = 0; i < size; ++i)
  {
    // d_x[i] += 2.0f * (x_pred[i] - x_true[i]) / size * d_y;
    d_x[i] = 2.0f * (x_pred[i] - x_true[i]) / size * d_y;
  }
}

auto swap(float &x, float &y) -> void
{
  float temp = x;
  x = y;
  y = temp;
}

auto flip_horizontal(float *x, int32_t height, int32_t width) -> void
{
  for (int32_t h = 0; h < height; ++h)
  {
    for (int32_t w = 0; w < width / 2; ++w)
    {
      for (int32_t c = 0; c < channels_rgb; ++c)
      {
        // reminder: try using swap
        float temp = x[h * width * channels_rgb + w * channels_rgb + c];
        x[h * width * channels_rgb + w * channels_rgb + c] = x[h * width * channels_rgb + (width - 1 - w) * channels_rgb + c];
        x[h * width * channels_rgb + (width - 1 - w) * channels_rgb + c] = temp;
      }
    }
  }
}

auto flip_vertical(float *x, int32_t height, int32_t width) -> void
{
  for (int32_t h = 0; h < height / 2; ++h)
  {
    for (int32_t w = 0; w < width; ++w)
    {
      for (int32_t c = 0; c < channels_rgb; ++c)
      {
        // reminder: try using swap
        float temp = x[h * width * channels_rgb + w * channels_rgb + c];
        x[h * width * channels_rgb + w * channels_rgb + c] = x[(height - 1 - h) * width * channels_rgb + w * channels_rgb + c];
        x[(height - 1 - h) * width * channels_rgb + w * channels_rgb + c] = temp;
      }
    }
  }
}

auto brightness_adjustment(float *x, int32_t height, int32_t width, float brightness_adjustment) -> void
{
  for (int32_t i = 0; i < height * width * channels_rgb; ++i)
  {
    x[i] += brightness_adjustment;
  }
}
