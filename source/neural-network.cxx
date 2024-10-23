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
                                       float *sum_2,
                                       int32_t x_h,
                                       int32_t x_w,
                                       int32_t x_c) -> void;

  auto pointwise_convolution_forward(float const *in,
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

  auto depthwise_convolution_forward(float const *x,
                                     float *y,
                                     float const *k,
                                     float const *b,
                                     int32_t height,
                                     int32_t width,
                                     int32_t channels) -> void;
  auto depthwise_convolution_backward(float const *d_y,
                                      float *d_x,
                                      float *d_k,
                                      [[maybe_unused]] float *d_b,
                                      float const *k,
                                      float const *x,
                                      int32_t height,
                                      int32_t width,
                                      int32_t channels) -> void;

  auto patchified_convolution_forward(float const *x,
                                      float *y,
                                      float const *k,
                                      float const *b,
                                      float *rows,
                                      int32_t x_h,
                                      int32_t x_w,
                                      int32_t k_n) -> void;
  auto patchified_convolution_backward(float const *d_y,
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
                                     float *sum_2,
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

auto matrix_multiply_accumulating(float const *a,
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

auto matrix_multiply_accumulating_transpose_a(float const *a,
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

auto matrix_multiply_accumulating_transpose_b(float const *a,
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

auto pointwise_convolution_forward(float const *in,
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

  matrix_multiply_accumulating(in, kernel, out, kernel_, height * width, channels_in, channels_out);
}

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
  matrix_multiply_accumulating_transpose_b(d_out, kernel, d_in, kernel_buffer, height * width, channels_out, channels_in);

  matrix_multiply_accumulating_transpose_a(in, d_out, d_kernel, channels_in, height * width, channels_out);

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

auto depthwise_convolution_forward(float const *x,
                                   float *y,
                                   float const *k,
                                   float const *b,
                                   int32_t height,
                                   int32_t width,
                                   int32_t channels) -> void
{
  int32_t constexpr kernel_height = 3;
  int32_t constexpr kernel_width = 3;

  int32_t constexpr padding = 1;

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
                                    [[maybe_unused]] float *d_b,
                                    float const *k,
                                    float const *x,
                                    int32_t height,
                                    int32_t width,
                                    int32_t channels) -> void
{
  int32_t constexpr kernel_height = 3;
  int32_t constexpr kernel_width = 3;

  int32_t constexpr padding = 1;

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

  for (int32_t h = 0; h < height_out; ++h)
  {
    for (int32_t w = 0; w < width_out; ++w)
    {
      for (int32_t c = 0; c < k_n; ++c)
      {
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
            rows[i] = x[(xh + kh) * x_w * x_c + (xw + kw) * x_c + xc];
            ++i;
          }
        }
      }

      matrix_multiply_accumulating(rows + i_,
                                   k,
                                   y + (xh / stride) * width_out * k_n + (xw / stride) * k_n,
                                   nullptr,
                                   1, // height_out * width_out,
                                   kernel_size * kernel_size * x_c,
                                   k_n);
    }
  }
}

auto patchified_convolution_backward(float const *d_y,
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

  matrix_multiply_accumulating_transpose_a(rows,
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

auto mean_squared_error_forward(float const *x_pred, float const *x_true, int32_t size) -> float
{
  double result = 0.0;
  for (int32_t i = 0; i < size; ++i)
  {
    result += square(x_pred[i] - x_true[i]);
  }
  result /= size;
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
