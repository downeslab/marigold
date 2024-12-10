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

  extern auto exp(float x) -> float;
  extern auto pow(float base, int32_t exponent) -> float;
  extern auto cos(float x) -> float;
  extern auto sin(float x) -> float;

  auto _start() -> void
  {
  }

  auto zero(float *__restrict__ x, int32_t size) -> void
  {
    for (int32_t i = 0; i < size; ++i)
    {
      x[i] = 0.0f;
    }
  }

  auto add_forward(float const *__restrict__ x_1,
                   float const *__restrict__ x_2,
                   float *__restrict__ y,
                   int32_t size) -> void;
  auto add_backward(float const *__restrict__ d_y,
                    float *__restrict__ d_x,
                    int32_t size) -> void;

  auto hard_swish_forward(float const *__restrict__ x,
                          float *__restrict__ y,
                          int32_t size) -> void;
  auto hard_swish_backward(float const *__restrict__ d_y,
                           float *__restrict__ d_x,
                           float const *__restrict__ x,
                           int32_t size) -> void;

  auto dropout_forward(float const *__restrict__ x,
                       float *__restrict__ y,
                       float const *__restrict__ mask,
                       int32_t x_h,
                       int32_t x_w,
                       int32_t x_c,
                       float drop_prob) -> void;
  auto dropout_backward(float const *__restrict__ d_y,
                        float *__restrict__ d_x,
                        float const *__restrict__ mask,
                        int32_t x_h,
                        int32_t x_w,
                        int32_t x_c) -> void;

  auto pixel_unshuffle_forward(float const *__restrict__ x,
                               float *__restrict__ y,
                               int32_t x_h,
                               int32_t x_w,
                               int32_t x_c) -> void;
  auto pixel_unshuffle_backward(float const *__restrict__ d_y,
                                float *__restrict__ d_x,
                                int32_t x_h,
                                int32_t x_w,
                                int32_t x_c) -> void;

  auto pixel_shuffle_forward(float const *__restrict__ x,
                             float *__restrict__ y,
                             int32_t x_h,
                             int32_t x_w,
                             int32_t x_c) -> void;
  auto pixel_shuffle_backward(float const *__restrict__ d_y,
                              float *__restrict__ d_x,
                              int32_t x_h,
                              int32_t x_w,
                              int32_t x_c) -> void;

  auto instance_normalization_forward(float const *__restrict__ x,
                                      float *__restrict__ y,
                                      float const *__restrict__ gamma,
                                      float const *__restrict__ beta,
                                      float *__restrict__ sample_mean,
                                      float *__restrict__ sample_std_dev,
                                      float epsilon,
                                      int32_t x_h,
                                      int32_t x_w,
                                      int32_t x_c) -> void;
  auto instance_normalization_backward(float const *__restrict__ d_y,
                                       float *__restrict__ d_x,
                                       float *__restrict__ d_gamma,
                                       float *__restrict__ d_beta,
                                       float const *__restrict__ gamma,
                                       float const *__restrict__ sample_mean,
                                       float const *__restrict__ sample_std_dev,
                                       float const *__restrict__ x,
                                       float *__restrict__ sum_1,
                                       float *__restrict__ sum_2,
                                       int32_t x_h,
                                       int32_t x_w,
                                       int32_t x_c) -> void;

  auto pointwise_convolution_forward(float const *__restrict__ in,
                                     float *__restrict__ out,
                                     float const *__restrict__ kernel,
                                     float const *__restrict__ bias,
                                     int32_t height,
                                     int32_t width,
                                     int32_t channels_in,
                                     int32_t channels_out) -> void;
  auto pointwise_convolution_backward(float const *__restrict__ d_out,
                                      float *__restrict__ d_in,
                                      float *__restrict__ d_kernel,
                                      float *__restrict__ d_bias,
                                      float const *__restrict__ in,
                                      float const *__restrict__ kernel,
                                      float *__restrict__ kernel_buffer,
                                      int32_t height,
                                      int32_t width,
                                      int32_t channels_in,
                                      int32_t channels_out) -> void;

  auto depthwise_convolution_forward(float const *__restrict__ x,
                                     float *__restrict__ y,
                                     float const *__restrict__ k,
                                     float const *__restrict__ b,
                                     int32_t height,
                                     int32_t width,
                                     int32_t channels) -> void;
  auto depthwise_convolution_backward(float const *__restrict__ d_y,
                                      float *__restrict__ d_x,
                                      float *__restrict__ d_k,
                                      float *__restrict__ d_b,
                                      float const *__restrict__ k,
                                      float const *__restrict__ x,
                                      int32_t height,
                                      int32_t width,
                                      int32_t channels) -> void;

  auto mean_squared_error_forward(float const *__restrict__ x_pred,
                                  float const *__restrict__ x_true,
                                  int32_t size) -> float;
  auto mean_squared_error_backward(float d_y,
                                   float *__restrict__ d_x,
                                   float const *__restrict__ x_pred,
                                   float const *__restrict__ x_true,
                                   int32_t size) -> void;

  auto update_parameters(float const *__restrict__ gradients,
                         float *__restrict__ parameters,
                         float *__restrict__ m,
                         float *__restrict__ v,
                         int32_t size,
                         float beta1,
                         float beta2,
                         float epsilon,
                         [[maybe_unused]] float schedule_multiplier,
                         float learning_rate,
                         [[maybe_unused]] float weight_decay,
                         int32_t t) -> void;

  auto draw_gaussians(float *__restrict__ data,
                      int32_t height,
                      int32_t width,
                      int32_t channels,
                      float const *__restrict__ coords,
                      float sigma) -> void;

  auto rgb_to_gray(float const *__restrict__ x,
                   float *__restrict__ y,
                   int32_t height,
                   int32_t width) -> void;

  auto resize_bilinear_rgba_to_rgb(uint8_t const *x,
                                   float *__restrict__ y,
                                   int32_t x_height,
                                   int32_t x_width,
                                   int32_t y_height,
                                   int32_t y_width) -> void;

  auto rotate_bilinear(float const *__restrict__ original,
                       float *__restrict__ rotated,
                       int32_t height,
                       int32_t width,
                       float theta) -> void;

  auto flip_horizontal(float *x, int32_t height, int32_t width) -> void;

  auto flip_vertical(float *x, int32_t height, int32_t width) -> void;

  auto adjust_brightness(float *x, int32_t height, int32_t width, float brightness) -> void;

  auto adjust_gamma(float *x, int32_t height, int32_t width, float gamma) -> void;
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
  constexpr auto swap(T &x, T &y) -> void
  {
    T temp = x;
    x = y;
    y = temp;
  }
}

auto add_forward(float const *__restrict__ x_1,
                 float const *__restrict__ x_2,
                 float *__restrict__ y,
                 int32_t size) -> void
{
  for (int32_t i = 0; i < size; ++i)
  {
    y[i] = x_1[i] + x_2[i];
  }
}

auto add_backward(float const *__restrict__ d_y,
                  float *__restrict__ d_x,
                  int32_t size) -> void
{
  for (int32_t i = 0; i < size; ++i)
  {
    d_x[i] += d_y[i];
  }
}

auto hard_swish_forward(float const *__restrict__ x,
                        float *__restrict__ y,
                        int32_t size) -> void
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

auto hard_swish_backward(float const *__restrict__ d_y,
                         float *__restrict__ d_x,
                         float const *__restrict__ x,
                         int32_t size) -> void
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

template <int32_t x_c>
auto dropout_forward_inner(float const *__restrict__ x,
                           float *__restrict__ y,
                           float const *__restrict__ mask,
                           int32_t x_h,
                           int32_t x_w,
                           float drop_prob) -> void
{
  for (int32_t h = 0; h < x_h; ++h)
  {
    for (int32_t w = 0; w < x_w; ++w)
    {
      for (int32_t c = 0; c < x_c; ++c)
      {
        int32_t i = h * x_w * x_c + w * x_c + c;

        y[i] = x[i] * mask[c] / (1.0 - drop_prob);
      }
    }
  }
}

auto dropout_forward(float const *__restrict__ x,
                     float *__restrict__ y,
                     float const *__restrict__ mask,
                     int32_t x_h,
                     int32_t x_w,
                     int32_t x_c,
                     float drop_prob) -> void
{
  if (x_c == 2 * 16)
  {
    dropout_forward_inner<2 * 16>(x, y, mask, x_h, x_w, drop_prob);
  }
  else if (x_c == 2 * 24)
  {
    dropout_forward_inner<2 * 24>(x, y, mask, x_h, x_w, drop_prob);
  }
  else if (x_c == 2 * 32)
  {
    dropout_forward_inner<2 * 32>(x, y, mask, x_h, x_w, drop_prob);
  }
  else if (x_c == 2 * 48)
  {
    dropout_forward_inner<2 * 48>(x, y, mask, x_h, x_w, drop_prob);
  }
  else if (x_c == 2 * 64)
  {
    dropout_forward_inner<2 * 64>(x, y, mask, x_h, x_w, drop_prob);
  }
}

template <int32_t x_c>
auto dropout_backward_inner(float const *__restrict__ d_y,
                            float *__restrict__ d_x,
                            float const *__restrict__ mask,
                            int32_t x_h,
                            int32_t x_w) -> void
{
  for (int32_t h = 0; h < x_h; ++h)
  {
    for (int32_t w = 0; w < x_w; ++w)
    {
      for (int32_t c = 0; c < x_c; ++c)
      {
        int32_t i = h * x_w * x_c + w * x_c + c;

        d_x[i] += d_y[i] * mask[c];
      }
    }
  }
}

auto dropout_backward(float const *__restrict__ d_y,
                      float *__restrict__ d_x,
                      float const *__restrict__ mask,
                      int32_t x_h,
                      int32_t x_w,
                      int32_t x_c) -> void
{
  if (x_c == 2 * 16)
  {
    dropout_backward_inner<2 * 16>(d_y, d_x, mask, x_h, x_w);
  }
  else if (x_c == 2 * 24)
  {
    dropout_backward_inner<2 * 24>(d_y, d_x, mask, x_h, x_w);
  }
  else if (x_c == 2 * 32)
  {
    dropout_backward_inner<2 * 32>(d_y, d_x, mask, x_h, x_w);
  }
  else if (x_c == 2 * 48)
  {
    dropout_backward_inner<2 * 48>(d_y, d_x, mask, x_h, x_w);
  }
  else if (x_c == 2 * 64)
  {
    dropout_backward_inner<2 * 64>(d_y, d_x, mask, x_h, x_w);
  }
}

template <int32_t x_c>
auto pixel_unshuffle_forward_inner(float const *__restrict__ x,
                                   float *__restrict__ y,
                                   int32_t x_h,
                                   int32_t x_w) -> void
{
  int32_t constexpr scale = 8;

  for (int32_t h = 0; h < x_h * scale; ++h)
  {
    for (int32_t w = 0; w < x_w * scale; ++w)
    {
      for (int32_t c = 0; c < x_c / square(scale); ++c)
      {
        int32_t x_i = h * (x_w * scale) * (x_c / square(scale));
        x_i += w * (x_c / square(scale));
        x_i += c;

        int32_t y_i = (h / scale) * x_w * x_c;
        y_i += (w / scale) * x_c;
        y_i += c * scale * scale + (h % scale) * scale + (w % scale);

        y[y_i] = x[x_i];
      }
    }
  }
}

auto pixel_unshuffle_forward(float const *__restrict__ x,
                             float *__restrict__ y,
                             int32_t x_h,
                             int32_t x_w,
                             int32_t x_c) -> void
{
  if (x_c == 8 * 8 * 1)
  {
    pixel_unshuffle_forward_inner<8 * 8 * 1>(x, y, x_h, x_w);
  }
  else if (x_c == 8 * 8 * 3)
  {
    pixel_unshuffle_forward_inner<8 * 8 * 3>(x, y, x_h, x_w);
  }
}

template <int32_t x_c>
auto pixel_unshuffle_backward_inner(float const *__restrict__ d_y,
                                    float *__restrict__ d_x,
                                    int32_t x_h,
                                    int32_t x_w) -> void
{
  int32_t constexpr scale = 8;

  for (int32_t h = 0; h < x_h * scale; ++h)
  {
    for (int32_t w = 0; w < x_w * scale; ++w)
    {
      for (int32_t c = 0; c < x_c / square(scale); ++c)
      {
        int32_t d_y_i = (h / scale) * x_w * x_c;
        d_y_i += (w / scale) * x_c;
        d_y_i += c * scale * scale + (h % scale) * scale + (w % scale);

        int32_t d_x_i = h * (x_w * scale) * (x_c / square(scale));
        d_x_i += w * (x_c / square(scale));
        d_x_i += c;

        d_x[d_x_i] += d_y[d_y_i];
      }
    }
  }
}

auto pixel_unshuffle_backward(float const *__restrict__ d_y,
                              float *__restrict__ d_x,
                              int32_t x_h,
                              int32_t x_w,
                              int32_t x_c) -> void
{
  if (x_c == 8 * 8 * 1)
  {
    pixel_unshuffle_backward_inner<8 * 8 * 1>(d_y, d_x, x_h, x_w);
  }
  else if (x_c == 8 * 8 * 3)
  {
    pixel_unshuffle_backward_inner<8 * 8 * 3>(d_y, d_x, x_h, x_w);
  }
}

template <int32_t x_c>
auto pixel_shuffle_forward_inner(float const *__restrict__ x,
                                 float *__restrict__ y,
                                 int32_t x_h,
                                 int32_t x_w) -> void
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

auto pixel_shuffle_forward(float const *__restrict__ x,
                           float *__restrict__ y,
                           int32_t x_h,
                           int32_t x_w,
                           int32_t x_c) -> void
{
  if (x_c == 4 * 4 * 1)
  {
    pixel_shuffle_forward_inner<4 * 4 * 1>(x, y, x_h, x_w);
  }
  else if (x_c == 4 * 4 * 2)
  {
    pixel_shuffle_forward_inner<4 * 4 * 2>(x, y, x_h, x_w);
  }
  else if (x_c == 4 * 4 * 3)
  {
    pixel_shuffle_forward_inner<4 * 4 * 3>(x, y, x_h, x_w);
  }
  else if (x_c == 4 * 4 * 4)
  {
    pixel_shuffle_forward_inner<4 * 4 * 4>(x, y, x_h, x_w);
  }
  else if (x_c == 4 * 4 * 5)
  {
    pixel_shuffle_forward_inner<4 * 4 * 5>(x, y, x_h, x_w);
  }
  else if (x_c == 4 * 4 * 6)
  {
    pixel_shuffle_forward_inner<4 * 4 * 6>(x, y, x_h, x_w);
  }
  else if (x_c == 4 * 4 * 7)
  {
    pixel_shuffle_forward_inner<4 * 4 * 7>(x, y, x_h, x_w);
  }
  else if (x_c == 4 * 4 * 8)
  {
    pixel_shuffle_forward_inner<4 * 4 * 8>(x, y, x_h, x_w);
  }
  else if (x_c == 4 * 4 * 9)
  {
    pixel_shuffle_forward_inner<4 * 4 * 9>(x, y, x_h, x_w);
  }
  else if (x_c == 4 * 4 * 10)
  {
    pixel_shuffle_forward_inner<4 * 4 * 10>(x, y, x_h, x_w);
  }
}

template <int32_t x_c>
auto pixel_shuffle_backward_inner(float const *__restrict__ d_y,
                                  float *__restrict__ d_x,
                                  int32_t x_h,
                                  int32_t x_w) -> void
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

        d_x[d_x_i] += d_y[d_y_i];
      }
    }
  }
}

auto pixel_shuffle_backward(float const *__restrict__ d_y,
                            float *__restrict__ d_x,
                            int32_t x_h,
                            int32_t x_w,
                            int32_t x_c) -> void
{
  if (x_c == 4 * 4 * 1)
  {
    pixel_shuffle_backward_inner<4 * 4 * 1>(d_y, d_x, x_h, x_w);
  }
  else if (x_c == 4 * 4 * 2)
  {
    pixel_shuffle_backward_inner<4 * 4 * 2>(d_y, d_x, x_h, x_w);
  }
  else if (x_c == 4 * 4 * 3)
  {
    pixel_shuffle_backward_inner<4 * 4 * 3>(d_y, d_x, x_h, x_w);
  }
  else if (x_c == 4 * 4 * 4)
  {
    pixel_shuffle_backward_inner<4 * 4 * 4>(d_y, d_x, x_h, x_w);
  }
  else if (x_c == 4 * 4 * 5)
  {
    pixel_shuffle_backward_inner<4 * 4 * 5>(d_y, d_x, x_h, x_w);
  }
  else if (x_c == 4 * 4 * 6)
  {
    pixel_shuffle_backward_inner<4 * 4 * 6>(d_y, d_x, x_h, x_w);
  }
  else if (x_c == 4 * 4 * 7)
  {
    pixel_shuffle_backward_inner<4 * 4 * 7>(d_y, d_x, x_h, x_w);
  }
  else if (x_c == 4 * 4 * 8)
  {
    pixel_shuffle_backward_inner<4 * 4 * 8>(d_y, d_x, x_h, x_w);
  }
  else if (x_c == 4 * 4 * 9)
  {
    pixel_shuffle_backward_inner<4 * 4 * 9>(d_y, d_x, x_h, x_w);
  }
  else if (x_c == 4 * 4 * 10)
  {
    pixel_shuffle_backward_inner<4 * 4 * 10>(d_y, d_x, x_h, x_w);
  }
}

template <int32_t x_c>
auto instance_normalization_forward_inner(float const *__restrict__ x,
                                          float *__restrict__ y,
                                          float const *__restrict__ gamma,
                                          float const *__restrict__ beta,
                                          float *__restrict__ sample_mean,
                                          float *__restrict__ sample_std_dev,
                                          float epsilon,
                                          int32_t x_h,
                                          int32_t x_w) -> void
{
  int32_t num = x_h * x_w;

  for (int32_t c = 0; c < x_c; ++c)
  {
    sample_mean[c] = 0.0f;
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

  for (int32_t c = 0; c < x_c; ++c)
  {
    sample_std_dev[c] = 0.0f;
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

auto instance_normalization_forward(float const *__restrict__ x,
                                    float *__restrict__ y,
                                    float const *__restrict__ gamma,
                                    float const *__restrict__ beta,
                                    float *__restrict__ sample_mean,
                                    float *__restrict__ sample_std_dev,
                                    float epsilon,
                                    int32_t x_h,
                                    int32_t x_w,
                                    int32_t x_c) -> void
{
  if (x_c == 16)
  {
    instance_normalization_forward_inner<16>(x,
                                             y,
                                             gamma,
                                             beta,
                                             sample_mean,
                                             sample_std_dev,
                                             epsilon, x_h, x_w);
  }
  else if (x_c == 24)
  {
    instance_normalization_forward_inner<24>(x,
                                             y,
                                             gamma,
                                             beta,
                                             sample_mean,
                                             sample_std_dev,
                                             epsilon, x_h, x_w);
  }
  else if (x_c == 32)
  {
    instance_normalization_forward_inner<32>(x,
                                             y,
                                             gamma,
                                             beta,
                                             sample_mean,
                                             sample_std_dev,
                                             epsilon, x_h, x_w);
  }
  else if (x_c == 48)
  {
    instance_normalization_forward_inner<48>(x,
                                             y,
                                             gamma,
                                             beta,
                                             sample_mean,
                                             sample_std_dev,
                                             epsilon, x_h, x_w);
  }
  else if (x_c == 64)
  {
    instance_normalization_forward_inner<64>(x,
                                             y,
                                             gamma,
                                             beta,
                                             sample_mean,
                                             sample_std_dev,
                                             epsilon, x_h, x_w);
  }
  else if (x_c == 2 * 16)
  {
    instance_normalization_forward_inner<2 * 16>(x,
                                                 y,
                                                 gamma,
                                                 beta,
                                                 sample_mean,
                                                 sample_std_dev,
                                                 epsilon, x_h, x_w);
  }
  else if (x_c == 2 * 24)
  {
    instance_normalization_forward_inner<2 * 24>(x,
                                                 y,
                                                 gamma,
                                                 beta,
                                                 sample_mean,
                                                 sample_std_dev,
                                                 epsilon, x_h, x_w);
  }
  else if (x_c == 2 * 32)
  {
    instance_normalization_forward_inner<2 * 32>(x,
                                                 y,
                                                 gamma,
                                                 beta,
                                                 sample_mean,
                                                 sample_std_dev,
                                                 epsilon, x_h, x_w);
  }
  else if (x_c == 2 * 48)
  {
    instance_normalization_forward_inner<2 * 48>(x,
                                                 y,
                                                 gamma,
                                                 beta,
                                                 sample_mean,
                                                 sample_std_dev,
                                                 epsilon, x_h, x_w);
  }
  else if (x_c == 2 * 64)
  {
    instance_normalization_forward_inner<2 * 64>(x,
                                                 y,
                                                 gamma,
                                                 beta,
                                                 sample_mean,
                                                 sample_std_dev,
                                                 epsilon, x_h, x_w);
  }
}

template <int32_t x_c>
auto instance_normalization_backward_inner(float const *__restrict__ d_y,
                                           float *__restrict__ d_x,
                                           float *__restrict__ d_gamma,
                                           float *__restrict__ d_beta,
                                           float const *__restrict__ gamma,
                                           float const *__restrict__ sample_mean,
                                           float const *__restrict__ sample_std_dev,
                                           float const *__restrict__ x,
                                           float *__restrict__ sum_1,
                                           float *__restrict__ sum_2,
                                           int32_t x_h,
                                           int32_t x_w) -> void
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
        sum_1[c] += d_y[i] * gamma[c] / num;
        sum_2[c] += d_y[i] * gamma[c] * ((x[i] - sample_mean[c]) / sample_std_dev[c]) / num;
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
        d_x[i] += (d_y[i] * gamma[c] - ((x[i] - sample_mean[c]) / sample_std_dev[c]) * sum_2[c] - sum_1[c]) / sample_std_dev[c];
      }
    }
  }
}

auto instance_normalization_backward(float const *__restrict__ d_y,
                                     float *__restrict__ d_x,
                                     float *__restrict__ d_gamma,
                                     float *__restrict__ d_beta,
                                     float const *__restrict__ gamma,
                                     float const *__restrict__ sample_mean,
                                     float const *__restrict__ sample_std_dev,
                                     float const *__restrict__ x,
                                     float *__restrict__ sum_1,
                                     float *__restrict__ sum_2,
                                     int32_t x_h,
                                     int32_t x_w,
                                     int32_t x_c) -> void
{
  if (x_c == 16)
  {
    instance_normalization_backward_inner<16>(d_y,
                                              d_x,
                                              d_gamma,
                                              d_beta,
                                              gamma,
                                              sample_mean,
                                              sample_std_dev,
                                              x, sum_1,
                                              sum_2,
                                              x_h,
                                              x_w);
  }
  else if (x_c == 24)
  {
    instance_normalization_backward_inner<24>(d_y,
                                              d_x,
                                              d_gamma,
                                              d_beta,
                                              gamma,
                                              sample_mean,
                                              sample_std_dev,
                                              x, sum_1,
                                              sum_2,
                                              x_h,
                                              x_w);
  }
  else if (x_c == 32)
  {
    instance_normalization_backward_inner<32>(d_y,
                                              d_x,
                                              d_gamma,
                                              d_beta,
                                              gamma,
                                              sample_mean,
                                              sample_std_dev,
                                              x, sum_1,
                                              sum_2,
                                              x_h,
                                              x_w);
  }
  else if (x_c == 48)
  {
    instance_normalization_backward_inner<48>(d_y,
                                              d_x,
                                              d_gamma,
                                              d_beta,
                                              gamma,
                                              sample_mean,
                                              sample_std_dev,
                                              x, sum_1,
                                              sum_2,
                                              x_h,
                                              x_w);
  }
  else if (x_c == 64)
  {
    instance_normalization_backward_inner<64>(d_y,
                                              d_x,
                                              d_gamma,
                                              d_beta,
                                              gamma,
                                              sample_mean,
                                              sample_std_dev,
                                              x, sum_1,
                                              sum_2,
                                              x_h,
                                              x_w);
  }
  else if (x_c == 2 * 16)
  {
    instance_normalization_backward_inner<2 * 16>(d_y,
                                                  d_x,
                                                  d_gamma,
                                                  d_beta,
                                                  gamma,
                                                  sample_mean,
                                                  sample_std_dev,
                                                  x, sum_1,
                                                  sum_2,
                                                  x_h,
                                                  x_w);
  }
  else if (x_c == 2 * 24)
  {
    instance_normalization_backward_inner<2 * 24>(d_y,
                                                  d_x,
                                                  d_gamma,
                                                  d_beta,
                                                  gamma,
                                                  sample_mean,
                                                  sample_std_dev,
                                                  x, sum_1,
                                                  sum_2,
                                                  x_h,
                                                  x_w);
  }
  else if (x_c == 2 * 32)
  {
    instance_normalization_backward_inner<2 * 32>(d_y,
                                                  d_x,
                                                  d_gamma,
                                                  d_beta,
                                                  gamma,
                                                  sample_mean,
                                                  sample_std_dev,
                                                  x, sum_1,
                                                  sum_2,
                                                  x_h,
                                                  x_w);
  }
  else if (x_c == 2 * 48)
  {
    instance_normalization_backward_inner<2 * 48>(d_y,
                                                  d_x,
                                                  d_gamma,
                                                  d_beta,
                                                  gamma,
                                                  sample_mean,
                                                  sample_std_dev,
                                                  x, sum_1,
                                                  sum_2,
                                                  x_h,
                                                  x_w);
  }
  else if (x_c == 2 * 64)
  {
    instance_normalization_backward_inner<2 * 64>(d_y,
                                                  d_x,
                                                  d_gamma,
                                                  d_beta,
                                                  gamma,
                                                  sample_mean,
                                                  sample_std_dev,
                                                  x, sum_1,
                                                  sum_2,
                                                  x_h,
                                                  x_w);
  }
}

template <int32_t channels_out>
auto pointwise_convolution_forward_inner(float const *__restrict__ in,
                                         float *__restrict__ out,
                                         float const *__restrict__ kernel,
                                         float const *__restrict__ bias,
                                         int32_t height,
                                         int32_t width,
                                         int32_t channels_in) -> void
{
  for (int32_t i_m = 0; i_m < height * width; ++i_m)
  {
    for (int32_t i_n = 0; i_n < channels_out; ++i_n)
    {
      out[i_m * channels_out + i_n] = bias[i_n];
    }

    for (int32_t i_k = 0; i_k < channels_in; i_k += 4)
    {
      float a_0 = in[i_m * channels_in + (i_k + 0)];
      float a_1 = in[i_m * channels_in + (i_k + 1)];
      float a_2 = in[i_m * channels_in + (i_k + 2)];
      float a_3 = in[i_m * channels_in + (i_k + 3)];
      for (int32_t i_n = 0; i_n < channels_out; ++i_n)
      {
        out[i_m * channels_out + i_n] += a_0 * kernel[(i_k + 0) * channels_out + i_n];
        out[i_m * channels_out + i_n] += a_1 * kernel[(i_k + 1) * channels_out + i_n];
        out[i_m * channels_out + i_n] += a_2 * kernel[(i_k + 2) * channels_out + i_n];
        out[i_m * channels_out + i_n] += a_3 * kernel[(i_k + 3) * channels_out + i_n];
      }
    }
  }
}

auto pointwise_convolution_forward(float const *__restrict__ in,
                                   float *__restrict__ out,
                                   float const *__restrict__ kernel,
                                   float const *__restrict__ bias,
                                   int32_t height,
                                   int32_t width,
                                   int32_t channels_in,
                                   int32_t channels_out) -> void
{
  if (channels_out == 16)
  {
    pointwise_convolution_forward_inner<16>(in,
                                            out,
                                            kernel,
                                            bias,
                                            height,
                                            width,
                                            channels_in);
  }
  else if (channels_out == 24)
  {
    pointwise_convolution_forward_inner<24>(in,
                                            out,
                                            kernel,
                                            bias,
                                            height,
                                            width,
                                            channels_in);
  }
  else if (channels_out == 32)
  {
    pointwise_convolution_forward_inner<32>(in,
                                            out,
                                            kernel,
                                            bias,
                                            height,
                                            width,
                                            channels_in);
  }
  else if (channels_out == 48)
  {
    pointwise_convolution_forward_inner<48>(in,
                                            out,
                                            kernel,
                                            bias,
                                            height,
                                            width,
                                            channels_in);
  }
  else if (channels_out == 64)
  {
    pointwise_convolution_forward_inner<64>(in,
                                            out,
                                            kernel,
                                            bias,
                                            height,
                                            width,
                                            channels_in);
  }
  else if (channels_out == 2 * 16)
  {
    pointwise_convolution_forward_inner<2 * 16>(in,
                                                out,
                                                kernel,
                                                bias,
                                                height,
                                                width,
                                                channels_in);
  }
  else if (channels_out == 2 * 24)
  {
    pointwise_convolution_forward_inner<2 * 24>(in,
                                                out,
                                                kernel,
                                                bias,
                                                height,
                                                width,
                                                channels_in);
  }
  else if (channels_out == 2 * 32)
  {
    pointwise_convolution_forward_inner<2 * 32>(in,
                                                out,
                                                kernel,
                                                bias,
                                                height,
                                                width,
                                                channels_in);
  }
  else if (channels_out == 2 * 48)
  {
    pointwise_convolution_forward_inner<2 * 48>(in,
                                                out,
                                                kernel,
                                                bias,
                                                height,
                                                width,
                                                channels_in);
  }
  else if (channels_out == 2 * 64)
  {
    pointwise_convolution_forward_inner<2 * 64>(in,
                                                out,
                                                kernel,
                                                bias,
                                                height,
                                                width,
                                                channels_in);
  }
  else if (channels_out == 4 * 4 * 1)
  {
    pointwise_convolution_forward_inner<4 * 4 * 1>(in,
                                                   out,
                                                   kernel,
                                                   bias,
                                                   height,
                                                   width,
                                                   channels_in);
  }
  else if (channels_out == 4 * 4 * 2)
  {
    pointwise_convolution_forward_inner<4 * 4 * 2>(in,
                                                   out,
                                                   kernel,
                                                   bias,
                                                   height,
                                                   width,
                                                   channels_in);
  }
  else if (channels_out == 4 * 4 * 3)
  {
    pointwise_convolution_forward_inner<4 * 4 * 3>(in,
                                                   out,
                                                   kernel,
                                                   bias,
                                                   height,
                                                   width,
                                                   channels_in);
  }
  else if (channels_out == 4 * 4 * 4)
  {
    pointwise_convolution_forward_inner<4 * 4 * 4>(in,
                                                   out,
                                                   kernel,
                                                   bias,
                                                   height,
                                                   width,
                                                   channels_in);
  }
  else if (channels_out == 4 * 4 * 5)
  {
    pointwise_convolution_forward_inner<4 * 4 * 5>(in,
                                                   out,
                                                   kernel,
                                                   bias,
                                                   height,
                                                   width,
                                                   channels_in);
  }
  else if (channels_out == 4 * 4 * 6)
  {
    pointwise_convolution_forward_inner<4 * 4 * 6>(in,
                                                   out,
                                                   kernel,
                                                   bias,
                                                   height,
                                                   width,
                                                   channels_in);
  }
  else if (channels_out == 4 * 4 * 7)
  {
    pointwise_convolution_forward_inner<4 * 4 * 7>(in,
                                                   out,
                                                   kernel,
                                                   bias,
                                                   height,
                                                   width,
                                                   channels_in);
  }
  else if (channels_out == 4 * 4 * 8)
  {
    pointwise_convolution_forward_inner<4 * 4 * 8>(in,
                                                   out,
                                                   kernel,
                                                   bias,
                                                   height,
                                                   width,
                                                   channels_in);
  }
  else if (channels_out == 4 * 4 * 9)
  {
    pointwise_convolution_forward_inner<4 * 4 * 9>(in,
                                                   out,
                                                   kernel,
                                                   bias,
                                                   height,
                                                   width,
                                                   channels_in);
  }
  else if (channels_out == 4 * 4 * 10)
  {
    pointwise_convolution_forward_inner<4 * 4 * 10>(in,
                                                    out,
                                                    kernel,
                                                    bias,
                                                    height,
                                                    width,
                                                    channels_in);
  }
}

template <int32_t channels_out>
auto pointwise_convolution_backward_inner(float const *__restrict__ d_out,
                                          float *__restrict__ d_in,
                                          float *__restrict__ d_kernel,
                                          float *__restrict__ d_bias,
                                          float const *__restrict__ in,
                                          float const *__restrict__ kernel,
                                          float *__restrict__ kernel_buffer,
                                          int32_t height,
                                          int32_t width,
                                          int32_t channels_in) -> void
{
  for (int32_t i_k = 0; i_k < channels_out; ++i_k)
  {
    for (int32_t i_n = 0; i_n < channels_in; ++i_n)
    {
      kernel_buffer[i_k * channels_in + i_n] = kernel[i_n * channels_out + i_k];
    }
  }

  for (int32_t i_m = 0; i_m < height * width; ++i_m)
  {
    for (int32_t i_k = 0; i_k < channels_out; i_k += 4)
    {
      float a_0 = d_out[i_m * channels_out + (i_k + 0)];
      float a_1 = d_out[i_m * channels_out + (i_k + 1)];
      float a_2 = d_out[i_m * channels_out + (i_k + 2)];
      float a_3 = d_out[i_m * channels_out + (i_k + 3)];
      for (int32_t i_n = 0; i_n < channels_in; ++i_n)
      {
        d_in[i_m * channels_in + i_n] += a_0 * kernel_buffer[(i_k + 0) * channels_in + i_n];
        d_in[i_m * channels_in + i_n] += a_1 * kernel_buffer[(i_k + 1) * channels_in + i_n];
        d_in[i_m * channels_in + i_n] += a_2 * kernel_buffer[(i_k + 2) * channels_in + i_n];
        d_in[i_m * channels_in + i_n] += a_3 * kernel_buffer[(i_k + 3) * channels_in + i_n];
      }
    }
  }

  for (int32_t i_k = 0; i_k < height * width; i_k += 4)
  {
    for (int32_t i_m = 0; i_m < channels_in; ++i_m)
    {
      float a_0 = in[(i_k + 0) * channels_in + i_m];
      float a_1 = in[(i_k + 1) * channels_in + i_m];
      float a_2 = in[(i_k + 2) * channels_in + i_m];
      float a_3 = in[(i_k + 3) * channels_in + i_m];
      for (int32_t i_n = 0; i_n < channels_out; ++i_n)
      {
        d_kernel[i_m * channels_out + i_n] += a_0 * d_out[(i_k + 0) * channels_out + i_n];
        d_kernel[i_m * channels_out + i_n] += a_1 * d_out[(i_k + 1) * channels_out + i_n];
        d_kernel[i_m * channels_out + i_n] += a_2 * d_out[(i_k + 2) * channels_out + i_n];
        d_kernel[i_m * channels_out + i_n] += a_3 * d_out[(i_k + 3) * channels_out + i_n];
      }
    }
  }

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

auto pointwise_convolution_backward(float const *__restrict__ d_out,
                                    float *__restrict__ d_in,
                                    float *__restrict__ d_kernel,
                                    float *__restrict__ d_bias,
                                    float const *__restrict__ in,
                                    float const *__restrict__ kernel,
                                    float *__restrict__ kernel_buffer,
                                    int32_t height,
                                    int32_t width,
                                    int32_t channels_in,
                                    int32_t channels_out) -> void
{
  if (channels_out == 16)
  {
    pointwise_convolution_backward_inner<16>(d_out,
                                             d_in,
                                             d_kernel,
                                             d_bias,
                                             in,
                                             kernel,
                                             kernel_buffer,
                                             height,
                                             width,
                                             channels_in);
  }
  else if (channels_out == 24)
  {
    pointwise_convolution_backward_inner<24>(d_out,
                                             d_in,
                                             d_kernel,
                                             d_bias,
                                             in,
                                             kernel,
                                             kernel_buffer,
                                             height,
                                             width,
                                             channels_in);
  }
  else if (channels_out == 32)
  {
    pointwise_convolution_backward_inner<32>(d_out,
                                             d_in,
                                             d_kernel,
                                             d_bias,
                                             in,
                                             kernel,
                                             kernel_buffer,
                                             height,
                                             width,
                                             channels_in);
  }
  else if (channels_out == 48)
  {
    pointwise_convolution_backward_inner<48>(d_out,
                                             d_in,
                                             d_kernel,
                                             d_bias,
                                             in,
                                             kernel,
                                             kernel_buffer,
                                             height,
                                             width,
                                             channels_in);
  }
  else if (channels_out == 64)
  {
    pointwise_convolution_backward_inner<64>(d_out,
                                             d_in,
                                             d_kernel,
                                             d_bias,
                                             in,
                                             kernel,
                                             kernel_buffer,
                                             height,
                                             width,
                                             channels_in);
  }
  else if (channels_out == 2 * 16)
  {
    pointwise_convolution_backward_inner<2 * 16>(d_out,
                                                 d_in,
                                                 d_kernel,
                                                 d_bias,
                                                 in,
                                                 kernel,
                                                 kernel_buffer,
                                                 height,
                                                 width,
                                                 channels_in);
  }
  else if (channels_out == 2 * 24)
  {
    pointwise_convolution_backward_inner<2 * 24>(d_out,
                                                 d_in,
                                                 d_kernel,
                                                 d_bias,
                                                 in,
                                                 kernel,
                                                 kernel_buffer,
                                                 height,
                                                 width,
                                                 channels_in);
  }
  else if (channels_out == 2 * 32)
  {
    pointwise_convolution_backward_inner<2 * 32>(d_out,
                                                 d_in,
                                                 d_kernel,
                                                 d_bias,
                                                 in,
                                                 kernel,
                                                 kernel_buffer,
                                                 height,
                                                 width,
                                                 channels_in);
  }
  else if (channels_out == 2 * 48)
  {
    pointwise_convolution_backward_inner<2 * 48>(d_out,
                                                 d_in,
                                                 d_kernel,
                                                 d_bias,
                                                 in,
                                                 kernel,
                                                 kernel_buffer,
                                                 height,
                                                 width,
                                                 channels_in);
  }
  else if (channels_out == 2 * 64)
  {
    pointwise_convolution_backward_inner<2 * 64>(d_out,
                                                 d_in,
                                                 d_kernel,
                                                 d_bias,
                                                 in,
                                                 kernel,
                                                 kernel_buffer,
                                                 height,
                                                 width,
                                                 channels_in);
  }
  else if (channels_out == 4 * 4 * 1)
  {
    pointwise_convolution_backward_inner<4 * 4 * 1>(d_out,
                                                    d_in,
                                                    d_kernel,
                                                    d_bias,
                                                    in,
                                                    kernel,
                                                    kernel_buffer,
                                                    height,
                                                    width,
                                                    channels_in);
  }
  else if (channels_out == 4 * 4 * 2)
  {
    pointwise_convolution_backward_inner<4 * 4 * 2>(d_out,
                                                    d_in,
                                                    d_kernel,
                                                    d_bias,
                                                    in,
                                                    kernel,
                                                    kernel_buffer,
                                                    height,
                                                    width,
                                                    channels_in);
  }
  else if (channels_out == 4 * 4 * 3)
  {
    pointwise_convolution_backward_inner<4 * 4 * 3>(d_out,
                                                    d_in,
                                                    d_kernel,
                                                    d_bias,
                                                    in,
                                                    kernel,
                                                    kernel_buffer,
                                                    height,
                                                    width,
                                                    channels_in);
  }
  else if (channels_out == 4 * 4 * 4)
  {
    pointwise_convolution_backward_inner<4 * 4 * 4>(d_out,
                                                    d_in,
                                                    d_kernel,
                                                    d_bias,
                                                    in,
                                                    kernel,
                                                    kernel_buffer,
                                                    height,
                                                    width,
                                                    channels_in);
  }
  else if (channels_out == 4 * 4 * 5)
  {
    pointwise_convolution_backward_inner<4 * 4 * 5>(d_out,
                                                    d_in,
                                                    d_kernel,
                                                    d_bias,
                                                    in,
                                                    kernel,
                                                    kernel_buffer,
                                                    height,
                                                    width,
                                                    channels_in);
  }
  else if (channels_out == 4 * 4 * 6)
  {
    pointwise_convolution_backward_inner<4 * 4 * 6>(d_out,
                                                    d_in,
                                                    d_kernel,
                                                    d_bias,
                                                    in,
                                                    kernel,
                                                    kernel_buffer,
                                                    height,
                                                    width,
                                                    channels_in);
  }
  else if (channels_out == 4 * 4 * 7)
  {
    pointwise_convolution_backward_inner<4 * 4 * 7>(d_out,
                                                    d_in,
                                                    d_kernel,
                                                    d_bias,
                                                    in,
                                                    kernel,
                                                    kernel_buffer,
                                                    height,
                                                    width,
                                                    channels_in);
  }
  else if (channels_out == 4 * 4 * 8)
  {
    pointwise_convolution_backward_inner<4 * 4 * 8>(d_out,
                                                    d_in,
                                                    d_kernel,
                                                    d_bias,
                                                    in,
                                                    kernel,
                                                    kernel_buffer,
                                                    height,
                                                    width,
                                                    channels_in);
  }
  else if (channels_out == 4 * 4 * 9)
  {
    pointwise_convolution_backward_inner<4 * 4 * 9>(d_out,
                                                    d_in,
                                                    d_kernel,
                                                    d_bias,
                                                    in,
                                                    kernel,
                                                    kernel_buffer,
                                                    height,
                                                    width,
                                                    channels_in);
  }
  else if (channels_out == 4 * 4 * 10)
  {
    pointwise_convolution_backward_inner<4 * 4 * 10>(d_out,
                                                     d_in,
                                                     d_kernel,
                                                     d_bias,
                                                     in,
                                                     kernel,
                                                     kernel_buffer,
                                                     height,
                                                     width,
                                                     channels_in);
  }
}

template <int32_t channels>
auto depthwise_convolution_forward_inner(float const *__restrict__ x,
                                         float *__restrict__ y,
                                         float const *__restrict__ k,
                                         float const *__restrict__ b,
                                         int32_t height,
                                         int32_t width) -> void
{
  int32_t constexpr kernel_height = 5;
  int32_t constexpr kernel_width = 5;

  int32_t constexpr padding = 2;

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

  // top left
  for (int32_t xh = 0; xh < padding; ++xh)
  {
    for (int32_t xw = 0; xw < padding; ++xw)
    {
      for (int32_t kh = padding; kh < kernel_height; ++kh)
      {
        for (int32_t kw = padding; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            int32_t y_i = xh * width * channels + xw * channels + xc;
            int32_t x_i = (xh + kh - padding) * width * channels + (xw + kw - padding) * channels + xc;
            int32_t k_i = kh * kernel_width * channels + kw * channels + xc;
            y[y_i] += x[x_i] * k[k_i];
          }
        }
      }
    }
  }

  // top middle
  for (int32_t xh = 0; xh < padding; ++xh)
  {
    for (int32_t xw = padding; xw < width - padding; ++xw)
    {
      for (int32_t kh = padding; kh < kernel_height; ++kh)
      {
        for (int32_t kw = 0; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            int32_t y_i = xh * width * channels + xw * channels + xc;
            int32_t x_i = (xh + kh - padding) * width * channels + (xw + kw - padding) * channels + xc;
            int32_t k_i = kh * kernel_width * channels + kw * channels + xc;
            y[y_i] += x[x_i] * k[k_i];
          }
        }
      }
    }
  }

  // top right
  for (int32_t xh = 0; xh < padding; ++xh)
  {
    for (int32_t xw = width - padding; xw < width; ++xw)
    {
      for (int32_t kh = padding; kh < kernel_height; ++kh)
      {
        for (int32_t kw = 0; kw < kernel_width - padding; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            int32_t y_i = xh * width * channels + xw * channels + xc;
            int32_t x_i = (xh + kh - padding) * width * channels + (xw + kw - padding) * channels + xc;
            int32_t k_i = kh * kernel_width * channels + kw * channels + xc;
            y[y_i] += x[x_i] * k[k_i];
          }
        }
      }
    }
  }

  // middle left
  for (int32_t xh = padding; xh < height - padding; ++xh)
  {
    for (int32_t xw = 0; xw < padding; ++xw)
    {
      for (int32_t kh = 0; kh < kernel_height; ++kh)
      {
        for (int32_t kw = padding; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            int32_t y_i = xh * width * channels + xw * channels + xc;
            int32_t x_i = (xh + kh - padding) * width * channels + (xw + kw - padding) * channels + xc;
            int32_t k_i = kh * kernel_width * channels + kw * channels + xc;
            y[y_i] += x[x_i] * k[k_i];
          }
        }
      }
    }
  }

  // middle middle
  for (int32_t xh = padding; xh < height - padding; ++xh)
  {
    for (int32_t xw = padding; xw < width - padding; ++xw)
    {
      for (int32_t kh = 0; kh < kernel_height; ++kh)
      {
        for (int32_t kw = 0; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            int32_t y_i = xh * width * channels + xw * channels + xc;
            int32_t x_i = (xh + kh - padding) * width * channels + (xw + kw - padding) * channels + xc;
            int32_t k_i = kh * kernel_width * channels + kw * channels + xc;
            y[y_i] += x[x_i] * k[k_i];
          }
        }
      }
    }
  }

  // middle right
  for (int32_t xh = padding; xh < height - padding; ++xh)
  {
    for (int32_t xw = width - padding; xw < width; ++xw)
    {
      for (int32_t kh = 0; kh < kernel_height; ++kh)
      {
        for (int32_t kw = 0; kw < kernel_width - padding; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            int32_t y_i = xh * width * channels + xw * channels + xc;
            int32_t x_i = (xh + kh - padding) * width * channels + (xw + kw - padding) * channels + xc;
            int32_t k_i = kh * kernel_width * channels + kw * channels + xc;
            y[y_i] += x[x_i] * k[k_i];
          }
        }
      }
    }
  }

  // bottom left
  for (int32_t xh = height - padding; xh < height; ++xh)
  {
    for (int32_t xw = 0; xw < padding; ++xw)
    {
      for (int32_t kh = 0; kh < kernel_height - padding; ++kh)
      {
        for (int32_t kw = padding; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            int32_t y_i = xh * width * channels + xw * channels + xc;
            int32_t x_i = (xh + kh - padding) * width * channels + (xw + kw - padding) * channels + xc;
            int32_t k_i = kh * kernel_width * channels + kw * channels + xc;
            y[y_i] += x[x_i] * k[k_i];
          }
        }
      }
    }
  }

  // bottom middle
  for (int32_t xh = height - padding; xh < height; ++xh)
  {
    for (int32_t xw = padding; xw < width - padding; ++xw)
    {
      for (int32_t kh = 0; kh < kernel_height - padding; ++kh)
      {
        for (int32_t kw = 0; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            int32_t y_i = xh * width * channels + xw * channels + xc;
            int32_t x_i = (xh + kh - padding) * width * channels + (xw + kw - padding) * channels + xc;
            int32_t k_i = kh * kernel_width * channels + kw * channels + xc;
            y[y_i] += x[x_i] * k[k_i];
          }
        }
      }
    }
  }

  // bottom right
  for (int32_t xh = height - padding; xh < height; ++xh)
  {
    for (int32_t xw = width - padding; xw < width; ++xw)
    {
      for (int32_t kh = 0; kh < kernel_height - padding; ++kh)
      {
        for (int32_t kw = 0; kw < kernel_width - padding; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            int32_t y_i = xh * width * channels + xw * channels + xc;
            int32_t x_i = (xh + kh - padding) * width * channels + (xw + kw - padding) * channels + xc;
            int32_t k_i = kh * kernel_width * channels + kw * channels + xc;
            y[y_i] += x[x_i] * k[k_i];
          }
        }
      }
    }
  }
}

auto depthwise_convolution_forward(float const *__restrict__ x,
                                   float *__restrict__ y,
                                   float const *__restrict__ k,
                                   float const *__restrict__ b,
                                   int32_t height,
                                   int32_t width,
                                   int32_t channels) -> void
{
  if (channels == 2 * 16)
  {
    depthwise_convolution_forward_inner<2 * 16>(x, y, k, b, height, width);
  }
  else if (channels == 2 * 24)
  {
    depthwise_convolution_forward_inner<2 * 24>(x, y, k, b, height, width);
  }
  else if (channels == 2 * 32)
  {
    depthwise_convolution_forward_inner<2 * 32>(x, y, k, b, height, width);
  }
  else if (channels == 2 * 48)
  {
    depthwise_convolution_forward_inner<2 * 48>(x, y, k, b, height, width);
  }
  else if (channels == 2 * 64)
  {
    depthwise_convolution_forward_inner<2 * 64>(x, y, k, b, height, width);
  }
}

template <int32_t channels>
auto depthwise_convolution_backward_inner(float const *__restrict__ d_y,
                                          float *__restrict__ d_x,
                                          float *__restrict__ d_k,
                                          float *__restrict__ d_b,
                                          float const *__restrict__ k,
                                          float const *__restrict__ x,
                                          int32_t height,
                                          int32_t width) -> void
{
  int32_t constexpr kernel_height = 5;
  int32_t constexpr kernel_width = 5;

  int32_t constexpr padding = 2;

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

  // top left
  for (int32_t xh = 0; xh < padding; ++xh)
  {
    for (int32_t kh = padding; kh < kernel_height; ++kh)
    {
      for (int32_t xw = 0; xw < padding; ++xw)
      {
        for (int32_t kw = padding; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            int32_t y_i = xh * width * channels + xw * channels + xc;
            int32_t x_i = (xh + kh - padding) * width * channels + (xw + kw - padding) * channels + xc;
            int32_t k_i = kh * kernel_width * channels + kw * channels + xc;
            d_k[k_i] += x[x_i] * d_y[y_i];
            d_x[x_i] += k[k_i] * d_y[y_i];
          }
        }
      }
    }
  }

  // top middle
  for (int32_t xh = 0; xh < padding; ++xh)
  {
    for (int32_t kh = padding; kh < kernel_height; ++kh)
    {
      for (int32_t xw = padding; xw < width - padding; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            int32_t y_i = xh * width * channels + xw * channels + xc;
            int32_t x_i = (xh + kh - padding) * width * channels + (xw + kw - padding) * channels + xc;
            int32_t k_i = kh * kernel_width * channels + kw * channels + xc;
            d_k[k_i] += x[x_i] * d_y[y_i];
            d_x[x_i] += k[k_i] * d_y[y_i];
          }
        }
      }
    }
  }

  // top right
  for (int32_t xh = 0; xh < padding; ++xh)
  {
    for (int32_t kh = padding; kh < kernel_height; ++kh)
    {
      for (int32_t xw = width - padding; xw < width; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width - padding; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            int32_t y_i = xh * width * channels + xw * channels + xc;
            int32_t x_i = (xh + kh - padding) * width * channels + (xw + kw - padding) * channels + xc;
            int32_t k_i = kh * kernel_width * channels + kw * channels + xc;
            d_k[k_i] += x[x_i] * d_y[y_i];
            d_x[x_i] += k[k_i] * d_y[y_i];
          }
        }
      }
    }
  }

  // middle left
  for (int32_t xh = padding; xh < height - padding; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height; ++kh)
    {
      for (int32_t xw = 0; xw < padding; ++xw)
      {
        for (int32_t kw = padding; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            int32_t y_i = xh * width * channels + xw * channels + xc;
            int32_t x_i = (xh + kh - padding) * width * channels + (xw + kw - padding) * channels + xc;
            int32_t k_i = kh * kernel_width * channels + kw * channels + xc;
            d_k[k_i] += x[x_i] * d_y[y_i];
            d_x[x_i] += k[k_i] * d_y[y_i];
          }
        }
      }
    }
  }

  // middle middle
  for (int32_t xh = padding; xh < height - padding; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height; ++kh)
    {
      for (int32_t xw = padding; xw < width - padding; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            int32_t y_i = xh * width * channels + xw * channels + xc;
            int32_t x_i = (xh + kh - padding) * width * channels + (xw + kw - padding) * channels + xc;
            int32_t k_i = kh * kernel_width * channels + kw * channels + xc;
            d_k[k_i] += x[x_i] * d_y[y_i];
            d_x[x_i] += k[k_i] * d_y[y_i];
          }
        }
      }
    }
  }

  // middle right
  for (int32_t xh = padding; xh < height - padding; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height; ++kh)
    {
      for (int32_t xw = width - padding; xw < width; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width - padding; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            int32_t y_i = xh * width * channels + xw * channels + xc;
            int32_t x_i = (xh + kh - padding) * width * channels + (xw + kw - padding) * channels + xc;
            int32_t k_i = kh * kernel_width * channels + kw * channels + xc;
            d_k[k_i] += x[x_i] * d_y[y_i];
            d_x[x_i] += k[k_i] * d_y[y_i];
          }
        }
      }
    }
  }

  // bottom left
  for (int32_t xh = height - padding; xh < height; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height - padding; ++kh)
    {
      for (int32_t xw = 0; xw < padding; ++xw)
      {
        for (int32_t kw = padding; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            int32_t y_i = xh * width * channels + xw * channels + xc;
            int32_t x_i = (xh + kh - padding) * width * channels + (xw + kw - padding) * channels + xc;
            int32_t k_i = kh * kernel_width * channels + kw * channels + xc;
            d_k[k_i] += x[x_i] * d_y[y_i];
            d_x[x_i] += k[k_i] * d_y[y_i];
          }
        }
      }
    }
  }

  // bottom middle
  for (int32_t xh = height - padding; xh < height; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height - padding; ++kh)
    {
      for (int32_t xw = padding; xw < width - padding; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            int32_t y_i = xh * width * channels + xw * channels + xc;
            int32_t x_i = (xh + kh - padding) * width * channels + (xw + kw - padding) * channels + xc;
            int32_t k_i = kh * kernel_width * channels + kw * channels + xc;
            d_k[k_i] += x[x_i] * d_y[y_i];
            d_x[x_i] += k[k_i] * d_y[y_i];
          }
        }
      }
    }
  }

  // bottom right
  for (int32_t xh = height - padding; xh < height; ++xh)
  {
    for (int32_t kh = 0; kh < kernel_height - padding; ++kh)
    {
      for (int32_t xw = width - padding; xw < width; ++xw)
      {
        for (int32_t kw = 0; kw < kernel_width - padding; ++kw)
        {
          for (int32_t xc = 0; xc < channels; ++xc)
          {
            int32_t y_i = xh * width * channels + xw * channels + xc;
            int32_t x_i = (xh + kh - padding) * width * channels + (xw + kw - padding) * channels + xc;
            int32_t k_i = kh * kernel_width * channels + kw * channels + xc;
            d_k[k_i] += x[x_i] * d_y[y_i];
            d_x[x_i] += k[k_i] * d_y[y_i];
          }
        }
      }
    }
  }
}

auto depthwise_convolution_backward(float const *__restrict__ d_y,
                                    float *__restrict__ d_x,
                                    float *__restrict__ d_k,
                                    float *__restrict__ d_b,
                                    float const *__restrict__ k,
                                    float const *__restrict__ x,
                                    int32_t height,
                                    int32_t width,
                                    int32_t channels) -> void
{
  if (channels == 2 * 16)
  {
    depthwise_convolution_backward_inner<2 * 16>(d_y, d_x, d_k, d_b, k, x, height, width);
  }
  else if (channels == 2 * 24)
  {
    depthwise_convolution_backward_inner<2 * 24>(d_y, d_x, d_k, d_b, k, x, height, width);
  }
  else if (channels == 2 * 32)
  {
    depthwise_convolution_backward_inner<2 * 32>(d_y, d_x, d_k, d_b, k, x, height, width);
  }
  else if (channels == 2 * 48)
  {
    depthwise_convolution_backward_inner<2 * 48>(d_y, d_x, d_k, d_b, k, x, height, width);
  }
  else if (channels == 2 * 64)
  {
    depthwise_convolution_backward_inner<2 * 64>(d_y, d_x, d_k, d_b, k, x, height, width);
  }
}

auto mean_squared_error_forward(float const *__restrict__ x_pred,
                                float const *__restrict__ x_true,
                                int32_t size) -> float
{
  double result = 0.0;
  for (int32_t i = 0; i < size; ++i)
  {
    result += square(x_pred[i] - x_true[i]);
  }
  result /= size;
  return static_cast<float>(result);
}

auto mean_squared_error_backward(float d_y,
                                 float *__restrict__ d_x,
                                 float const *__restrict__ x_pred,
                                 float const *__restrict__ x_true,
                                 int32_t size) -> void
{
  for (int32_t i = 0; i < size; ++i)
  {
    // d_x[i] += 2.0f * (x_pred[i] - x_true[i]) / size * d_y;
    d_x[i] = 2.0f * (x_pred[i] - x_true[i]) / size * d_y;
  }
}

auto update_parameters(float const *__restrict__ gradients,
                       float *__restrict__ parameters,
                       float *__restrict__ m,
                       float *__restrict__ v,
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
  constexpr float b2 = 0.99;

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

auto mean(float const *__restrict__ x, int32_t size) -> float
{
  float result = 0.0f;

  for (int32_t i = 0; i < size; ++i)
  {
    result += x[i];
  }
  result /= size;
  return result;
}

auto std_dev(float const *__restrict__ x, int32_t size, float mean) -> float
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

auto draw_gaussians(float *__restrict__ data,
                    int32_t height,
                    int32_t width,
                    int32_t channels,
                    float const *__restrict__ coords,
                    float sigma) -> void
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

auto rgb_to_gray(float const *__restrict__ x,
                 float *__restrict__ y,
                 int32_t height,
                 int32_t width) -> void
{
  for (int32_t h = 0; h < height; ++h)
  {
    for (int32_t w = 0; w < width; ++w)
    {
      y[h * width + w] = 0.2125 * x[h * width * channels_rgb + w * channels_rgb + 0];
      y[h * width + w] += 0.7154 * x[h * width * channels_rgb + w * channels_rgb + 1];
      y[h * width + w] += 0.0721 * x[h * width * channels_rgb + w * channels_rgb + 2];
    }
  }
}

auto resize_bilinear_rgba_to_rgb(uint8_t const *x,
                                 float *__restrict__ y,
                                 int32_t x_height,
                                 int32_t x_width,
                                 int32_t y_height,
                                 int32_t y_width) -> void
{
  float h_ratio = (x_height - 1.0f) / (y_height - 1.0f);
  float w_ratio = (x_width - 1.0f) / (y_width - 1.0f);

  for (int32_t h = 0; h < y_height; ++h)
  {
    int32_t h_low = max(0.0f, __builtin_floorf(h_ratio * h));
    int32_t h_high = min(x_height - 1.0f, __builtin_ceilf(h_ratio * h));

    float y_weight = (h_ratio * h) - h_low;

    for (int32_t w = 0; w < y_width; ++w)
    {
      int32_t w_low = max(0.0f, __builtin_floorf(w_ratio * w));
      int32_t w_high = min(x_width - 1.0f, __builtin_ceilf(w_ratio * w));

      float x_weight = (w_ratio * w) - w_low;

      for (int32_t c = 0; c < channels_rgb; ++c)
      {
        float a_ = x[h_low * x_width * channels_rgba + w_low * channels_rgba + c] / 255.0f;
        float b_ = x[h_low * x_width * channels_rgba + w_high * channels_rgba + c] / 255.0f;
        float c_ = x[h_high * x_width * channels_rgba + w_low * channels_rgba + c] / 255.0f;
        float d_ = x[h_high * x_width * channels_rgba + w_high * channels_rgba + c] / 255.0f;

        float value = 0.0f;
        value += a_ * (1.0f - x_weight) * (1.0f - y_weight);
        value += b_ * x_weight * (1.0f - y_weight);
        value += c_ * (1.0f - x_weight) * y_weight;
        value += d_ * x_weight * y_weight;

        y[h * y_width * channels_rgb + w * channels_rgb + c] = value;
      }
    }
  }
}

auto rotate_bilinear(float const *__restrict__ original,
                     float *__restrict__ rotated,
                     int32_t height,
                     int32_t width,
                     float theta) -> void
{
  float cos_theta = cos(theta);
  float sin_theta = sin(theta);
  float rotation_matrix[4] = {cos_theta, sin_theta, -sin_theta, cos_theta};

  float mean = 0.0f;
  for (int32_t i = 0; i < height * width * channels_rgb; ++i)
  {
    mean += original[i] / (height * width * channels_rgb);
  }

  for (int32_t y = 0; y < height; ++y)
  {
    for (int32_t x = 0; x < width; ++x)
    {
      float y_temp = y;
      float x_temp = x;

      y_temp -= height / 2.0f;
      x_temp -= width / 2.0f;

      float y_prime = rotation_matrix[0] * y_temp + rotation_matrix[1] * x_temp;
      float x_prime = rotation_matrix[2] * y_temp + rotation_matrix[3] * x_temp;

      y_prime += height / 2.0f;
      x_prime += width / 2.0f;

      for (int32_t c = 0; c < channels_rgb; ++c)
      {
        if ((y_prime >= 0) && (y_prime < height - 1) && (x_prime >= 0) && (x_prime < width - 1))
        {
          int32_t y_low = __builtin_floorf(y_prime);
          int32_t x_low = __builtin_floorf(x_prime);
          int32_t y_high = __builtin_ceilf(y_prime);
          int32_t x_high = __builtin_ceilf(x_prime);

          float y_weight = y_prime - y_low;
          float x_weight = x_prime - x_low;

          float a_ = original[y_low * width * channels_rgb + x_low * channels_rgb + c];
          float b_ = original[y_low * width * channels_rgb + x_high * channels_rgb + c];
          float c_ = original[y_high * width * channels_rgb + x_low * channels_rgb + c];
          float d_ = original[y_high * width * channels_rgb + x_high * channels_rgb + c];

          float value = 0.0f;
          value += a_ * (1.0f - y_weight) * (1.0f - x_weight);
          value += b_ * (1.0f - y_weight) * x_weight;
          value += c_ * y_weight * (1.0f - x_weight);
          value += d_ * y_weight * x_weight;

          rotated[y * width * channels_rgb + x * channels_rgb + c] = value;
        }
        else
        {
          rotated[y * width * channels_rgb + x * channels_rgb + c] = mean;
        }
      }
    }
  }
}

auto flip_horizontal(float *x, int32_t height, int32_t width) -> void
{
  for (int32_t h = 0; h < height; ++h)
  {
    for (int32_t w = 0; w < width / 2; ++w)
    {
      for (int32_t c = 0; c < channels_rgb; ++c)
      {
        swap(x[h * width * channels_rgb + w * channels_rgb + c], x[h * width * channels_rgb + (width - 1 - w) * channels_rgb + c]);
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
        swap(x[h * width * channels_rgb + w * channels_rgb + c], x[(height - 1 - h) * width * channels_rgb + w * channels_rgb + c]);
      }
    }
  }
}

auto adjust_brightness(float *x, int32_t height, int32_t width, float brightness) -> void
{
  for (int32_t i = 0; i < height * width * channels_rgb; ++i)
  {
    x[i] += brightness;
  }
}

auto adjust_gamma(float *x, int32_t height, int32_t width, float gamma) -> void
{
  for (int32_t i = 0; i < height * width * channels_rgb; ++i)
  {
    x[i] = pow(x[i], gamma);
  }
}
