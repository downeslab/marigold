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

/*
Please refer to https://prng.di.unimi.it/splitmix64.c and https://prng.di.unimi.it/xoshiro256starstar.c
for the original source of the SplitMix64 and xoshiro256** random number generators used in this file, respectively. Both random number generators are reproduced here in accordance with the terms of the
CC0 license (https://creativecommons.org/publicdomain/zero/1.0/) under which they were originally
made available.
*/

#include <float.h>
#include <limits.h>
#include <stdalign.h>
#include <stdatomic.h>
#include <stdint.h>

namespace
{

  uint64_t split_mix_64_state;
  uint64_t split_mix_64_calls;

  auto split_mix_64_seed(uint64_t seed) -> void
  {
    split_mix_64_state = seed;
    split_mix_64_calls = 0;
  }

  auto split_mix_64_next() -> uint64_t
  {
    uint64_t z = (split_mix_64_state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
  }

  uint64_t xoshiro_star_star_state[4];
  uint64_t xoshiro_star_star_calls;

  auto xoshiro_star_star_next() -> uint64_t
  {
    auto rotl = [](uint64_t x, int k) -> uint64_t
    {
      return (x << k) | (x >> (64 - k));
    };

    uint64_t result = rotl(xoshiro_star_star_state[1] * 5, 7) * 9;
    uint64_t t = xoshiro_star_star_state[1] << 17;

    xoshiro_star_star_state[2] ^= xoshiro_star_star_state[0];
    xoshiro_star_star_state[3] ^= xoshiro_star_star_state[1];
    xoshiro_star_star_state[1] ^= xoshiro_star_star_state[2];
    xoshiro_star_star_state[0] ^= xoshiro_star_star_state[3];

    xoshiro_star_star_state[2] ^= t;
    xoshiro_star_star_state[3] = rotl(xoshiro_star_star_state[3], 45);

    ++xoshiro_star_star_calls;

    return result;
  }

}

extern "C"
{
  auto seed(uint64_t seed) -> void
  {
    split_mix_64_seed(seed);

    xoshiro_star_star_state[0] = split_mix_64_next();
    xoshiro_star_star_state[1] = split_mix_64_next();
    xoshiro_star_star_state[2] = split_mix_64_next();
    xoshiro_star_star_state[3] = split_mix_64_next();

    xoshiro_star_star_calls = 0;
  }

  auto random_float() -> float
  {
    uint64_t next = xoshiro_star_star_next();
    float next_float = static_cast<float>((next >> 11) * 0x1.0p-53);
    return next_float;
  }

  auto random_integer(uint32_t low, uint32_t high) -> uint32_t
  {
    float next_float = random_float();
    uint32_t next_integer = low + static_cast<uint32_t>(next_float * (high - low));
    return next_integer;
  }

  auto _start() -> void
  {
  }
}
