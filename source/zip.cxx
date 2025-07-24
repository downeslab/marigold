/*
Copyright (C) 2024–2025 Gregory Teicher

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

namespace
{

  uint32_t crc_32_table[256] = {};

  auto initialize_crc_32_table() -> void
  {
    for (int32_t i = 0; i < 256; ++i)
    {
      uint32_t crc_32 = static_cast<uint32_t>(i);
      for (int32_t j = 0; j < 8; ++j)
      {
        crc_32 = crc_32 & 1 ? crc_32 >> 1 ^ 0xedb88320 : crc_32 >> 1;
      }
      crc_32_table[i] = crc_32;
    }
  }

}

extern "C"
{
  auto crc_32_table_lookup(int32_t index) -> uint32_t
  {
    return crc_32_table[index];
  }

  auto ms_dos_time(int32_t second, int32_t minute, int32_t hour) -> uint16_t
  {
    uint16_t data = 0;
    data = (data & 0xffe0) | (second >> 1);
    data = (data & 0xf81f) | (minute << 5);
    data = (data & 0x7ff) | (hour << 11);
    return data;
  }

  auto ms_dos_date(int32_t date, int32_t month, int32_t year) -> uint16_t
  {
    uint16_t data = 0;
    data = (data & 0xffe0) | date;
    data = (data & 0x1fe1f) | (month << 5);
    data = (data & 0x1ff) | (year << 9);
    return data;
  }

  auto _start() -> void
  {
    initialize_crc_32_table();
  }
}
