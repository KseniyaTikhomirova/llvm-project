//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/detail/config.hpp> // namespace macro

#include <OffloadAPI.h>

#include <cassert>
#include <vector>

#ifndef _LIBSYCL_OFFLOAD_TOPOLOGY
#  define _LIBSYCL_OFFLOAD_TOPOLOGY

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

// Minimal span-like view
template <class T> struct range_view {
  const T *ptr{};
  size_t len{};
  const T *begin() const { return ptr; }
  const T *end() const { return ptr + len; }
  const T &operator[](size_t i) const { return ptr[i]; }
  size_t size() const { return len; }
};

// Contiguous global storage of platform handlers and device handles (grouped by
// platform) for a backend.
struct OffloadTopology {
  OffloadTopology() : OlBackend(OL_PLATFORM_BACKEND_UNKNOWN) {}
  OffloadTopology(ol_platform_backend_t OlBackend) : OlBackend(OlBackend) {}

  struct Range {
    uint32_t Begin = 0, Count = 0;
  };

  void set_backend(ol_platform_backend_t B) { OlBackend = B; }

  // Platforms for this backend
  range_view<ol_platform_handle_t> platforms() const {
    return {Platforms.data(), Platforms.size()};
  }

  // Devices for a specific platform (platform_id is index into Platforms)
  range_view<ol_device_handle_t> devicesForPlatform(size_t PlatformId) const {
    if (PlatformId >= PlatformDevices.size())
      return {nullptr, 0};
    const auto R = PlatformDevices[PlatformId];
    return {Devices.data() + R.Begin, R.Count};
  }

  size_t getFirstDeviceIndexForPlatform(size_t PlatformId) const {
    assert(PlatformId < PlatformDevices.size());
    const auto R = PlatformDevices[PlatformId];
    return R.Begin;
  }

  // Register new platform and devices into this topology under that platform.
  void
  registerNewPlatformAndDevices(ol_platform_handle_t NewPlatform,
                                std::vector<ol_device_handle_t> &&NewDevs) {
    Platforms.push_back(NewPlatform);

    Range R;
    R.Begin = Devices.size();
    R.Count = NewDevs.size();
    Devices.insert(Devices.end(), NewDevs.begin(), NewDevs.end());
    PlatformDevices.push_back(R);
  }

  ol_platform_backend_t backend() { return OlBackend; }

private:
  ol_platform_backend_t OlBackend = OL_PLATFORM_BACKEND_UNKNOWN;

  // Platforms and devices belonging to this backend (flattened)
  std::vector<ol_platform_handle_t> Platforms; // platforms for this backend
  std::vector<ol_device_handle_t>
      Devices; // devices for this backend (grouped by platform)

  // Vector holding range of devices for each platform (index is platform index
  // within Platforms)
  std::vector<Range>
      PlatformDevices; // PlatformDevices.size() == Platforms.size()
};

// Initialize the topologies by calling olIterateDevices.
void discoverOffloadDevices();

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_OFFLOAD_TOPOLOGY
