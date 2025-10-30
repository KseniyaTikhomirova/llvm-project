//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_OFFLOAD_TOPOLOGY
#define _LIBSYCL_OFFLOAD_TOPOLOGY

#include <sycl/__impl/detail/config.hpp>

#include <OffloadAPI.h>

#include <cassert>
#include <unordered_map>
#include <vector>

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

using PlatformWithDevStorageType =
    std::unordered_map<ol_platform_handle_t, std::vector<ol_device_handle_t>>;

// Contiguous global storage of platform handlers and device handles (grouped by
// platform) for a backend.
struct OffloadTopology {
  OffloadTopology() : OlBackend(OL_PLATFORM_BACKEND_UNKNOWN) {}
  OffloadTopology(ol_platform_backend_t OlBackend) : OlBackend(OlBackend) {}

  void set_backend(ol_platform_backend_t B) { OlBackend = B; }

  // Platforms for this backend
  range_view<ol_platform_handle_t> platforms() const {
    return {Platforms.data(), Platforms.size()};
  }

  // Devices for a specific platform (platform_id is index into Platforms)
  range_view<ol_device_handle_t> devicesForPlatform(size_t PlatformId) const {
    if (PlatformId >= PlatformDevices.size())
      return {nullptr, 0};
    return PlatformDevices[PlatformId].first;
  }

  size_t getFirstDeviceIndexForPlatform(size_t PlatformId) const {
    assert(PlatformId < PlatformDevices.size());
    return PlatformDevices[PlatformId].second;
  }

  // Register new platform and devices into this topology under that platform.
  void
  registerNewPlatformsAndDevices(PlatformWithDevStorageType &PlatformsAndDev,
                                 size_t TotalDevCount) {
    if (!PlatformsAndDev.size())
      return;

    Platforms.reserve(PlatformsAndDev.size());
    PlatformDevices.reserve(Platforms.size());
    Devices.reserve(TotalDevCount);

    for (auto &[NewPlatform, NewDevs] : PlatformsAndDev) {
      Platforms.push_back(NewPlatform);

      size_t StartIdx = Devices.size();
      range_view<ol_device_handle_t> R{Devices.data() + Devices.size(),
                                       NewDevs.size()};
      Devices.insert(Devices.end(), NewDevs.begin(), NewDevs.end());
      PlatformDevices.push_back({R, StartIdx});
    }

    assert(TotalDevCount == Devices.size());
  }

  ol_platform_backend_t backend() { return OlBackend; }

private:
  ol_platform_backend_t OlBackend = OL_PLATFORM_BACKEND_UNKNOWN;

  // Platforms and devices belonging to this backend (flattened)
  std::vector<ol_platform_handle_t> Platforms;
  std::vector<ol_device_handle_t> Devices; // sorted by platform

  // Vector holding range of devices for each platform (index is platform index
  // within Platforms)
  std::vector<std::pair<range_view<ol_device_handle_t>, size_t>>
      PlatformDevices; // PlatformDevices.size() == Platforms.size()
};

// Initialize the topologies by calling olIterateDevices.
void discoverOffloadDevices();

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_OFFLOAD_TOPOLOGY
