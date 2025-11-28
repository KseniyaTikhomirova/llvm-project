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

#include <detail/common.hpp>

#include <OffloadAPI.h>

#include <cassert>
#include <map>
#include <unordered_map>
#include <vector>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

using PlatformWithDevStorageType =
    std::unordered_map<ol_platform_handle_t, std::multimap<ol_device_type_t, ol_device_handle_t>>;

// Contiguous global storage of platform handlers and device handles (grouped by
// platform) for a backend.
struct OffloadTopology {
  OffloadTopology() : MBackend(OL_PLATFORM_BACKEND_UNKNOWN) {}
  OffloadTopology(ol_platform_backend_t OlBackend) : MBackend(OlBackend) {}

  void set_backend(ol_platform_backend_t B) { MBackend = B; }

  // Platforms for this backend
  range_view<const ol_platform_handle_t> platforms() const {
    return {MPlatforms.data(), MPlatforms.size()};
  }

  // Devices for a specific platform (platform_id is index into Platforms)
  range_view<std::pair<ol_device_handle_t, ol_device_type_t>> devicesForPlatform(size_t PlatformId) const {
    if (PlatformId >= MDevRangePerPlatformId.size())
      return {nullptr, 0};
    return MDevRangePerPlatformId[PlatformId];
  }

  // Register new platform and devices into this topology under that platform.
  void
  registerNewPlatformsAndDevices(PlatformWithDevStorageType &PlatformsAndDev,
                                 size_t TotalDevCount);

  ol_platform_backend_t backend() const { return MBackend; }

private:
  ol_platform_backend_t MBackend = OL_PLATFORM_BACKEND_UNKNOWN;

  // Platforms and devices belonging to this backend (flattened)
  std::vector<ol_platform_handle_t> MPlatforms;
  std::vector<std::pair<ol_device_handle_t, ol_device_type_t>> MDevices; // sorted by platform and type

  // Vector holding range of devices for each platform (index is platform index
  // within Platforms)
  std::vector<range_view<std::pair<ol_device_handle_t, ol_device_type_t>>>
      MDevRangePerPlatformId; // PlatformDevices.size() == Platforms.size()
};

// Initialize the topologies by calling olIterateDevices.
void discoverOffloadDevices();

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_OFFLOAD_TOPOLOGY
