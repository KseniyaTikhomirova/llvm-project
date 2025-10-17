//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/offload/offload_topology.hpp>
#include <detail/offload/offload_utils.hpp>
#include <detail/global_handler.hpp>

#include <array>
#include <vector>
#include <unordered_map>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

void discoverOffloadDevices() {
  static std::once_flag DiscoverOnce;
  std::call_once(DiscoverOnce, [&]() {
    call_and_throw(olInit);

    using StorageType = std::array<std::unordered_map<ol_platform_handle_t,
                                  std::vector<ol_device_handle_t>>,
               OL_PLATFORM_BACKEND_LAST>;
    StorageType Mapping;
    //TODO check
    call_nocheck(olIterateDevices,
        [](ol_device_handle_t Dev, void *User) -> bool {
          auto *Data = static_cast<StorageType*>(User);
          ol_platform_handle_t Plat = nullptr;
          ol_result_t Res =
              call_nocheck(olGetDeviceInfo,
                  Dev, OL_DEVICE_INFO_PLATFORM, sizeof(Plat), &Plat);
          if (Res != OL_SUCCESS)
            return true; // continue

          ol_platform_backend_t OlBackend = OL_PLATFORM_BACKEND_UNKNOWN;
          Res = call_nocheck(olGetPlatformInfo,
              Plat, OL_PLATFORM_INFO_BACKEND, sizeof(OlBackend), &OlBackend);
          if (Res != OL_SUCCESS)
            return true; // continue

          if (OL_PLATFORM_BACKEND_HOST == OlBackend ||
              OL_PLATFORM_BACKEND_UNKNOWN == OlBackend)
            return true; // skip host/unknown backend

          // Ensure backend index fits into array size
          if (OlBackend >= OL_PLATFORM_BACKEND_LAST)
            return true;

          (*Data)[static_cast<size_t>(OlBackend)][Plat].push_back(Dev);
          return true;
        },
        &Mapping);
    // Now register all platforms and devices into the topologies
    auto &OffloadTopologies = GlobalHandler::instance().getOffloadTopologies();
    for (size_t I = 0; I < OL_PLATFORM_BACKEND_LAST; ++I) {
      OffloadTopology &Topo = OffloadTopologies[I];
      Topo.set_backend(static_cast<ol_platform_backend_t>(I));
      for (auto &PltAndDevs : Mapping[I])
        Topo.registerNewPlatformAndDevices(PltAndDevs.first,
                                               std::move(PltAndDevs.second));
    }
  });
}

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL