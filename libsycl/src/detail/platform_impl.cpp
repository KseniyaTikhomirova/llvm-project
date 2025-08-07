//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/detail/config.hpp> // namespace macro

#include <detail/adapter_impl.hpp>
#include <detail/platform_impl.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

std::vector<platform> platform_impl::getPlatforms() {
  // See which platform we want to be served by which adapter.
  // There should be just one adapter serving each backend.
  std::vector<adapter_impl *> &Adapters = adapter_impl::getAdapters();
  std::vector<std::pair<platform, adapter_impl *>> PlatformsWithAdapter;

  // Then check backend-specific adapters
  for (auto &Adapter : Adapters) {
    const auto &AdapterPlatforms = getAdapterPlatforms(*Adapter);
    for (const auto &P : AdapterPlatforms) {
      PlatformsWithAdapter.push_back({P, Adapter});
    }
  }

  // For the selected platforms register them with their adapters
  std::vector<platform> Platforms;
  for (auto &Platform : PlatformsWithAdapter) {
    auto &Adapter = Platform.second;
    std::lock_guard<std::mutex> Guard(*Adapter->getAdapterMutex());
    Adapter->getPlatformId(getSyclObjImpl(Platform.first)->getHandleRef());
    Platforms.push_back(Platform.first);
  }

  // This initializes a function-local variable whose destructor is invoked as
  // the SYCL shared library is first being unloaded.
  GlobalHandler::registerStaticVarShutdownHandler();

  return Platforms;
}

platform_impl::platform_impl(ur_platform_handle_t Platform,
                             adapter_impl &Adapter)
    : MPlatform(Platform), MAdapter(Adapter) {

  // Find out backend of the platform
  ur_backend_t UrBackend = UR_BACKEND_UNKNOWN;
  Adapter.call_nocheck<UrApiKind::urPlatformGetInfo>(
      APlatform, UR_PLATFORM_INFO_BACKEND, sizeof(ur_backend_t), &UrBackend,
      nullptr);
  MBackend = convertUrBackend(UrBackend);
}

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
