//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/detail/config.hpp> // namespace macro

#include <detail/global_handler.hpp>
#include <detail/platform_impl.hpp>
#include <detail/ur/adapter_impl.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

platform_impl *
platform_impl::getOrMakePlatformImpl(ur_platform_handle_t UrPlatform,
                                     adapter_impl &Adapter) {
  platform_impl *Result = nullptr;
  {
    const std::lock_guard<std::mutex> Guard(
        GlobalHandler::instance().getPlatformsMutex());

    std::vector<platform_impl *> &PlatformCache =
        GlobalHandler::instance().getPlatforms();

    for (const auto &PlatImpl : PlatformCache) {
      if (PlatImpl->getHandleRef() == UrPlatform)
        return PlatImpl;
    }

    // GlobalHandler is responsible of destruction at the end of program.
    Result = new platform_impl(UrPlatform, Adapter, Adapter.getBackend());
    PlatformCache.emplace_back(Result);
  }

  return Result;
}

std::vector<platform> platform_impl::getPlatforms() {
  // See which platform we want to be served by which adapter.
  // There should be just one adapter serving each backend.
  std::vector<adapter_impl *> &Adapters = adapter_impl::getAdapters();
  std::vector<std::pair<platform, adapter_impl *>> PlatformsWithAdapter;

  // Then check backend-specific adapters
  for (auto &Adapter : Adapters) {
    const auto &AdapterPlatforms = getAdapterPlatforms(Adapter);
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

  return Platforms;
}

platform_impl::platform_impl(ur_platform_handle_t Platform,
                             adapter_impl &Adapter, const backend &Backend)
    : MPlatform(Platform), MAdapter(Adapter), MBackend(Backend) {}

static bool IsBannedPlatform(platform Platform) {
  // The NVIDIA OpenCL platform is currently not compatible with DPC++
  // since it is only 1.2 but gets selected by default in many systems
  // There is also no support on the PTX backend for OpenCL consumption,
  // and there have been some internal reports.
  // To avoid problems on default users and deployment of DPC++ on platforms
  // where CUDA is available, the OpenCL support is disabled.
  //
  // There is also no support for the AMD HSA backend for OpenCL consumption,
  // as well as reported problems with device queries, so AMD OpenCL support
  // is disabled as well.
  //
  // auto IsMatchingOpenCL = [](platform Platform, const std::string_view name)
  // {
  //   const bool HasNameMatch = Platform.get_info<info::platform::name>().find(
  //                                 name) != std::string::npos;
  //   const auto Backend = detail::getSyclObjImpl(Platform)->getBackend();
  //   const bool IsMatchingOCL = (HasNameMatch && Backend == backend::opencl);
  //   return IsMatchingOCL;
  // };
  // return IsMatchingOpenCL(Platform, "NVIDIA CUDA") ||
  //        IsMatchingOpenCL(Platform, "AMD Accelerated Parallel Processing");
  return false;
}

std::vector<platform>
platform_impl::getAdapterPlatforms(adapter_impl *&Adapter) {
  std::vector<platform> Platforms;

  auto UrPlatforms = Adapter->getUrPlatforms();
  if (UrPlatforms.empty()) {
    return Platforms;
  }

  for (const auto &UrPlatform : UrPlatforms) {
    platform Platform = detail::createSyclObjFromImpl<platform>(
        getOrMakePlatformImpl(UrPlatform, *Adapter));
    if (IsBannedPlatform(Platform))
      continue;
    bool HasAnyDevices = false;
    // Platform.get_devices() increments the device count for the platform
    // and if the platform is banned (like OpenCL for AMD), it can cause
    // incorrect device numbering, when used with ONEAPI_DEVICE_SELECTOR.
    //   HasAnyDevices = !Platform.get_devices(info::device_type::all).empty();

    // The SYCL spec says that a platform has one or more devices. ( SYCL
    // 2020 4.6.2 ) If we have an empty platform, we don't report it back
    // from platform::get_platforms().
    if (HasAnyDevices) {
      Platforms.push_back(Platform);
    }
  }
  return Platforms;
}

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
