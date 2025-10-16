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

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

platform_impl*
platform_impl::getOrMakePlatformImpl(ol_platform_handle_t Platform) {
  platform_impl *Result = nullptr;
  {
    const std::lock_guard<std::mutex> Guard(
        GlobalHandler::instance().getPlatformsMutex());

    std::vector<platform_impl *> &PlatformCache =
        GlobalHandler::instance().getPlatforms();

    for (const auto &PlatImpl : PlatformCache) {
      if (PlatImpl->getHandleRef() == Platform)
        return PlatImpl;
    }

    // GlobalHandler is responsible of destruction at the end of program.
    Result = new platform_impl(Platform);
    PlatformCache.emplace_back(Result);
  }

  return Result;
}

std::vector<platform> platform_impl::getPlatforms() {
  std::vector<platform> Platforms;
  return Platforms;
}

platform_impl::platform_impl(ol_platform_handle_t Platform)
    : MPlatform(Platform) {
        // TODO
    }

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL