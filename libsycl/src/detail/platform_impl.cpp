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
  return {};
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
