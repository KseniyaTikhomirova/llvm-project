//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/detail/config.hpp> // namespace macro
#include <sycl/__impl/backend.hpp> // sycl::backend
#include <sycl/__impl/platform.hpp> // sycl::platform

#include <memory> // std::enable_shared_from_this
#include <vector> // std::vector

#ifndef _LIBSYCL_PLATFORM_IMPL
#define _LIBSYCL_PLATFORM_IMPL

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

class platform_impl : public std::enable_shared_from_this<platform_impl> {
public:

  /// Returns the backend associated with this platform.
  backend getBackend() const noexcept { return MBackend; }

  /// Returns all SYCL platforms from all backends that are available in the
  /// system.
  static std::vector<platform> getPlatforms();

private:
  ur_platform_handle_t MPlatform {};
  backend MBackend;
};

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_PLATFORM_IMPL
