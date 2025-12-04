//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/device_impl.hpp>
#include <detail/platform_impl.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

bool device_impl::has(aspect Aspect) const {
    switch (Aspect) {
    case (aspect::cpu):
      return is_cpu();
    case (aspect::gpu):
      return is_gpu();
    case (aspect::accelerator):
      return is_accelerator();
    case (aspect::custom):
      return false;
    case (aspect::emulated):
      return false;
    case (aspect::host_debuggable):
      return false;
    default:
      // Other aspects are not implemented yet
      return false;
    }
  }

  backend device_impl::getBackend() const { return MPlatform.getBackend(); }

    } // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
