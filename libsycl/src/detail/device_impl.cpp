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

info::device_type device_impl::getDeviceType() const {
  return get_info<info::device::device_type>();
}

bool device_impl::is_cpu() const {
  return getDeviceType() == info::device_type::cpu;
}

bool device_impl::is_gpu() const {
  return getDeviceType() == info::device_type::gpu;
}

bool device_impl::is_accelerator() const {
  return getDeviceType() == info::device_type::accelerator;
}

backend device_impl::getBackend() const { return MPlatform.getBackend(); }

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
