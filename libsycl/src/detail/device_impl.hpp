//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_DEVICE_IMPL
#define _LIBSYCL_DEVICE_IMPL

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/device.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {
    class device_impl : public std::enable_shared_from_this<device_impl> {
  struct private_tag {
    explicit private_tag() = default;
  };

public:
  /// Constructs a SYCL device instance using the provided
  /// UR device instance.
  //
  // Must be called through `platform_impl::getOrMakeDeviceImpl` only.
  // `private_tag` ensures that is true.
  explicit device_impl(ur_device_handle_t Device, platform_impl &Platform,
                       private_tag);

  ~device_impl();

  bool is_cpu() const {  return false;  }

  bool is_gpu() const {    return true;  }

  bool is_accelerator() const { return false; }

  backend getBackend() const { return MPlatform.getBackend(); }

  platform_impl &getPlatformImpl() const { return MPlatform; }

private:
  ol_device_handle_t MDevice ={};
  platform_impl &MPlatform;
}; 

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_DEVICE_IMPL