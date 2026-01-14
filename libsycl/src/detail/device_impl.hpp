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

#include <detail/offload/offload_utils.hpp>
#include <detail/platform_impl.hpp>

#include <OffloadAPI.h>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

class device_impl {
  struct private_tag {
    explicit private_tag() = default;
  };
  friend class platform_impl;

public:
  /// Constructs a SYCL device instance using the provided
  /// offload device instance.
  explicit device_impl(ol_device_handle_t Device, platform_impl &Platform,
                       private_tag)
      : MOffloadDevice(Device), MPlatform(Platform) {}
  ~device_impl() = default;

  /// Queries device type from offloading runtime
  info::device_type getDeviceType() const;

  /// Check if device is a CPU device
  bool is_cpu() const;

  /// Check if device is a GPU device
  bool is_gpu() const;

  /// Check if device is an accelerator device
  bool is_accelerator() const;

  /// Returns the backend associated with this device.
  backend getBackend() const noexcept;

  /// Returns the implementation class object of platform associated with this
  /// device.
  platform_impl &getPlatformImpl() const { return MPlatform; }

  /// Check if this device has a specified aspect
  bool has(aspect Aspect) const;

  /// Queries this device for information requested by the template parameter
  /// param
  template <typename Param> typename Param::return_type get_info() const {
    using namespace info::device;
    using Map = info_ol_mapping<ol_device_info_t>;

    constexpr ol_device_info_t olInfo = map_info_desc<Param, ol_device_info_t>(
        Map::M<device_type>{OL_DEVICE_INFO_TYPE},
        Map::M<name>{OL_DEVICE_INFO_NAME},
        Map::M<vendor>{OL_DEVICE_INFO_VENDOR},
        Map::M<driver_version>{OL_DEVICE_INFO_DRIVER_VERSION});

    size_t ExpectedSize = 0;
    call_and_throw(olGetDeviceInfoSize, MOffloadDevice, olInfo, &ExpectedSize);

    if constexpr (std::is_same_v<typename Param::return_type, std::string>) {
      std::string Result;
      Result.resize(ExpectedSize - 1);
      call_and_throw(olGetDeviceInfo, MOffloadDevice, olInfo, ExpectedSize,
                     Result.data());
      return Result;
    } else if constexpr (olInfo == OL_DEVICE_INFO_TYPE) {
      assert((sizeof(DescType) == ExpectedSize) &&
             "Size of info descriptor reported by backend doesn't match with "
             "expected.");
      ol_device_type_t olType{};
      call_and_throw(olGetDeviceInfo, MOffloadDevice, olInfo, sizeof(olType),
                     &olType);
      return convertDeviceTypeToSYCL(olType);
    } else
      static_assert(false && "Info descriptor is not properly supported");
  }

  ol_device_handle_t getOLHandle() { return MOffloadDevice; }

private:
  ol_device_handle_t MOffloadDevice = {};
  platform_impl &MPlatform;
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_DEVICE_IMPL
