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

#include <detail/platform_impl.hpp>
#include <detail/offload/offload_utils.hpp>

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
  /// UR device instance.
  //
  // Must be called through `platform_impl` method only. `private_tag` ensures that is true.
  explicit device_impl(ol_device_handle_t Device, platform_impl &Platform, private_tag)
      : MOffloadDevice(Device), MPlatform(Platform) {}
  ~device_impl() = default;

  info::device_type getDeviceType() const {
    // TODO: get info
    return info::device_type::gpu;
  }

  bool is_cpu() const { return getDeviceType() == info::device_type::cpu; }

  bool is_gpu() const { return getDeviceType() == info::device_type::gpu; }

  bool is_accelerator() const { return getDeviceType() == info::device_type::accelerator;  }

  backend getBackend() const;

  platform_impl &getPlatformImpl() const { return MPlatform; }

  bool has(aspect Aspect) const;

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

    using DescType = typename Param::return_type;
    return DescType{};
    // if constexpr (std::is_same_v<DescType, std::string>) {
    //   std::string Result;
    //   Result.resize(ExpectedSize - 1);
    //   call_and_throw(olGetDeviceInfo, MOffloadDevice, olInfo, ExpectedSize,
    //                  Result.data());
    //   return Result;
    // } else {
    //   assert((sizeof(DescType) == ExpectedSize) &&
    //          "Size of info descriptor reported by backend doesn't match with
    //          " "expected.");
    //   DescType Result{};
    //   call_and_throw(olGetDeviceInfo, MOffloadDevice, olInfo, sizeof(Result),
    //                  &Result.data());
    //  // ktikhomi add convertion, we don't have 1:1 values mapping
    //   return Result;
    // }
  }

private:
  ol_device_handle_t MOffloadDevice = {};
  platform_impl &MPlatform;
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_DEVICE_IMPL
