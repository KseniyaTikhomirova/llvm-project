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
  class device_impl {

public:
  /// Constructs a SYCL device instance using the provided
  /// UR device instance.
  //
  // Must be called through `platform_impl::getOrMakeDeviceImpl` only.
  // `private_tag` ensures that is true.
  explicit device_impl(ol_device_handle_t Device, platform_impl &Platform);

  ~device_impl();

  info::device_type getDeviceType()
  {
    // get info
    return info::device_type::gpu;
  }

  bool is_cpu() const { return getDeviceType() == info::device_type::cpu; }

  bool is_gpu() const { return getDeviceType() == info::device_type::gpu; }

  bool is_accelerator() const { return getDeviceType() == info::device_type::accelerator;  }

  backend getBackend() const { return MPlatform.getBackend(); }

  platform_impl &getPlatformImpl() const { return MPlatform; }

  template <typename Param> typename Param::return_type get_info() const {
    using namespace info::device;
    using Map = info_ol_mapping<ol_device_info_t>;

    constexpr ol_device_info_t olInfo = map_info_desc<Param, ol_device_info_t>(
        Map::M<device_type>{OL_DEVICE_INFO_TYPE},
        Map::M<name>{OL_DEVICE_INFO_NAME},
        Map::M<vendor>{OL_DEVICE_INFO_VENDOR},
        Map::M<driver_version>{OL_DEVICE_INFO_DRIVER_VERSION});

    size_t ExpectedSize = 0;
    call_and_throw(olGetPlatformInfoSize, MOffloadPlatform, olInfo,
                   &ExpectedSize);

    using DescType = Param::return_type;
    if constexpr (std::is_same_v<DescType, std::string>) {
      std::string Result;
      Result.resize(ExpectedSize - 1);
      call_and_throw(olGetDeviceInfo, MOffloadPlatform, olInfo, ExpectedSize,
                     Result.data());
      return Result;
    } else {
      assert((sizeof(DescType) == ExpectedSize) &&
             "Size of info descriptor reported by backend doesn't match with "
             "expected.") DescType Result{};
      call_and_throw(olGetDeviceInfo, MOffloadPlatform, olInfo, sizeof(result),
                     &Result.data());
      return Result;
    }
  }

private:
  ol_device_handle_t MDevice ={};
  platform_impl &MPlatform;
}; 

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_DEVICE_IMPL