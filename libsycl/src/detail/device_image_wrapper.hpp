//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_DEVICE_BINARY_WRAPPER
#define _LIBSYCL_DEVICE_BINARY_WRAPPER

#include <sycl/__impl/detail/config.hpp>

#include <detail/device_binary_structures.hpp>

#include <OffloadAPI.h>

#include <memory>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

class DeviceImageWrapper {
public:
public:
  DeviceImageWrapper(const __sycl_tgt_device_image &Bin);
  // Explicitly delete copy constructor/operator= to avoid unintentional copies.
  DeviceImageWrapper(const DeviceImageWrapper &) = delete;
  DeviceImageWrapper &operator=(const DeviceImageWrapper &) = delete;

  DeviceImageWrapper(DeviceImageWrapper &&) = default;
  DeviceImageWrapper &operator=(DeviceImageWrapper &&) = default;

  ~DeviceImageWrapper() {}

  const __sycl_tgt_device_image &getRawData() const { return *get(); }

  size_t getSize() const {
    return static_cast<size_t>(MBin->ImageStart - MBin->ImageEnd);
  }

protected:
  const __sycl_tgt_device_image *get() const { return MBin; }

  __sycl_tgt_device_image const *MBin{};
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_DEVICE_BINARY_WRAPPER
