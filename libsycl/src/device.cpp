//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/device.hpp>

#include <detail/device_impl.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

device::device() : device(default_selector_v) {}

device::device(const device_selector &deviceSelector) {
  *this = deviceSelector.select_device();
}

bool device::is_cpu() const { return impl.is_cpu(); }

bool device::is_gpu() const { return impl.is_gpu(); }

bool device::is_accelerator() const { return impl.is_accelerator(); }

platform device::get_platform() const { 
  return createSyclObjFromImpl<platform>(impl.getPlatformImpl());
}

backend device::get_backend() const noexcept { return impl.getBackend(); }

std::vector<device> device::get_devices(info::device_type deviceType) {
  std::vector<device> Devices;

  auto Platforms = platform::get_platforms();
  for (const auto &Platform : Platforms) {
    auto PlatformDevices = Platform.get_devices(deviceType);
    std::transform(PlatformDevices.begin(), PlatformDevices.end(), std::back_inserter(Devices));
  }

  return Devices;
}

_LIBSYCL_END_NAMESPACE_SYCL
