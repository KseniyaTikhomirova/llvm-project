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

template <info::partition_property prop>
std::vector<device> device::create_sub_devices(size_t ComputeUnits) const {
  throw exception(make_error_code(errc::feature_not_supported),
                    "Partitioning is not supported.");
}

template _LIBSYCL_EXPORT std::vector<device>
device::create_sub_devices<info::partition_property::partition_equally>(
    size_t ComputeUnits) const;

template <info::partition_property prop>
std::vector<device>
device::create_sub_devices(const std::vector<size_t> &Counts) const {
    throw exception(make_error_code(errc::feature_not_supported),
                    "Partitioning is not supported.");
}

template _LIBSYCL_EXPORT std::vector<device>
device::create_sub_devices<info::partition_property::partition_by_counts>(
    const std::vector<size_t> &Counts) const;

template <info::partition_property prop>
std::vector<device> device::create_sub_devices(
    info::partition_affinity_domain AffinityDomain) const {
    throw exception(make_error_code(errc::feature_not_supported),
                    "Partitioning is not supported.");
}

template _LIBSYCL_EXPORT std::vector<device> device::create_sub_devices<
    info::partition_property::partition_by_affinity_domain>(
    info::partition_affinity_domain AffinityDomain) const;

template <typename Param>
detail::is_device_info_desc_t<Param> device::get_info() const {
  return getImpl().get_info<Param>();
}

#define _LIBSYCL_EXPORT_GET_INFO(Desc)                                         \
  template _LIBSYCL_EXPORT detail::is_device_info_desc_t<info::device::Desc>   \
  device::get_info<info::device::Desc>() const;
_LIBSYCL_EXPORT_GET_INFO(device_type)
_LIBSYCL_EXPORT_GET_INFO(name)
_LIBSYCL_EXPORT_GET_INFO(vendor)
_LIBSYCL_EXPORT_GET_INFO(driver_version)
#undef _LIBSYCL_EXPORT_GET_INFO

_LIBSYCL_END_NAMESPACE_SYCL
