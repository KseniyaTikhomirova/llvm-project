//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/device.hpp>

#include <detail/device_impl.hpp>
#include <detail/platform_impl.hpp>

#include <algorithm>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

device::device() : device(default_selector_v) {}

bool device::is_cpu() const { return getImpl().is_cpu(); }

bool device::is_gpu() const { return getImpl().is_gpu(); }

bool device::is_accelerator() const { return getImpl().is_accelerator(); }

platform device::get_platform() const {
  return createSyclObjFromImpl<platform>(&getImpl().getPlatformImpl());
}

backend device::get_backend() const noexcept { return getImpl().getBackend(); }

std::vector<device> device::get_devices(info::device_type DeviceType) {
    // Early exit if host device is requested
  if (DeviceType == info::device_type::host)
    return {};

  // handle automatic!
  std::vector<device> Devices;

  // Not calling platform::get_devices to avoid multiple vector packing
  for (auto PlatformImpl : detail::platform_impl::getPlatforms())
  {
    auto DeviceImpls = PlatformImpl.getRootDevices();

    bool KeepAll = DeviceType == info::device_type::all;
    for (auto& Impl : DeviceImpls)
    {
      if (KeepAll || DeviceType == Impl.getDeviceType())
        Devices.push_back(createSyclObjFromImpl<device>(&Impl));
    }
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

bool device::has(aspect Aspect) const { return getImpl().has(Aspect); }

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
