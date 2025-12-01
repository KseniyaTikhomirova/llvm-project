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

template <typename Param>
detail::is_platform_info_desc_t<Param> platform::get_info() const {
  return detail::is_platform_info_desc_t{};
}

template _LIBSYCL_EXPORT std::vector<device> device::create_sub_devices<
    info::partition_property::partition_by_affinity_domain>(
    info::partition_affinity_domain AffinityDomain) const;

#define _LIBSYCL_EXPORT_GET_INFO(Desc)                                         \
  template _LIBSYCL_EXPORT detail::is_device_info_desc_t<info::device::Desc>   \
  device::get_info<info::device::Desc>() const;
_LIBSYCL_EXPORT_GET_INFO(device_type)
_LIBSYCL_EXPORT_GET_INFO(vendor_id)
// _LIBSYCL_EXPORT_GET_INFO(max_compute_units)
// _LIBSYCL_EXPORT_GET_INFO(max_work_item_dimensions)
// //_LIBSYCL_EXPORT_GET_INFO(max_work_item_sizes)
// _LIBSYCL_EXPORT_GET_INFO(max_work_group_size)
// _LIBSYCL_EXPORT_GET_INFO(max_num_sub_groups)
// _LIBSYCL_EXPORT_GET_INFO(sub_group_sizes)
// _LIBSYCL_EXPORT_GET_INFO(preferred_vector_width_char)
// _LIBSYCL_EXPORT_GET_INFO(preferred_vector_width_short)
// _LIBSYCL_EXPORT_GET_INFO(preferred_vector_width_int)
// _LIBSYCL_EXPORT_GET_INFO(preferred_vector_width_long)
// _LIBSYCL_EXPORT_GET_INFO(preferred_vector_width_float)
// _LIBSYCL_EXPORT_GET_INFO(preferred_vector_width_double)
// _LIBSYCL_EXPORT_GET_INFO(preferred_vector_width_half)
// _LIBSYCL_EXPORT_GET_INFO(native_vector_width_char)
// _LIBSYCL_EXPORT_GET_INFO(native_vector_width_short)
// _LIBSYCL_EXPORT_GET_INFO(native_vector_width_int)
// _LIBSYCL_EXPORT_GET_INFO(native_vector_width_long)
// _LIBSYCL_EXPORT_GET_INFO(native_vector_width_float)
// _LIBSYCL_EXPORT_GET_INFO(native_vector_width_double)
// _LIBSYCL_EXPORT_GET_INFO(native_vector_width_half)
// _LIBSYCL_EXPORT_GET_INFO(max_clock_frequency)
// _LIBSYCL_EXPORT_GET_INFO(address_bits)
// _LIBSYCL_EXPORT_GET_INFO(max_mem_alloc_size)
// _LIBSYCL_EXPORT_GET_INFO(image_support)
// _LIBSYCL_EXPORT_GET_INFO(max_read_image_args)
// _LIBSYCL_EXPORT_GET_INFO(max_write_image_args)
// _LIBSYCL_EXPORT_GET_INFO(image2d_max_height)
// _LIBSYCL_EXPORT_GET_INFO(image2d_max_width)
// _LIBSYCL_EXPORT_GET_INFO(image3d_max_height)
// _LIBSYCL_EXPORT_GET_INFO(image3d_max_width)
// _LIBSYCL_EXPORT_GET_INFO(image3d_max_depth)
// _LIBSYCL_EXPORT_GET_INFO(image_max_buffer_size)
// _LIBSYCL_EXPORT_GET_INFO(max_samplers)
// _LIBSYCL_EXPORT_GET_INFO(max_parameter_size)
// _LIBSYCL_EXPORT_GET_INFO(mem_base_addr_align)
// _LIBSYCL_EXPORT_GET_INFO(half_fp_config)
// _LIBSYCL_EXPORT_GET_INFO(single_fp_config)
// _LIBSYCL_EXPORT_GET_INFO(double_fp_config)
// _LIBSYCL_EXPORT_GET_INFO(global_mem_cache_type)
// _LIBSYCL_EXPORT_GET_INFO(global_mem_cache_line_size)
// _LIBSYCL_EXPORT_GET_INFO(global_mem_cache_size)
// _LIBSYCL_EXPORT_GET_INFO(global_mem_size)
// _LIBSYCL_EXPORT_GET_INFO(max_constant_buffer_size)
// _LIBSYCL_EXPORT_GET_INFO(max_constant_args)
// _LIBSYCL_EXPORT_GET_INFO(local_mem_type)
// _LIBSYCL_EXPORT_GET_INFO(local_mem_size)
// _LIBSYCL_EXPORT_GET_INFO(error_correction_support)
// _LIBSYCL_EXPORT_GET_INFO(host_unified_memory)
// _LIBSYCL_EXPORT_GET_INFO(atomic_memory_order_capabilities)
// _LIBSYCL_EXPORT_GET_INFO(atomic_fence_order_capabilities)
// _LIBSYCL_EXPORT_GET_INFO(atomic_memory_scope_capabilities)
// _LIBSYCL_EXPORT_GET_INFO(atomic_fence_scope_capabilities)
// _LIBSYCL_EXPORT_GET_INFO(profiling_timer_resolution)
// _LIBSYCL_EXPORT_GET_INFO(is_endian_little)
// _LIBSYCL_EXPORT_GET_INFO(is_available)
// _LIBSYCL_EXPORT_GET_INFO(is_compiler_available)
// _LIBSYCL_EXPORT_GET_INFO(is_linker_available)
// _LIBSYCL_EXPORT_GET_INFO(execution_capabilities)
// _LIBSYCL_EXPORT_GET_INFO(queue_profiling)
// _LIBSYCL_EXPORT_GET_INFO(built_in_kernels)
// _LIBSYCL_EXPORT_GET_INFO(built_in_kernel_ids)
// _LIBSYCL_EXPORT_GET_INFO(platform)
// _LIBSYCL_EXPORT_GET_INFO(name)
// _LIBSYCL_EXPORT_GET_INFO(vendor)
// _LIBSYCL_EXPORT_GET_INFO(driver_version)
// _LIBSYCL_EXPORT_GET_INFO(profile)
// _LIBSYCL_EXPORT_GET_INFO(version)
// _LIBSYCL_EXPORT_GET_INFO(backend_version)
// _LIBSYCL_EXPORT_GET_INFO(aspects)
// _LIBSYCL_EXPORT_GET_INFO(extensions)
// _LIBSYCL_EXPORT_GET_INFO(printf_buffer_size)
// _LIBSYCL_EXPORT_GET_INFO(preferred_interop_user_sync)
// _LIBSYCL_EXPORT_GET_INFO(parent_device)
// _LIBSYCL_EXPORT_GET_INFO(partition_max_sub_devices)
// _LIBSYCL_EXPORT_GET_INFO(partition_properties)
// _LIBSYCL_EXPORT_GET_INFO(partition_affinity_domains)
// _LIBSYCL_EXPORT_GET_INFO(partition_type_property)
// _LIBSYCL_EXPORT_GET_INFO(partition_type_affinity_domain)
#undef _LIBSYCL_EXPORT_GET_INFO

_LIBSYCL_END_NAMESPACE_SYCL
