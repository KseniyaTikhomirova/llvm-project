//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of SYCL 2020 device info types.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_INFO_DEVICE_HPP
#define _LIBSYCL___IMPL_INFO_DEVICE_HPP

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/info/desc_base.hpp>
#include <sycl/__impl/info/device_type.hpp>

#include <cstdint>
#include <string>
#include <vector>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class device;

namespace detail {
template <typename T>
using is_device_info_desc_t = typename is_info_desc<T, device>::return_type;
} // namespace detail

// A.3. Device information descriptors
namespace info {

enum class partition_property : std::intptr_t {
  no_partition = 0,
  partition_equally,
  partition_by_counts,
  partition_by_affinity_domain
};

enum class partition_affinity_domain : std::intptr_t {
  not_applicable = 0,
  numa,
  L4_cache,
  L3_cache,
  L2_cache,
  L1_cache,
  next_partitionable
};

namespace device {
// 4.6.4.4. Information descriptors

struct device_type : detail::info_desc_tag<device_type, sycl::device> {
  using return_type = sycl::info::device_type;
};
struct vendor_id : detail::info_desc_tag<vendor_id, sycl::device> {
  using return_type = std::uint32_t;
};
// struct max_compute_units : detail::info_desc_tag<max_compute_units,
// sycl::device> {
//   using return_type = std::uint32_t;
// };
// struct max_work_item_dimensions :
// detail::info_desc_tag<max_work_item_dimensions, sycl::device> {
//   using return_type = std::uint32_t;
// };
// struct max_work_item_sizes : detail::info_desc_tag<max_work_item_sizes,
// sycl::device> {
//   using return_type = range<Dimensions>;
// };
// //to be specialized?
// struct max_work_item_sizes : detail::info_desc_tag<max_work_item_sizes,
// sycl::device> {
//   using return_type = range<Dimensions>;
// };
// struct max_work_group_size : detail::info_desc_tag<max_work_group_size,
// sycl::device> {
//   using return_type = std::size_t;
// };
// struct max_num_sub_groups : detail::info_desc_tag<max_num_sub_groups,
// sycl::device> {
//   using return_type = std::uint32_t;
// };
// struct sub_group_sizes : detail::info_desc_tag<sub_group_sizes, sycl::device>
// {
//   using return_type = std::vector<std::size_t>;
// };
// struct preferred_vector_width_char :
// detail::info_desc_tag<preferred_vector_width_char, sycl::device> {
//   using return_type = std::uint32_t;
// };
// struct preferred_vector_width_short :
// detail::info_desc_tag<preferred_vector_width_short, sycl::device> {
//   using return_type = std::uint32_t;
// };
// struct preferred_vector_width_int :
// detail::info_desc_tag<preferred_vector_width_int, sycl::device> {
//   using return_type = std::uint32_t;
// };
// struct preferred_vector_width_long :
// detail::info_desc_tag<preferred_vector_width_long, sycl::device> {
//   using return_type = std::uint32_t;
// };
// struct preferred_vector_width_float :
// detail::info_desc_tag<preferred_vector_width_float, sycl::device> {
//   using return_type = std::uint32_t;
// };
// struct preferred_vector_width_double :
// detail::info_desc_tag<preferred_vector_width_double, sycl::device> {
//   using return_type = std::uint32_t;
// };
// struct preferred_vector_width_half :
// detail::info_desc_tag<preferred_vector_width_half, sycl::device> {
//   using return_type = std::uint32_t;
// };
// struct native_vector_width_char :
// detail::info_desc_tag<native_vector_width_char, sycl::device> {
//   using return_type = std::uint32_t;
// };
// struct native_vector_width_short :
// detail::info_desc_tag<native_vector_width_short, sycl::device> {
//   using return_type = std::uint32_t;
// };
// struct native_vector_width_int :
// detail::info_desc_tag<native_vector_width_int, sycl::device> {
//   using return_type = std::uint32_t;
// };
// struct native_vector_width_long :
// detail::info_desc_tag<native_vector_width_long, sycl::device> {
//   using return_type = std::uint32_t;
// };
// struct native_vector_width_float :
// detail::info_desc_tag<native_vector_width_float, sycl::device> {
//   using return_type = std::uint32_t;
// };
// struct native_vector_width_double :
// detail::info_desc_tag<native_vector_width_double, sycl::device> {
//   using return_type = std::uint32_t;
// };
// struct native_vector_width_half :
// detail::info_desc_tag<native_vector_width_half, sycl::device> {
//   using return_type = std::uint32_t;
// };
// struct max_clock_frequency : detail::info_desc_tag<max_clock_frequency,
// sycl::device> {
//   using return_type = std::uint32_t;
// };
// struct address_bits : detail::info_desc_tag<address_bits, sycl::device> {
//   using return_type = std::uint32_t;
// };
// struct max_mem_alloc_size : detail::info_desc_tag<max_mem_alloc_size,
// sycl::device> {
//   using return_type = std::uint64_t;
// };
// struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020, use
// device::has(aspect::image)") image_support :
// detail::info_desc_tag<image_support, sycl::device> {
//   using return_type = bool;
// };
// struct max_read_image_args : detail::info_desc_tag<max_read_image_args,
// sycl::device> {
//   using return_type = std::uint32_t;
// };
// struct max_write_image_args : detail::info_desc_tag<max_write_image_args,
// sycl::device> {
//   using return_type = std::uint32_t;
// };
// struct image2d_max_width : detail::info_desc_tag<image2d_max_width,
// sycl::device> {
//   using return_type = std::size_t;
// };
// struct image2d_max_height : detail::info_desc_tag<image2d_max_height,
// sycl::device> {
//   using return_type = std::size_t;
// };
// struct image3d_max_width : detail::info_desc_tag<image3d_max_width,
// sycl::device> {
//   using return_type = std::size_t;
// };
// struct image3d_max_height : detail::info_desc_tag<image3d_max_height,
// sycl::device> {
//   using return_type = std::size_t;
// };
// struct image3d_max_depth : detail::info_desc_tag<image3d_max_depth,
// sycl::device> {
//   using return_type = std::size_t;
// };
// struct image_max_buffer_size : detail::info_desc_tag<image_max_buffer_size,
// sycl::device> {
//   using return_type = std::size_t;
// };
// struct max_samplers : detail::info_desc_tag<max_samplers, sycl::device> {
//   using return_type = std::uint32_t;
// };
// struct max_parameter_size : detail::info_desc_tag<max_parameter_size,
// sycl::device> {
//   using return_type = std::size_t;
// };
// struct mem_base_addr_align : detail::info_desc_tag<mem_base_addr_align,
// sycl::device> {
//   using return_type = std::uint32_t;
// };
// struct half_fp_config : detail::info_desc_tag<half_fp_config, sycl::device> {
//   using return_type = std::vector<info::fp_config>;
// };
// struct single_fp_config : detail::info_desc_tag<single_fp_config,
// sycl::device> {
//   using return_type = std::vector<info::fp_config>;
// };
// struct double_fp_config : detail::info_desc_tag<double_fp_config,
// sycl::device> {
//   using return_type = std::vector<info::fp_config>;
// };
// struct global_mem_cache_type : detail::info_desc_tag<global_mem_cache_type,
// sycl::device> {
//   using return_type = info::global_mem_cache_type;
// };
// struct global_mem_cache_line_size :
// detail::info_desc_tag<global_mem_cache_line_size, sycl::device> {
//   using return_type = std::uint32_t;
// };
// struct global_mem_cache_size : detail::info_desc_tag<global_mem_cache_size,
// sycl::device> {
//   using return_type = std::uint64_t;
// };
// struct global_mem_size : detail::info_desc_tag<global_mem_size, sycl::device>
// {
//   using return_type = std::uint64_t;
// };
// struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020")
// max_constant_buffer_size : detail::info_desc_tag<max_constant_buffer_size,
// sycl::device> {
//   using return_type = std::uint64_t;
// };
// struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020") max_constant_args :
// detail::info_desc_tag<max_constant_args, sycl::device> {
//   using return_type = std::uint32_t;
// };
// struct local_mem_type : detail::info_desc_tag<local_mem_type, sycl::device> {
//   using return_type = info::local_mem_type;
// };
// struct local_mem_size : detail::info_desc_tag<local_mem_size, sycl::device> {
//   using return_type = std::uint64_t;
// };
// struct error_correction_support :
// detail::info_desc_tag<error_correction_support, sycl::device> {
//   using return_type = bool;
// };
// struct __SYCL2020_DEPRECATED("use device::has() with one of the aspect::usm_*
// aspects instead") host_unified_memory :
// detail::info_desc_tag<host_unified_memory, sycl::device> {
//   using return_type = bool;
// };
// struct atomic_memory_order_capabilities :
// detail::info_desc_tag<atomic_memory_order_capabilities, sycl::device> {
//   using return_type = std::vector<memory_order>;
// };
// struct atomic_fence_order_capabilities :
// detail::info_desc_tag<atomic_fence_order_capabilities, sycl::device> {
//   using return_type = std::vector<memory_order>;
// };
// struct atomic_memory_scope_capabilities :
// detail::info_desc_tag<atomic_memory_scope_capabilities, sycl::device> {
//   using return_type = std::vector<memory_scope>;
// };
// struct atomic_fence_scope_capabilities :
// detail::info_desc_tag<atomic_fence_scope_capabilities, sycl::device> {
//   using return_type = std::vector<memory_scope>;
// };
// struct profiling_timer_resolution :
// detail::info_desc_tag<profiling_timer_resolution, sycl::device> {
//   using return_type = std::size_t;
// };
// struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020, check the byte order
// of the host system instead, the host and the device are required to have the
// same byte order") is_endian_little : detail::info_desc_tag<is_endian_little,
// sycl::device> {
//   using return_type = bool;
// };
// struct is_available : detail::info_desc_tag<is_available, sycl::device> {
//   using return_type = bool;
// };
// struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020, use
// device::has(aspect::online_compiler) instead") is_compiler_available :
// detail::info_desc_tag<is_compiler_available, sycl::device> {
//   using return_type = bool;
// };
// struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020, use
// device::has(aspect::online_linker) instead") is_linker_available :
// detail::info_desc_tag<is_linker_available, sycl::device> {
//   using return_type = bool;
// };
// struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020")
// execution_capabilities : detail::info_desc_tag<execution_capabilities,
// sycl::device> {
//   using return_type = std::vector<info::execution_capability>;
// };
// struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020, use
// device::has(aspect::queue_profiling) instead") queue_profiling :
// detail::info_desc_tag<queue_profiling, sycl::device> {
//   using return_type = bool;
// };
// struct built_in_kernel_ids : detail::info_desc_tag<built_in_kernel_ids,
// sycl::device> {
//   using return_type = std::vector<kernel_id>;
// };
// struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020, use
// info::device::built_in_kernel_ids instead") built_in_kernels :
// detail::info_desc_tag<built_in_kernels, sycl::device> {
//   using return_type = std::vector<std::string>;
// };
// struct platform : detail::info_desc_tag<platform, sycl::device> {
//   using return_type = platform;
// };
struct name : detail::info_desc_tag<name, sycl::device> {
  using return_type = std::string;
};
struct vendor : detail::info_desc_tag<vendor, sycl::device> {
  using return_type = std::string;
};
struct driver_version : detail::info_desc_tag<driver_version, sycl::device> {
  using return_type = std::string;
};
// struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020") profile :
// detail::info_desc_tag<profile, sycl::device> {
//   using return_type = std::string;
// };
struct version : detail::info_desc_tag<version, sycl::device> {
  using return_type = std::string;
};
struct backend_version : detail::info_desc_tag<backend_version, sycl::device> {
  using return_type = std::string;
};
// struct aspects : detail::info_desc_tag<aspects, sycl::device> {
//   using return_type = std::vector<aspect>;
// };
// struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020, use
// info::device::aspects instead") extensions :
// detail::info_desc_tag<extensions, sycl::device> {
//   using return_type = std::vector<std::string>;
// };
// struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020") printf_buffer_size :
// detail::info_desc_tag<printf_buffer_size, sycl::device> {
//   using return_type = std::size_t;
// };
// struct __SYCL2020_DEPRECATED("deprecated in SYCL 2020")
// preferred_interop_user_sync :
// detail::info_desc_tag<preferred_interop_user_sync, sycl::device> {
//   using return_type = bool;
// };
// struct parent_device : detail::info_desc_tag<parent_device, sycl::device> {
//   using return_type = device;
// };
// struct partition_max_sub_devices :
// detail::info_desc_tag<partition_max_sub_devices, sycl::device> {
//   using return_type = std::uint32_t;
// };
// struct partition_properties : detail::info_desc_tag<partition_properties,
// sycl::device> {
//   using return_type = std::vector<info::partition_property>;
// };
// struct partition_affinity_domains :
// detail::info_desc_tag<partition_affinity_domains, sycl::device> {
//   using return_type = std::vector<info::partition_affinity_domain>;
// };
// struct partition_type_property :
// detail::info_desc_tag<partition_type_property, sycl::device> {
//   using return_type = info::partition_property;
// };
// struct partition_type_affinity_domain :
// detail::info_desc_tag<partition_type_affinity_domain, sycl::device> {
//   using return_type = info::partition_affinity_domain;
// };

} // namespace device
} // namespace info

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_INFO_DEVICE_HPP
