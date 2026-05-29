//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL 2020 nd range and index space
/// identifiers (4.9.1.).
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_ND_INDEX_SPACE_CLASSES_HPP
#define _LIBSYCL___IMPL_ND_INDEX_SPACE_CLASSES_HPP

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/index_space_classes.hpp>
#include <sycl/__impl/memory_enums.hpp>
#include <sycl/__spirv/spirv_vars.hpp>

#include <cstddef>
#include <type_traits>
#include <variant>
// #include <array> //std::size_t

_LIBSYCL_BEGIN_NAMESPACE_SYCL

template <int> class nd_item;
namespace detail {
class Builder;
class UnifiedRangeView;
} // namespace detail

// SYCL 2020 4.9.1.8. sub_group class.
// The sub_group class encapsulates all functionality required to represent a
// particular sub-group within a parallel execution.
class sub_group {
public:
  using id_type = id<1>;
  using range_type = sycl::range<1>;
  using linear_id_type = std::uint32_t;
  static constexpr int dimensions = 1;
  static constexpr memory_scope fence_scope = memory_scope::sub_group;

  sub_group(const sub_group &rhs) = default;
  sub_group(sub_group &&rhs) = default;
  sub_group &operator=(const sub_group &rhs) = default;
  sub_group &operator=(sub_group &&rhs) = default;

  friend bool operator==(const sub_group &lhs, const sub_group &rhs) {
    return lhs.get_group_id() == rhs.get_group_id();
  }

  friend bool operator!=(const sub_group &lhs, const sub_group &rhs) {
    return !(lhs == rhs);
  }

  id_type get_group_id() const noexcept { return __spirv_BuiltInSubgroupId(); }

  id_type get_local_id() const noexcept {
    return __spirv_BuiltInSubgroupLocalInvocationId();
  }

  range_type get_local_range() const noexcept {
    return __spirv_BuiltInSubgroupSize();
  }

  range_type get_group_range() const noexcept {
    return __spirv_BuiltInNumSubgroups();
  }

  range_type get_max_local_range() const noexcept {
    return __spirv_BuiltInSubgroupMaxSize();
  }

  linear_id_type get_group_linear_id() const noexcept {
    return static_cast<linear_id_type>(get_group_id()[0]);
  }

  linear_id_type get_local_linear_id() const noexcept {
    return static_cast<linear_id_type>(get_local_id()[0]);
  }

  linear_id_type get_group_linear_range() const noexcept {
    return static_cast<linear_id_type>(get_group_range()[0]);
  }

  linear_id_type get_local_linear_range() const noexcept {
    return static_cast<linear_id_type>(get_local_range()[0]);
  }

  bool leader() const noexcept { return get_local_linear_id() == 0; }

protected:
  sub_group() = default;

  template <int dimensions> friend class sycl::nd_item;
};

template <int Dimensions = 1> class group {
public:
  using id_type = id<Dimensions>;
  using range_type = range<Dimensions>;
  using linear_id_type = std::size_t;
  static constexpr int dimensions = Dimensions;
  static constexpr memory_scope fence_scope = memory_scope::work_group;

  group(const group &rhs) = default;
  group(group &&rhs) = default;
  group &operator=(const group &rhs) = default;
  group &operator=(group &&rhs) = default;

  id<Dimensions> get_group_id() const noexcept {
    return __spirv::initBuiltInWorkgroupId<Dimensions, id<Dimensions>>();
  }

  std::size_t get_group_id(int dimension) const noexcept {
    return get_group_id()[dimension];
  }

  id<Dimensions> get_local_id() const noexcept {
    return __spirv::initBuiltInLocalInvocationId<Dimensions, id<Dimensions>>();
  }

  std::size_t get_local_id(int dimension) const noexcept {
    return get_local_id()[dimension];
  }

  range<Dimensions> get_local_range() const noexcept {
    return __spirv::initBuiltInWorkgroupSize<Dimensions, range<Dimensions>>();
  }

  std::size_t get_local_range(int dimension) const noexcept {
    return get_local_range()[dimension];
  }

  range<Dimensions> get_group_range() const noexcept {
    return __spirv::initBuiltInNumWorkgroups<Dimensions, range<Dimensions>>();
  }

  std::size_t get_group_range(int dimension) const noexcept {
    return get_group_range()[dimension];
  }

  range<Dimensions> get_max_local_range() const noexcept {
    return get_local_range();
  }

  std::size_t operator[](int dimension) const noexcept {
    return get_group_id(dimension);
  }

  // Follows SYCl2020 3.11.1. Linearization.
  static std::size_t linearize_id(const id<Dimensions> &Id,
                                  const range<Dimensions> &Range) noexcept {
    if constexpr (Dimensions == 1) {
      return Id[0];
    } else if constexpr (Dimensions == 2) {
      return Id[0] * Range[1] + Id[1];
    } else {
      return Id[0] * Range[1] * Range[2] + Id[1] * Range[2] + Id[2];
    }
  }

  std::size_t get_group_linear_id() const noexcept {
    return linearize_id(get_group_id(), get_group_range());
  }

  std::size_t get_local_linear_id() const noexcept {
    return linearize_id(get_local_id(), get_local_range());
  }

  static std::size_t
  multiply_all_dims(const range<Dimensions> &Range) noexcept {
    if constexpr (Dimensions == 1) {
      return Range[0];
    } else if constexpr (Dimensions == 2) {
      return Range[0] * Range[1];
    } else {
      return Range[0] * Range[1] * Range[2];
    }
  }

  std::size_t get_group_linear_range() const noexcept {
    auto groupRange = get_group_range();
    return multiply_all_dims(groupRange);
  }

  std::size_t get_local_linear_range() const noexcept {
    auto localRange = get_local_range();
    return multiply_all_dims(localRange);
  }

  bool leader() const noexcept { return (get_local_linear_id() == 0); }

  // TODO: implement parallel_for_work_item, async_work_group_copy and wait_for.

protected:
  group() = default;

  template <int dimensions> friend class sycl::nd_item;
};

// SYCL 2020 4.9.1.2. nd_range class.
// nd_range<int Dimensions> defines the iteration domain of both the
// work-groups and the overall dispatch.
template <int Dimensions = 1> class nd_range {
  static_assert(Dimensions >= 1 && Dimensions <= 3,
                "nd_range can only be 1-, 2-, or 3-dimensional.");

public:
  static constexpr int dimensions = Dimensions;

  nd_range(const nd_range<Dimensions> &rhs) = default;
  nd_range(nd_range<Dimensions> &&rhs) = default;
  nd_range<Dimensions> &operator=(const nd_range<Dimensions> &rhs) = default;
  nd_range<Dimensions> &operator=(nd_range<Dimensions> &&rhs) = default;

  friend bool operator==(const nd_range<Dimensions> &lhs,
                         const nd_range<Dimensions> &rhs) {
    return (rhs.MGlobalSize == lhs.MGlobalSize) &&
           (rhs.MLocalSize == lhs.MLocalSize) && (rhs.MOffset == lhs.MOffset);
  }

  friend bool operator!=(const nd_range<Dimensions> &lhs,
                         const nd_range<Dimensions> &rhs) {
    return !(lhs == rhs);
  }

  __SYCL2020_DEPRECATED("offset is deprecated in SYCL2020")
  nd_range(range<Dimensions> globalSize, range<Dimensions> localSize,
           id<Dimensions> offset) noexcept
      : MGlobalSize(globalSize), MLocalSize(localSize), MOffset(offset) {}

  nd_range(range<Dimensions> globalSize, range<Dimensions> localSize)
      : MGlobalSize(globalSize), MLocalSize(localSize),
        MOffset(id<Dimensions>()) {}

  range<Dimensions> get_global_range() const noexcept { return MGlobalSize; }
  range<Dimensions> get_local_range() const noexcept { return MLocalSize; }
  range<Dimensions> get_group_range() const noexcept {
    return MGlobalSize / MLocalSize;
  }
  __SYCL2020_DEPRECATED("offset is deprecated in SYCL2020")
  id<Dimensions> get_offset() const noexcept { return MOffset; }

  // add deprecated macro and add it to the index_space_classes.hpp file
protected:
  range<Dimensions> MGlobalSize;
  range<Dimensions> MLocalSize;
  id<Dimensions> MOffset;

  friend class detail::UnifiedRangeView;
};

// SYCL2020 4.9.1.5. nd_item class.
// nd_item<int Dimensions> identifies an instance of the function object
// executing at each point in an nd_range<int Dimensions> passed to a
// parallel_for call.
template <int Dimensions = 1> class nd_item {
public:
  static constexpr int dimensions = Dimensions;

  nd_item(const nd_item &rhs) = default;
  nd_item(nd_item &&rhs) = default;
  nd_item &operator=(const nd_item &rhs) = default;
  nd_item &operator=(nd_item &&rhs) = default;

  friend bool operator==(const nd_item &lhs, const nd_item &rhs) {
    // https://github.com/KhronosGroup/SYCL-Docs/issues/532
    return true;
  }

  friend bool operator!=(const nd_item &lhs, const nd_item &rhs) {
    return !(lhs == rhs);
  }

  id<Dimensions> get_global_id() const noexcept {
    return __spirv::initBuiltInGlobalInvocationId<Dimensions, id<Dimensions>>();
  }

  std::size_t get_global_id(int dimension) const noexcept {
    return get_global_id()[dimension];
  }

  std::size_t get_global_linear_id() const noexcept {
    size_t LinId = 0;
    id<Dimensions> Index = get_global_id();
    range<Dimensions> Extent = get_global_range();
    id<Dimensions> Offset = get_offset();
    if constexpr (1 == Dimensions) {
      LinId = Index[0] - Offset[0];
    } else if constexpr (2 == Dimensions) {
      LinId = (Index[0] - Offset[0]) * Extent[1] + Index[1] - Offset[1];
    } else {
      LinId = (Index[0] - Offset[0]) * Extent[1] * Extent[2] +
              (Index[1] - Offset[1]) * Extent[2] + Index[2] - Offset[2];
    }
    return LinId;
  }

  id<Dimensions> get_local_id() const noexcept {
    return __spirv::initBuiltInLocalInvocationId<Dimensions, id<Dimensions>>();
  }

  std::size_t get_local_id(int dimension) const noexcept {
    return get_local_id()[dimension];
  }

  std::size_t get_local_linear_id() const noexcept {
    size_t LinId = 0;
    id<Dimensions> Index = get_local_id();
    range<Dimensions> Extent = get_local_range();
    if (1 == Dimensions) {
      LinId = Index[0];
    } else if (2 == Dimensions) {
      LinId = Index[0] * Extent[1] + Index[1];
    } else {
      LinId =
          Index[0] * Extent[1] * Extent[2] + Index[1] * Extent[2] + Index[2];
    }
    return LinId;
  }

  group<Dimensions> get_group() const noexcept { return group<Dimensions>(); }

  sub_group get_sub_group() const noexcept { return sub_group(); }

  std::size_t get_group(int dimension) const noexcept {
    return get_group_id()[dimension];
  }

  std::size_t get_group_linear_id() const noexcept {
    size_t LinId = 0;
    id<Dimensions> Index = get_group_id();
    range<Dimensions> Extent = get_group_range();
    if (1 == Dimensions) {
      LinId = Index[0];
    } else if (2 == Dimensions) {
      LinId = Index[0] * Extent[1] + Index[1];
    } else {
      LinId =
          Index[0] * Extent[1] * Extent[2] + Index[1] * Extent[2] + Index[2];
    }
    return LinId;
  }

  range<Dimensions> get_group_range() const noexcept {
    return __spirv::initBuiltInNumWorkgroups<Dimensions, range<Dimensions>>();
  }

  std::size_t get_group_range(int dimension) const noexcept {
    return get_group_range()[dimension];
  }

  range<Dimensions> get_global_range() const noexcept {
    return __spirv::initBuiltInGlobalSize<Dimensions, range<Dimensions>>();
  }

  std::size_t get_global_range(int dimension) const noexcept {
    return get_global_range()[dimension];
  }

  range<Dimensions> get_local_range() const noexcept {
    return __spirv::initBuiltInWorkgroupSize<Dimensions, range<Dimensions>>();
  }

  std::size_t get_local_range(int dimension) const noexcept {
    return get_local_range()[dimension];
  }

  // Deprecated in SYCL 2020.
  __SYCL2020_DEPRECATED("offsets are deprecated in SYCL 2020")
  id<Dimensions> get_offset() const noexcept {
    return __spirv::initBuiltInGlobalOffset<Dimensions, id<Dimensions>>();
  }

  nd_range<Dimensions> get_nd_range() const noexcept {
    return nd_range<Dimensions>(get_global_range(), get_local_range(),
                                get_offset());
  }

  // TODO: add wait_for and async_work_group_copy once builtins are implemented.

protected:
  friend class detail::Builder;

  nd_item() = default;

  id<Dimensions> get_group_id() const {
    return __spirv::initBuiltInWorkgroupId<Dimensions, id<Dimensions>>();
  }
};

  _LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_ND_INDEX_SPACE_CLASSES_HPP
