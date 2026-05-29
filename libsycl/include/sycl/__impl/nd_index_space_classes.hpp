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

#include <cstddef>
#include <type_traits>
#include <variant>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

class Builder;

} // namespace detail

// SYCL2020 4.9.1.5. nd_item class.
// nd_item<int Dimensions> identifies an instance of the function object
// executing at each point in an nd_range<int Dimensions> passed to a
// parallel_for call.
template <int Dimensions = 1> class nd_item {
public:
  static constexpr int dimensions = Dimensions;

  nd_item() = delete;

  // ktikhomi to add
  /* -- common interface members -- */

  id<Dimensions> get_global_id() const noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv::initBuiltInGlobalInvocationId<Dimensions, id<Dimensions>>();
#else
    return {};
#endif
  }

  std::size_t get_global_id(int dimension) const noexcept {
    return get_global_id()[dimension];
  }

  std::size_t get_global_linear_id() const noexcept {
    size_t LinId = 0;
    id<Dimensions> Index = get_global_id();
    range<Dimensions> Extent = get_global_range();
    id<Dimensions> Offset = get_offset();
    if (1 == Dimensions) {
      LinId = Index[0] - Offset[0];
    } else if (2 == Dimensions) {
      LinId = (Index[0] - Offset[0]) * Extent[1] + Index[1] - Offset[1];
    } else {
      LinId = (Index[0] - Offset[0]) * Extent[1] * Extent[2] +
              (Index[1] - Offset[1]) * Extent[2] + Index[2] - Offset[2];
    }
    return LinId;
  }

  id<Dimensions> get_local_id() const noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv::initBuiltInLocalInvocationId<Dimensions, id<Dimensions>>();
#else
    return {};
#endif
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

  group<Dimensions> get_group() const noexcept {
    // ktikhomi to check and implement group class
    // TODO: ideally Group object should be stateless and have a contructor with
    // no arguments.
    return detail::Builder::createGroup(get_global_range(), get_local_range(),
                                        get_group_range(), get_group_id());
  }

  // ktikhomi to check
  // Out-of-class definition in sub_group.hpp
  sub_group get_sub_group() const noexcept;

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
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv::initBuiltInNumWorkgroups<Dimensions, range<Dimensions>>();
#else
    return {};
#endif
  }

  std::size_t get_group_range(int dimension) const noexcept {
    return get_group_range()[dimension];
  }

  range<Dimensions> get_global_range() const noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv::initBuiltInGlobalSize<Dimensions, range<Dimensions>>();
#else
    return {};
#endif
  }

  std::size_t get_global_range(int dimension) const noexcept {
    return get_global_range()[dimension];
  }

  range<Dimensions> get_local_range() const noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv::initBuiltInWorkgroupSize<Dimensions, range<Dimensions>>();
#else
    return {};
#endif
  }

  std::size_t get_local_range(int dimension) const noexcept {
    return get_local_range()[dimension];
  }

  // Deprecated in SYCL 2020.
  __SYCL2020_DEPRECATED("offsets are deprecated in SYCL 2020")
  id<Dimensions> get_offset() const noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv::initBuiltInGlobalOffset<Dimensions, id<Dimensions>>();
#else
    return {};
#endif
  }

  nd_range<Dimensions> get_nd_range() const noexcept {
    return nd_range<Dimensions>(get_global_range(), get_local_range(),
                                get_offset());
  }

  // TODO: add wait_for and async_work_group_copy once builtins are implemented.

protected:
  friend class detail::Builder;

  // nd_item(const item<Dimensions, true> &, const item<Dimensions, false> &,
  //         const group<Dimensions> &) {}

  id<Dimensions> get_group_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv::initBuiltInWorkgroupId<Dimensions, id<Dimensions>>();
#else
    return {};
#endif
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
    nd_range() = default;

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
             id<Dimensions> offset = id<Dimensions>()) noexcept
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
  };

  _LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_ND_INDEX_SPACE_CLASSES_HPP
