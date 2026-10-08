//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL accessor class (4.7.6.).
/// Only the specialization for target::local, deprecated in SYCL 2020
/// in favour of sycl::local_accessor, is defined so far.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_ACCESSOR_HPP
#define _LIBSYCL___IMPL_ACCESSOR_HPP

#include <sycl/__impl/access_enums.hpp>
#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/local_accessor_base.hpp>
#include <sycl/__impl/index_space_classes.hpp>
#include <sycl/__impl/property_list.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class handler;

// access::placeholder and target::local are deprecated by SYCL 2020, but
// libsycl has to name them to declare the accessor class.
_LIBSYCL_SUPPRESS_DEPRECATED_PUSH

/// Provides access to a SYCL memory object from within a kernel or a host
/// task.
// TODO: only the target::local specialization below is defined so far, the
// primary template is incomplete.
template <typename DataT, int Dimensions = 1,
          access_mode AccessMode =
              (std::is_const_v<DataT> ? access_mode::read
                                      : access_mode::read_write),
          target AccessTarget = target::device,
          access::placeholder IsPlaceholder = access::placeholder::false_t>
class accessor;

/// Provides access to work-group local memory allocated for a kernel.
///
/// This is the SYCL 1.2.1 form of a local accessor, deprecated in SYCL 2020 in
/// favour of sycl::local_accessor. Naming target::local reports the
/// deprecation, so the specialization itself is not marked deprecated.
///
/// \sa detail::LocalAccessorBase for the shared implementation and for the
/// host and device representation of a local accessor.
template <typename DataT, int Dimensions, access_mode AccessMode,
          access::placeholder IsPlaceholder>
class _LIBSYCL_SPECIAL_CLASS
    accessor<DataT, Dimensions, AccessMode, target::local, IsPlaceholder>
    : public detail::LocalAccessorBase<DataT, Dimensions> {
  using Base = detail::LocalAccessorBase<DataT, Dimensions>;

  // SYCL 2020 4.7.6.9.4.7: this specialization is only available for
  // access_mode::read_write and access_mode::atomic.
  static_assert(AccessMode == access_mode::read_write ||
                    AccessMode == access_mode::atomic,
                "the accessor specialization with target::local is only "
                "available for access_mode::read_write and "
                "access_mode::atomic");

  static_assert(AccessMode != access_mode::atomic,
                "access_mode::atomic is not supported for a local accessor: "
                "sycl::atomic was removed in SYCL 2020, use sycl::atomic_ref "
                "instead");

  static_assert(!std::is_const_v<DataT>,
                "a const qualified DataT is not allowed for the accessor "
                "specialization with target::local, which is only available "
                "for access_mode::read_write");

public:
  using value_type = typename Base::value_type;
  using size_type = typename Base::size_type;

  accessor() = default;

  template <int Dims = Dimensions, std::enable_if_t<Dims == 0, bool> = true>
  accessor(handler &, const property_list & = {}) : Base(range<1>{1}) {}

  template <int Dims = Dimensions, std::enable_if_t<(Dims > 0), bool> = true>
  accessor(range<Dimensions> allocationSize, handler &,
           const property_list & = {})
      : Base(allocationSize) {}

  friend bool operator==(const accessor &Lhs, const accessor &Rhs) {
    return Lhs.isEqual(Rhs);
  }

  friend bool operator!=(const accessor &Lhs, const accessor &Rhs) {
    return !(Lhs == Rhs);
  }

  std::size_t get_size() const noexcept { return this->byte_size(); }

  std::size_t get_count() const noexcept { return this->size(); }

private:
  // Called by the device compiler at the beginning of the offload kernel entry
  // point. Must be defined here: an inherited __init is not accepted for a
  // class marked with _LIBSYCL_SPECIAL_CLASS.
  void __init(typename Base::LocalPtrT BlockBase,
              range<Base::AdjustedDims> Range, std::size_t Offset) {
    this->initialize(BlockBase, Range, Offset);
  }

  friend struct std::hash<accessor>;
};

_LIBSYCL_SUPPRESS_DEPRECATED_POP

_LIBSYCL_END_NAMESPACE_SYCL

_LIBSYCL_SUPPRESS_DEPRECATED_PUSH
template <typename DataT, int Dimensions, sycl::access_mode AccessMode,
          sycl::access::placeholder IsPlaceholder>
struct std::hash<sycl::accessor<DataT, Dimensions, AccessMode,
                                sycl::target::local, IsPlaceholder>> {
  std::size_t operator()(
      const sycl::accessor<DataT, Dimensions, AccessMode, sycl::target::local,
                           IsPlaceholder> &Acc) const {
    return std::hash<std::uint64_t>{}(Acc.getId());
  }
};
_LIBSYCL_SUPPRESS_DEPRECATED_POP

#endif // _LIBSYCL___IMPL_ACCESSOR_HPP
