//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL 2020 local_accessor class
/// (4.7.6.11.), which allocates work-group local memory for a kernel.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_LOCAL_ACCESSOR_HPP
#define _LIBSYCL___IMPL_LOCAL_ACCESSOR_HPP

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/local_accessor_base.hpp>
#include <sycl/__impl/index_space_classes.hpp>
#include <sycl/__impl/property_list.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <utility>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class handler;

/// Provides access to work-group local memory allocated for a kernel.
///
/// \sa detail::LocalAccessorBase for the shared implementation and for the
/// host and device representation of a local accessor.
template <typename DataT, int Dimensions = 1>
class _LIBSYCL_SPECIAL_CLASS local_accessor
    : public detail::LocalAccessorBase<DataT, Dimensions> {
  using Base = detail::LocalAccessorBase<DataT, Dimensions>;

public:
  using value_type = typename Base::value_type;
  using size_type = typename Base::size_type;

  local_accessor() = default;

  template <int Dims = Dimensions, std::enable_if_t<Dims == 0, bool> = true>
  local_accessor(handler &, const property_list & = {}) : Base(range<1>{1}) {}

  template <int Dims = Dimensions, std::enable_if_t<(Dims > 0), bool> = true>
  local_accessor(range<Dimensions> allocationSize, handler &,
                 const property_list & = {})
      : Base(allocationSize) {}

  /// Implicit conversion from a read-write to a read-only local accessor.
  // template <typename OtherDataT,
  //           std::enable_if_t<
  //               std::is_const_v<DataT> &&
  //                   std::is_same_v<OtherDataT, std::remove_const_t<DataT>>,
  //               bool> = true>
  // local_accessor(const local_accessor<OtherDataT, Dimensions> &Other)
  // noexcept
  //     : Base(Other) {}

  friend bool operator==(const local_accessor &Lhs, const local_accessor &Rhs) {
    return Lhs.isEqual(Rhs);
  }

  friend bool operator!=(const local_accessor &Lhs, const local_accessor &Rhs) {
    return !(Lhs == Rhs);
  }

  void swap(local_accessor &Other) { this->swapBase(Other); }

  // template <int Dims = Dimensions,
  //           std::enable_if_t<Dims == 0 && !std::is_const_v<DataT>, bool> =
  //           true>
  // const local_accessor &operator=(const value_type &Other) const {
  //   *this->getPtr() = Other;
  //   return *this;
  // }

  // template <int Dims = Dimensions,
  //           std::enable_if_t<Dims == 0 && !std::is_const_v<DataT>, bool> =
  //           true>
  // const local_accessor &operator=(value_type &&Other) const {
  //   *this->getPtr() = std::move(Other);
  //   return *this;
  // }

private:
  // Called by the device compiler at the beginning of the offload kernel entry
  // // point. Must be defined here: an inherited __init is not accepted for a
  // // class marked with _LIBSYCL_SPECIAL_CLASS.
  // void __init(typename Base::LocalPtrT BlockBase,
  //             range<Base::AdjustedDims> Range, std::size_t Offset) {
  //   this->initialize(BlockBase, Range, Offset);
  // }

  friend struct std::hash<local_accessor>;
};

_LIBSYCL_END_NAMESPACE_SYCL

template <typename DataT, int Dimensions>
struct std::hash<sycl::local_accessor<DataT, Dimensions>> {
  std::size_t
  operator()(const sycl::local_accessor<DataT, Dimensions> &Acc) const {
    return std::hash<std::uint64_t>{}(Acc.getId());
  }
};

#endif // _LIBSYCL___IMPL_LOCAL_ACCESSOR_HPP
