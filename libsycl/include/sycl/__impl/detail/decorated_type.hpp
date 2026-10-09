//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the mapping between a SYCL address space and the
/// address space qualified type that represents it, and the traits that
/// recover an address space from a type and strip it again.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_DETAIL_DECORATED_TYPE_HPP
#define _LIBSYCL___IMPL_DETAIL_DECORATED_TYPE_HPP

#include <sycl/__impl/access_enums.hpp>
#include <sycl/__impl/detail/config.hpp>

#include <type_traits>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

/// Maps an address space to the address space qualified version of
/// ElementType. In a host compilation every specialization yields the
/// unqualified type.
template <typename ElementType, access::address_space Space>
struct DecoratedType;

template <typename ElementType>
struct DecoratedType<ElementType, access::address_space::global_space> {
  using type = ElementType _LIBSYCL_GLOBAL_AS;
};

template <typename ElementType>
struct DecoratedType<ElementType, access::address_space::local_space> {
  using type = ElementType _LIBSYCL_LOCAL_AS;
};

template <typename ElementType>
struct DecoratedType<ElementType, access::address_space::private_space> {
  using type = ElementType _LIBSYCL_PRIVATE_AS;
};

template <typename ElementType>
struct DecoratedType<ElementType, access::address_space::generic_space> {
  using type = ElementType _LIBSYCL_GENERIC_AS;
};

_LIBSYCL_SUPPRESS_DEPRECATED_PUSH
// The constant address space is read-only, so the decorated type is const
// qualified in a device compilation.
template <typename ElementType>
struct DecoratedType<ElementType, access::address_space::constant_space> {
#ifdef __SYCL_DEVICE_ONLY__
  using type = const ElementType _LIBSYCL_CONSTANT_AS;
#else
  using type = ElementType;
#endif
};
_LIBSYCL_SUPPRESS_DEPRECATED_POP

template <typename ElementType, access::address_space Space>
using DecoratedType_t = typename DecoratedType<ElementType, Space>::type;

/// \return the address space of a decorated type. An unqualified type is
/// reported as generic_space, which is what the default address space of
/// device code maps to.
template <typename T> struct DeduceAS {
  static constexpr access::address_space value =
      access::address_space::generic_space;
};

#ifdef __SYCL_DEVICE_ONLY__
template <typename T> struct DeduceAS<T _LIBSYCL_GLOBAL_AS> {
  static constexpr access::address_space value =
      access::address_space::global_space;
};

template <typename T> struct DeduceAS<T _LIBSYCL_LOCAL_AS> {
  static constexpr access::address_space value =
      access::address_space::local_space;
};

template <typename T> struct DeduceAS<T _LIBSYCL_PRIVATE_AS> {
  static constexpr access::address_space value =
      access::address_space::private_space;
};

template <typename T> struct DeduceAS<T _LIBSYCL_GENERIC_AS> {
  static constexpr access::address_space value =
      access::address_space::generic_space;
};

_LIBSYCL_SUPPRESS_DEPRECATED_PUSH
template <typename T> struct DeduceAS<T _LIBSYCL_CONSTANT_AS> {
  static constexpr access::address_space value =
      access::address_space::constant_space;
};
_LIBSYCL_SUPPRESS_DEPRECATED_POP
#endif // __SYCL_DEVICE_ONLY__

template <typename T> struct DeduceAS<T *> : DeduceAS<T> {};
template <typename T> struct DeduceAS<const T> : DeduceAS<T> {};

template <typename T>
inline constexpr access::address_space deduceAS = DeduceAS<T>::value;

/// Strips the address space from a type, see sycl::remove_decoration.
template <typename T> struct RemoveDecoration {
  using type = T;
};

#ifdef __SYCL_DEVICE_ONLY__
template <typename T> struct RemoveDecoration<T _LIBSYCL_GLOBAL_AS> {
  using type = T;
};

template <typename T> struct RemoveDecoration<T _LIBSYCL_LOCAL_AS> {
  using type = T;
};

template <typename T> struct RemoveDecoration<T _LIBSYCL_PRIVATE_AS> {
  using type = T;
};

template <typename T> struct RemoveDecoration<T _LIBSYCL_GENERIC_AS> {
  using type = T;
};

_LIBSYCL_SUPPRESS_DEPRECATED_PUSH
template <typename T> struct RemoveDecoration<T _LIBSYCL_CONSTANT_AS> {
  using type = T;
};
_LIBSYCL_SUPPRESS_DEPRECATED_POP
#endif // __SYCL_DEVICE_ONLY__

template <typename T> struct RemoveDecoration<T *> {
  using type = typename RemoveDecoration<T>::type *;
};

template <typename T> struct RemoveDecoration<T &> {
  using type = typename RemoveDecoration<T>::type &;
};

/// \return true if a pointer in address space Src can be cast to address
/// space Dst without a run-time check.
inline constexpr bool isAddressSpaceCastPossible(access::address_space Src,
                                                 access::address_space Dst) {
  // The constant address space is not interchangeable with any other one.
  if (Src == constantSpace || Dst == constantSpace)
    return Src == Dst;

  constexpr auto Generic = access::address_space::generic_space;
  return Src == Dst || Src == Generic || Dst == Generic;
}

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_DETAIL_DECORATED_TYPE_HPP
