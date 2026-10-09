//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains helpers to cast a pointer between SYCL address spaces.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_DETAIL_ADDRESS_SPACE_CAST_HPP
#define _LIBSYCL___IMPL_DETAIL_ADDRESS_SPACE_CAST_HPP

#include <sycl/__impl/access_enums.hpp>
#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/decorated_type.hpp>

#include <type_traits>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

#ifdef __SYCL_DEVICE_ONLY__
// SPIR-V storage classes accepted by
// __builtin_spirv_generic_cast_to_ptr_explicit.
enum class SPIRVStorageClass : int {
  Workgroup = 4,      // local
  CrossWorkgroup = 5, // global
  Function = 7        // private
};
#endif

/// Casts Ptr to address space Space without a run-time check. The cast must be
/// known to be valid, i.e. either the source or the destination is the generic
/// address space, or the two are the same.
template <access::address_space Space, typename ElementType>
auto staticAddressCast(ElementType *Ptr) {
  using Pointee =
      std::remove_pointer_t<typename RemoveDecoration<ElementType *>::type>;
  using DstT = DecoratedType_t<Pointee, Space> *;

  static_assert(isAddressSpaceCastPossible(deduceAS<ElementType *>, Space),
                "the source and the destination address space of a static "
                "address space cast must be compatible");

  // A C-style cast is required: reinterpret_cast is not allowed to change the
  // address space of a pointer.
  return (DstT)Ptr;
}

/// Casts Ptr to address space Space, checking at run time that Ptr really
/// points into that address space. \return nullptr if it does not.
template <access::address_space Space, bool AllowUnsupported = false,
          typename ElementType>
auto dynamicAddressCast(ElementType *Ptr) {
  using Pointee =
      std::remove_pointer_t<typename RemoveDecoration<ElementType *>::type>;
  using DstT = DecoratedType_t<Pointee, Space> *;

  constexpr access::address_space SrcSpace = deduceAS<ElementType *>;
  constexpr auto Generic = access::address_space::generic_space;

  if constexpr (!isAddressSpaceCastPossible(SrcSpace, Space)) {
    // SYCL 2020 4.7.7.1: an implementation must return nullptr if the value of
    // the pointer is not compatible with the destination address space.
    return (DstT) nullptr;
  } else if constexpr (SrcSpace != Generic || Space == Generic) {
    // No run-time check is needed: the source already is in the destination
    // address space, or the destination is the generic address space.
    return staticAddressCast<Space>(Ptr);
  } else {
#ifdef __SYCL_DEVICE_ONLY__
    using RemoveCvT = std::remove_cv_t<Pointee>;
    auto *Unqualified =
        const_cast<RemoveCvT *>(reinterpret_cast<const RemoveCvT *>(Ptr));
    if constexpr (Space == access::address_space::global_space)
      return (DstT)__builtin_spirv_generic_cast_to_ptr_explicit(
          Unqualified, static_cast<int>(SPIRVStorageClass::CrossWorkgroup));
    else if constexpr (Space == access::address_space::local_space)
      return (DstT)__builtin_spirv_generic_cast_to_ptr_explicit(
          Unqualified, static_cast<int>(SPIRVStorageClass::Workgroup));
    else if constexpr (Space == access::address_space::private_space)
      return (DstT)__builtin_spirv_generic_cast_to_ptr_explicit(
          Unqualified, static_cast<int>(SPIRVStorageClass::Function));
    else {
      // TODO: the constant address space has no corresponding SPIR-V storage
      // class accepted by the builtin.
      static_assert(AllowUnsupported,
                    "a dynamic address space cast to this address space is "
                    "not supported yet");
      return staticAddressCast<Space>(Ptr);
    }
#else
    // There is a single address space on host, so the cast always succeeds.
    return staticAddressCast<Space>(Ptr);
#endif
  }
}

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_DETAIL_ADDRESS_SPACE_CAST_HPP
