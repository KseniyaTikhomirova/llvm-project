//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// to add
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_DETAIL_KERNEL_ARG_HELPERS
#define _LIBSYCL___IMPL_DETAIL_KERNEL_ARG_HELPERS

#include <sycl/__impl/index_space_classes.hpp>

#include <sycl/__impl/detail/config.hpp>

#ifdef __SYCL_DEVICE_ONLY__
#include <sycl/__spirv/spirv_vars.hpp>
#endif

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

    template <int N>
  static inline constexpr bool is_valid_dimensions = (N > 0) && (N < 4);

  class Builder {
public:
  Builder() = delete;
  
  #ifdef __SYCL_DEVICE_ONLY__
  template <int Dims> static const id<Dims> getElement(id<Dims> *) {
    static_assert(is_valid_dimensions<Dims>, "invalid dimensions");
    return __spirv::initBuiltInGlobalInvocationId<Dims, id<Dims>>();
  }

    template <int Dims, bool WithOffset>
  static std::enable_if_t<WithOffset, item<Dims, WithOffset>>
  createItem(const range<Dims> &Extent, const id<Dims> &Index,
             const id<Dims> &Offset) {
    return item<Dims, WithOffset>(Extent, Index, Offset);
  }

  template <int Dims, bool WithOffset>
  static std::enable_if_t<!WithOffset, item<Dims, WithOffset>>
  createItem(const range<Dims> &Extent, const id<Dims> &Index) {
    return item<Dims, WithOffset>(Extent, Index);
  }

  template <int Dims, bool WithOffset>
  static std::enable_if_t<WithOffset, const item<Dims, WithOffset>> getItem()
  {
    static_assert(is_valid_dimensions<Dims>, "invalid dimensions");
    id<Dims> GlobalId{__spirv::initBuiltInGlobalInvocationId<Dims,
    id<Dims>>()}; range<Dims> GlobalSize{__spirv::initBuiltInGlobalSize<Dims,
    range<Dims>>()}; id<Dims>
    GlobalOffset{__spirv::initBuiltInGlobalOffset<Dims, id<Dims>>()}; return
    createItem<Dims, true>(GlobalSize, GlobalId, GlobalOffset);
  }

  template <int Dims, bool WithOffset>
  static std::enable_if_t<!WithOffset, const item<Dims, WithOffset>>
  getItem() {
    static_assert(is_valid_dimensions<Dims>, "invalid dimensions");
    id<Dims> GlobalId{__spirv::initBuiltInGlobalInvocationId<Dims,
    id<Dims>>()}; range<Dims> GlobalSize{__spirv::initBuiltInGlobalSize<Dims,
    range<Dims>>()}; return createItem<Dims, false>(GlobalSize, GlobalId);
  }

  template <int Dims, bool WithOffset>
  static auto getElement(item<Dims, WithOffset> *)
      -> decltype(getItem<Dims, WithOffset>()) {
    return getItem<Dims, WithOffset>();
  }

    #endif // __SYCL_DEVICE_ONLY__
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_DETAIL_KERNEL_ARG_HELPERS
