//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// to add
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_DETAIL_UNIFIED_RANGE_VIEW_HPP
#define _LIBSYCL___IMPL_DETAIL_UNIFIED_RANGE_VIEW_HPP

#include <sycl/__impl/detail/arg_wrapper.hpp>
#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/default_async_handler.hpp>
#include <sycl/__impl/detail/obj_utils.hpp>

#include <sycl/__impl/async_handler.hpp>
#include <sycl/__impl/device.hpp>
#include <sycl/__impl/event.hpp>
#include <sycl/__impl/index_space_classes.hpp>
#include <sycl/__impl/property_list.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

class UnifiedRangeView {

public:
  UnifiedRangeView() = default;
  UnifiedRangeView(const UnifiedRangeView &Desc) = default;
  UnifiedRangeView(UnifiedRangeView &&Desc) = default;
  UnifiedRangeView &operator=(const UnifiedRangeView &Desc) = default;
  UnifiedRangeView &operator=(UnifiedRangeView &&Desc) = default;

  template <int Dims>
  UnifiedRangeView(sycl::range<Dims> &N)
      : MGlobalSize(&(N[0])), MDims(size_t(Dims)) {}

  // template <int Dims>
  // UnifiedRangeView(sycl::nd_range<Dims> &ExecutionRange)
  //     : MGlobalSize(&(ExecutionRange.globalSize[0])),
  //       MLocalSize(&(ExecutionRange.localSize[0])),
  //       MOffset(&(ExecutionRange.offset[0])), MDims{size_t(Dims)} {}

  UnifiedRangeView(const size_t *GlobalSize, const size_t *LocalSize,
                   const size_t *Offset, size_t Dims)
      : MGlobalSize(GlobalSize), MLocalSize(LocalSize), MOffset(Offset),
        MDims(Dims) {}

  const size_t *MGlobalSize = nullptr;
  const size_t *MLocalSize = nullptr;
  const size_t *MOffset = nullptr;
  size_t MDims = 1;
};
} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_DETAIL_UNIFIED_RANGE_VIEW_HPP
