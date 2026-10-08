//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains helpers for kernel submission entry points.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_DETAIL_KERNEL_SUBMISSION_HPP
#define _LIBSYCL___IMPL_DETAIL_KERNEL_SUBMISSION_HPP

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/get_device_kernel_info.hpp>
#include <sycl/__impl/detail/kernel_arg_helpers.hpp>
#include <sycl/__impl/detail/local_accessor_base.hpp>
#include <sycl/__impl/exception.hpp>
#include <sycl/__impl/index_space_classes.hpp>
#include <sycl/__impl/nd_item.hpp>
#include <sycl/__impl/nd_range.hpp>

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

template <int Dims>
void checkNDRangeAndThrow(const sycl::nd_range<Dims> &ExecutionRange) {
  if (ExecutionRange.get_global_range() != range<Dims>{} &&
      (ExecutionRange.get_local_range().size() == 0 ||
       ExecutionRange.get_global_range() % ExecutionRange.get_local_range() !=
           range<Dims>{}))
    throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                          "Invalid nd_range submission: global size must be "
                          "evenly divisible by local size.");
}

template <typename DerivedT> class KernelSubmissionBase {
protected:
  // TODO: take the kernel object by const reference in both entry points once
  // the compiler passes a mutable device kernel object to
  // sycl_reify_kernel_parameters for a const one.
  // Until then the by-value parameter keeps it mutable on device.
  template <typename KernelName, typename KernelType>
#ifdef SYCL_LANGUAGE_VERSION
  [[clang::sycl_kernel_entry_point(KernelName)]]
#endif
  void submitSingleTask(KernelType KernelFunc) {
    KernelFunc();
  }

  template <typename KernelName, typename ElementType, typename KernelType>
#ifdef SYCL_LANGUAGE_VERSION
  [[clang::sycl_kernel_entry_point(KernelName)]]
#endif
  void submitParallelFor(KernelType KernelFunc) {
    KernelFunc(detail::Builder::getElement(detail::declptr<ElementType>()));
  }

  // TODO: handle arrays of local accessors.
  template <typename KI, typename... Args>
  auto sycl_kernel_launch(Args &&...KernelArgs) {
    static_assert(sizeof...(KernelArgs) == 1,
                  "sycl_kernel_launch expects only 1 argument now: the "
                  "callable object passed to kernel invocation by the user.");

    return [this, &KernelArgs...](const auto &...LocalAccessors) {
      const std::size_t LocalMemSize =
          LocalAccessorReifier::getBlockSize(LocalAccessors...);

      auto &KernelObj = std::get<0>(std::tie(KernelArgs...));
      static_cast<DerivedT *>(this)->submitKernelImpl(
          detail::getDeviceKernelInfo<typename KI::kernel_name>(
              KI::kernel_entry_point_name),
          &KernelObj, sizeof(KernelObj), LocalMemSize);
    };
  }

  // Called at the start of the kernel. Must be static: 'this' isn't available
  // in the generated device entry point.
  template <typename KI, typename... Args>
  static auto sycl_reify_kernel_parameters(Args &...) {
    return [](auto &...LocalAccessors) {
      LocalAccessorReifier::bind(LocalAccessors...);
    };
  }

  template <typename KernelName, int Dims, template <int> class Range,
            typename... Rest>
  void parallelForImpl(const Range<Dims> & /*NumWorkItems*/,
                       Rest &&...RestArgs) {
    if constexpr (sizeof...(Rest) != 1)
      throw sycl::exception(sycl::make_error_code(errc::feature_not_supported),
                            "Reductions are not supported");

    using KernelType =
        std::decay_t<detail::nth_type_t<sizeof...(Rest) - 1, Rest...>>;
    constexpr bool IsNdRangeSubmission =
        std::is_same_v<Range<Dims>, nd_range<Dims>>;
    using SuggestedArgType =
        std::conditional_t<IsNdRangeSubmission, nd_item<Dims>, item<Dims>>;
    using LambdaArgType =
        sycl::detail::lambda_arg_type<KernelType, SuggestedArgType>;

    if constexpr (IsNdRangeSubmission) {
      static_assert(
          std::is_convertible_v<sycl::nd_item<Dims>, LambdaArgType>,
          "Kernel argument of a sycl::parallel_for with sycl::nd_range "
          "must be sycl::nd_item or be convertible from sycl::nd_item");
    } else {
      static_assert(
          std::is_convertible_v<sycl::item<Dims>, LambdaArgType> ||
              std::is_convertible_v<sycl::item<Dims, false>, LambdaArgType>,
          "Kernel argument of a sycl::parallel_for with sycl::range "
          "must be sycl::item or be convertible from sycl::item");
    }

    using TransformedLambdaArgType = std::conditional_t<
        IsNdRangeSubmission, nd_item<Dims>,
        std::conditional_t<
            std::is_convertible_v<sycl::item<Dims>, LambdaArgType>, item<Dims>,
            std::conditional_t<
                std::is_convertible_v<sycl::item<Dims, false>, LambdaArgType>,
                item<Dims, false>, LambdaArgType>>>;

    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    return submitParallelFor<NameT, TransformedLambdaArgType, KernelType>(
        std::forward<Rest>(RestArgs)...);
  }
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_DETAIL_KERNEL_SUBMISSION_HPP
