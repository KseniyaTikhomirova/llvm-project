//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL queue class, which
/// schedules kernels on a device.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_QUEUE_HPP
#define _LIBSYCL___IMPL_QUEUE_HPP

#include <sycl/__impl/detail/arg_wrapper.hpp>
#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/default_async_handler.hpp>
#include <sycl/__impl/detail/obj_utils.hpp>
#include <sycl/__impl/detail/unified_range_view.hpp>

#include <sycl/__impl/async_handler.hpp>
#include <sycl/__impl/device.hpp>
#include <sycl/__impl/event.hpp>
#include <sycl/__impl/index_space_classes.hpp>
#include <sycl/__impl/property_list.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class context;

namespace detail {
class QueueImpl;
class UnifiedRangeView;


template <typename, typename T> struct checkFuncSignature {
  static_assert(false,
                "Second template parameter is required to be of function type");
};

template <typename F, typename RetT, typename... Args>
struct checkFuncSignature<F, RetT(Args...)> {
public:
  static constexpr bool value = std::is_invocable_r<void, F, Args...>::value;
};

} // namespace detail

// SYCL 2020 4.6.5. Queue class.
class _LIBSYCL_EXPORT queue {
public:
  queue(const queue &rhs) = default;

  queue(queue &&rhs) = default;

  queue &operator=(const queue &rhs) = default;

  queue &operator=(queue &&rhs) = default;

  friend bool operator==(const queue &lhs, const queue &rhs) {
    return lhs.impl == rhs.impl;
  }

  friend bool operator!=(const queue &lhs, const queue &rhs) {
    return !(lhs == rhs);
  }

  /// Constructs a SYCL queue instance using the device returned by an instance
  /// of default_selector.
  ///
  /// \param propList is a list of properties for queue construction.
  explicit queue(const property_list &propList = {})
      : queue(detail::SelectDevice(default_selector_v),
              detail::defaultAsyncHandler, propList) {}

  /// Constructs a SYCL queue instance with an async_handler using the device
  /// returned by an instance of default_selector.
  ///
  /// \param asyncHandler is a SYCL asynchronous exception handler.
  /// \param propList is a list of properties for queue construction.
  explicit queue(const async_handler &asyncHandler,
                 const property_list &propList = {})
      : queue(detail::SelectDevice(default_selector_v), asyncHandler,
              propList) {}

  /// Constructs a SYCL queue instance using the device identified by the
  /// device selector provided.
  /// \param deviceSelector is SYCL 2020 Device Selector, a simple callable that
  /// takes a device and returns an int
  /// \param propList is a list of properties for queue construction.
  template <
      typename DeviceSelector,
      typename = detail::EnableIfDeviceSelectorIsInvocable<DeviceSelector>>
  explicit queue(const DeviceSelector &deviceSelector,
                 const property_list &propList = {})
      : queue(detail::SelectDevice(deviceSelector), detail::defaultAsyncHandler,
              propList) {}

  /// Constructs a SYCL queue instance using the device identified by the
  /// device selector provided.
  /// \param deviceSelector is SYCL 2020 Device Selector, a simple callable that
  /// takes a device and returns an int
  /// \param asyncHandler is a SYCL asynchronous exception handler.
  /// \param propList is a list of properties for queue construction.
  template <
      typename DeviceSelector,
      typename = detail::EnableIfDeviceSelectorIsInvocable<DeviceSelector>>
  explicit queue(const DeviceSelector &deviceSelector,
                 const async_handler &asyncHandler,
                 const property_list &propList = {})
      : queue(detail::SelectDevice(deviceSelector), asyncHandler, propList) {}

  /// Constructs a SYCL queue instance using the device provided.
  ///
  /// \param syclDevice is an instance of SYCL device.
  /// \param propList is a list of properties for queue construction.
  explicit queue(const device &syclDevice, const property_list &propList = {})
      : queue(syclDevice, detail::defaultAsyncHandler, propList) {}

  /// Constructs a SYCL queue instance with an async_handler using the device
  /// provided.
  ///
  /// \param syclDevice is an instance of SYCL device.
  /// \param asyncHandler is a SYCL asynchronous exception handler.
  /// \param propList is a list of properties for queue construction.
  explicit queue(const device &syclDevice, const async_handler &asyncHandler,
                 const property_list &propList = {});

  /// Returns the SYCL backend that is associated with this queue.
  ///
  /// \return the backend associated with this queue.
  backend get_backend() const noexcept;

  /// Returns context that is associated with this queue.
  ///
  /// \return an associated SYCL context.
  context get_context() const;

  /// Returns device that is associated with this queue.
  ///
  /// \return SYCL device this queue was constructed with.
  device get_device() const;

  /// Returns whether the queue is in order or out of order.
  ///
  /// Equivalent to has_property<property::queue::in_order>().
  ///
  /// \return true if queue is in order.
  bool is_in_order() const;

  /// Queries SYCL queue for information.
  ///
  /// The return type depends on information being queried.
  template <typename Param> typename Param::return_type get_info() const;

  /// Queries SYCL queue for SYCL backend-specific information.
  ///
  /// The return type depends on information being queried.
  template <typename Param>
  typename Param::return_type get_backend_info() const;

  template <typename KernelName, typename KernelType>
  event single_task(const KernelType& kernelFunc)
  {
    return single_task<KernelName, KernelType>({}, kernelFunc);
  }

  template <typename KernelName, typename KernelType>
  event single_task(event depEvent, const KernelType &kernelFunc) {
    return single_task<KernelName, KernelType>({depEvent}, kernelFunc);
  }

  // The approach with void sycl_kernel_launch(pack of arguments) implies that we can use or copy arguments only during that call.
  // Since it ass only kernel arguments as parameters and returns void - we have to split setting of extra kernel data like event dependencies and range 
  // and getting result event from arguments handling and direct kernel submision if it is possible.
  // Key stages:
  // 1) passing to queue (or handler in future) dependency events and range (for parallel_for), saving them in queue (copy/move).
  // 2) wrapping kernel arguments into typeless wrappers (pointer based, initially no copy) and passing to the queue. Then depending on scenario (without host tasks and accessors 
  // we should be able to submit everything directly) collection of arguments is converted to preferred liboffload structure (no copy of objects, copy of pointers) and passed to liboffload or RT does deep copy of provided arguments (simple copy of pointer of USM and copy of value for other arguments) to keep them alive till kernel enqueue outside parent submit call.
  // 3) getting event associated with kernel enqueue.
  // Key notes:
  // 1) Having these 3 separated calls is not the best solution but the only one allowing to avoid copy for some scenarios (otherwise we have to do deep copy always and then do joined kernel submission outside sycl_kernel_launch scope). 
  // 2) submit must be thread-safe. Since we have 3 calls we need to keep kernel params and resulting event in a per queue + per thread/per kernel way. 
  // To achieve this without copy and joined kernel submission queue (in future - handler) stores thread_local data for kernel submission.
  // thread_local can't be used for non-static class members so they are static. Given: same queue can be used from different threads but thread can't use different queues at the same moment; that means that we actually need per thread storage and static thread_local KernelData should be able to perform as expected. 

  template <typename KernelName, typename KernelType>
  event single_task(const std::vector<event> &depEvents,
                    const KernelType &kernelFunc) {
    static_assert(
        (detail::checkFuncSignature<std::remove_reference_t<KernelType>,
                                    void()>::value),
        "sycl::queue::single_task() requires a kernel instead of command "
        "group. ");

    setKernelParameters(depEvents);
    kernel_single_task<KernelName, KernelType>(kernelFunc);
    return getLastEvent();
  }

  //   template <typename KernelName, int Dims, typename... Rest>
  //   event parallel_for(range<Dims> numWorkItems, Rest &&...rest) {
  //     return parallel_for(numWorkItems, {}, rest...);
  //   }

  //   template <typename KernelName, int Dims, typename... Rest>
  //   event parallel_for(range<Dims> numWorkItems, event depEvent, Rest
  //   &&...rest) {
  //     return parallel_for(numWorkItems, {depEvent}, rest...);
  //   }

  //   template <typename RetType, typename Func, typename Arg>
  //   static Arg member_ptr_helper(RetType (Func::*)(Arg) const);

  //   // Non-const version of the above template to match functors whose
  //   // 'operator()' is declared w/o the 'const' qualifier.
  //   template <typename RetType, typename Func, typename Arg>
  //   static Arg member_ptr_helper(RetType (Func::*)(Arg));

  //   template <typename F, typename SuggestedArgType>
  //   decltype(member_ptr_helper(&F::operator())) argument_helper(int);

  //   template <typename F, typename SuggestedArgType>
  //   SuggestedArgType argument_helper(...);

  //   template <typename F, typename SuggestedArgType>
  //   using lambda_arg_type = decltype(argument_helper<F,
  //   SuggestedArgType>(0));

  //   template <typename KernelName, int Dims, typename... Rest>
  //   event parallel_for(range<Dims> numWorkItems,
  //                      const std::vector<event> &depEvents, Rest &&...rest) {
  //     if constexpr (sizeof...(Rest) != 1)
  //       throw sycl::exception(unsupported, "Reductions are not supported.");

  // #ifndef __SYCL_DEVICE_ONLY__
  //     detail::checkValueRange<Dims>(Range);
  // #endif
  //     setKernelParameters(depEvents, numWorkItems);

  //     using LambdaArgType = sycl::detail::lambda_arg_type<Rest, item<Dims>>;
  //     // If 1D kernel argument is an integral type, convert it to
  //     sycl::item<1>
  //     // If user type is convertible from sycl::item/sycl::nd_item, use
  //     // sycl::item/sycl::nd_item to transport item information
  //     using TransformedArgType = std::conditional_t<
  //         std::is_integral<LambdaArgType>::value && Dims == 1, item<Dims>,
  //         typename detail::TransformUserItemType<Dims, LambdaArgType>::type>;

  //     kernel_parallel_for<KernelName, TransformedArgType, ... Rest>(rest...);
  //     return getLastEvent();
  //   }

  void wait();

private:
// #ifdef SYCL_LANGUAGE_VERSION
#define _LIBSYCL_ENTRY_POINT_ATTR__(KernelName)                                \
  [[clang::sycl_kernel_entry_point(KernelName)]]
  // #else
  // #define _LIBSYCL_ENTRY_POINT_ATTR__(KernelName)
  // #endif // SYCL_LANGUAGE_VERSION

  template <typename KernelName, typename KernelType>
  _LIBSYCL_ENTRY_POINT_ATTR__(KernelName)
  void kernel_single_task(const KernelType &KernelFunc) {
    KernelFunc();
  }

  // template <int N>
  // static inline constexpr bool is_valid_dimensions = (N > 0) && (N < 4);

  // template <typename T> T *declptr() { return static_cast<T *>(nullptr); }
  // template <int Dims> static const id<Dims> getElement(id<Dims> *) {
  //   static_assert(is_valid_dimensions<Dims>, "invalid dimensions");
  //   return __spirv::initBuiltInGlobalInvocationId<Dims, id<Dims>>();
  // }

  // template <int Dims, bool WithOffset>
  // static std::enable_if_t<WithOffset, const item<Dims, WithOffset>> getItem()
  // {
  //   static_assert(is_valid_dimensions<Dims>, "invalid dimensions");
  //   id<Dims> GlobalId{__spirv::initBuiltInGlobalInvocationId<Dims,
  //   id<Dims>>()}; range<Dims> GlobalSize{__spirv::initBuiltInGlobalSize<Dims,
  //   range<Dims>>()}; id<Dims>
  //   GlobalOffset{__spirv::initBuiltInGlobalOffset<Dims, id<Dims>>()}; return
  //   createItem<Dims, true>(GlobalSize, GlobalId, GlobalOffset);
  // }

  // template <int Dims, bool WithOffset>
  // static std::enable_if_t<!WithOffset, const item<Dims, WithOffset>>
  // getItem() {
  //   static_assert(is_valid_dimensions<Dims>, "invalid dimensions");
  //   id<Dims> GlobalId{__spirv::initBuiltInGlobalInvocationId<Dims,
  //   id<Dims>>()}; range<Dims> GlobalSize{__spirv::initBuiltInGlobalSize<Dims,
  //   range<Dims>>()}; return createItem<Dims, false>(GlobalSize, GlobalId);
  // }

  // template <int Dims, bool WithOffset>
  // static auto getElement(item<Dims, WithOffset> *)
  //     -> decltype(getItem<Dims, WithOffset>()) {
  //   return getItem<Dims, WithOffset>();
  // }

  // template <typename KernelName, typename ElementType, typename KernelType>
  // __SYCL_ENTRY_POINT_ATTR__ static void
  // kernel_parallel_for(const KernelType &KernelFunc) {
  //   KernelFunc(getElement(detail::declptr<ElementType>()));
  // }

  template <typename, typename... Args>
  void sycl_kernel_launch(const char *KernelName, Args... args) {

    detail::ArgCollection TypelessArgs;
    // check is device copyable
    (TypelessArgs.addArg(args), ...);

    submitKernelImpl(KernelName, TypelessArgs);
  }

  queue(const std::shared_ptr<detail::QueueImpl> &Impl) : impl(Impl) {}
  std::shared_ptr<detail::QueueImpl> impl;

  event getLastEvent();
  void submitKernelImpl(const char *KernelName,
                        detail::ArgCollection &TypelessArgs);
  void setKernelParameters(const std::vector<event> &Events,
                           const detail::UnifiedRangeView &Range = {});

  friend sycl::detail::ImplUtils;
}; // class queue

_LIBSYCL_END_NAMESPACE_SYCL

template <>
struct std::hash<sycl::queue> : public sycl::detail::HashBase<sycl::queue> {};

#endif // _LIBSYCL___IMPL_QUEUE_HPP
