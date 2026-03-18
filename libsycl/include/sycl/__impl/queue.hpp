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

#include <sycl/__impl/async_handler.hpp>
#include <sycl/__impl/device.hpp>
#include <sycl/__impl/event.hpp>
#include <sycl/__impl/index_space_classes.hpp>
#include <sycl/__impl/property_list.hpp>

#include <sycl/__impl/detail/arg_wrapper.hpp>
#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/default_async_handler.hpp>
#include <sycl/__impl/detail/obj_utils.hpp>
#include <sycl/__impl/detail/unified_range_view.hpp>
#include <sycl/__impl/detail/kernel_arg_helpers.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class context;

namespace detail {
class kernel_impl;

/// This class is the default KernelName template parameter type for kernel
/// invocation APIs such as single_task.
class auto_name {};

/// Helper struct to get a kernel name type based on given \c Name and \c Type
/// types: if \c Name is undefined (is a \c auto_name) then \c Type becomes
/// the \c Name.
template <typename Name, typename Type> struct get_kernel_name_t {
  using name = Name;
};

/// Specialization for the case when \c Name is undefined.
/// This is only legal with our compiler with the unnamed lambda extension or if
/// the kernel is a functor object. For the case where \c Type is a lambda
/// function and unnamed lambdas are disabled, the compiler will issue a
/// diagnostic.
template <typename Type> struct get_kernel_name_t<detail::auto_name, Type> {
  using name = Type;
};

} // namespace detail

namespace detail {
class QueueImpl;
class UnifiedRangeView;

// template <typename, typename T> struct checkFuncSignature {
//   static_assert(false,
//                 "Second template parameter is required to be of function type");
// };

// template <typename F, typename RetT, typename... Args>
// struct checkFuncSignature<F, RetT(Args...)> {
// public:
//   static constexpr bool value = std::is_invocable_r<void, F, Args...>::value;
// };

  template <typename RetType, typename Func, typename Arg>
    static Arg member_ptr_helper(RetType (Func::*)(Arg) const);

    // Non-const version of the above template to match functors whose
    // 'operator()' is declared w/o the 'const' qualifier.
    template <typename RetType, typename Func, typename Arg>
    static Arg member_ptr_helper(RetType (Func::*)(Arg));

    template <typename F, typename SuggestedArgType>
    decltype(member_ptr_helper(&F::operator())) argument_helper(int);

    template <typename F, typename SuggestedArgType>
    SuggestedArgType argument_helper(...);

    template <typename F, typename SuggestedArgType>
    using lambda_arg_type = decltype(argument_helper<F, SuggestedArgType>(0));

    template <typename T> T *declptr() { return static_cast<T *>(nullptr); }

  #if __has_builtin(__type_pack_element)
template <int N, typename... Ts>
using nth_type_t = __type_pack_element<N, Ts...>;
#else
template <int N, typename T, typename... Ts> struct nth_type {
  using type = typename nth_type<N - 1, Ts...>::type;
};

template <typename T, typename... Ts> struct nth_type<0, T, Ts...> {
  using type = T;
};

template <int N, typename... Ts>
using nth_type_t = typename nth_type<N, Ts...>::type;
#endif

template <int Dims, typename LambdaArgType> struct TransformUserItemType {
  using type = std::conditional_t<std::is_convertible_v<item<Dims>, LambdaArgType>,  item<Dims>, LambdaArgType>;
};

template <typename, typename T> struct check_fn_signature {
  static_assert(std::integral_constant<T, false>::value,
                "Second template parameter is required to be of function type");
};

template <typename F, typename RetT, typename... Args>
struct check_fn_signature<F, RetT(Args...)> {
private:
  template <typename T>
  static constexpr auto check(T *) -> typename std::is_same<
      decltype(std::declval<T>().operator()(std::declval<Args>()...)),
      RetT>::type;

  template <typename> static constexpr std::false_type check(...);

  using type = decltype(check<F>(0));

public:
  static constexpr bool value = type::value;
};

template <typename F, typename... Args>
static constexpr bool check_kernel_lambda_takes_args() {
  return check_fn_signature<std::remove_reference_t<F>, void(Args...)>::value;
}


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
  /// \param deviceSelector is a SYCL 2020 Device Selector, a simple callable
  /// that takes a device and returns an int
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
  /// \param deviceSelector is a SYCL 2020 Device Selector, a simple callable
  /// that takes a device and returns an int
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

  /// \return the SYCL backend associated with this queue.
  backend get_backend() const noexcept;

  /// \return the associated SYCL context.
  context get_context() const;

  /// \return the SYCL device this queue was constructed with.
  device get_device() const;

  /// Equivalent to has_property<property::queue::in_order>().
  ///
  /// \return true if and only if the queue is in order.
  bool is_in_order() const;

  /// Queries the queue for information.
  ///
  /// The return type depends on information being queried.
  template <typename Param> typename Param::return_type get_info() const;

  /// Queries the queue for SYCL backend-specific information.
  ///
  /// The return type depends on the information being queried.
  template <typename Param>
  typename Param::return_type get_backend_info() const;

  template <typename KernelName = detail::auto_name, typename KernelType>
  event single_task(const KernelType &kernelFunc) {
    return single_task<KernelName, KernelType>({}, kernelFunc);
  }

  template <typename KernelName = detail::auto_name, typename KernelType>
  event single_task(event depEvent, const KernelType &kernelFunc) {
    return single_task<KernelName, KernelType>({depEvent}, kernelFunc);
  }

  // The approach with void sycl_kernel_launch(pack of arguments) implies that
  // we can use or copy arguments only during that call. Since it ass only
  // kernel arguments as parameters and returns void - we have to split setting
  // of extra kernel data like event dependencies and range and getting result
  // event from arguments handling and direct kernel submision if it is
  // possible. Key stages: 1) passing to queue (or handler in future) dependency
  // events and range (for parallel_for), saving them in queue (copy/move). 2)
  // wrapping kernel arguments into typeless wrappers (pointer based, initially
  // no copy) and passing to the queue. Then depending on scenario (without host
  // tasks and accessors we should be able to submit everything directly)
  // collection of arguments is converted to preferred liboffload structure (no
  // copy of objects, copy of pointers) and passed to liboffload or RT does deep
  // copy of provided arguments (simple copy of pointer of USM and copy of value
  // for other arguments) to keep them alive till kernel enqueue outside parent
  // submit call. 3) getting event associated with kernel enqueue. Key notes: 1)
  // Having these 3 separated calls is not the best solution but the only one
  // allowing to avoid copy for some scenarios (otherwise we have to do deep
  // copy always and then do joined kernel submission outside sycl_kernel_launch
  // scope). 2) submit must be thread-safe. Since we have 3 calls we need to
  // keep kernel params and resulting event in a per queue + per thread/per
  // kernel way. To achieve this without copy and joined kernel submission queue
  // (in future - handler) stores thread_local data for kernel submission.
  // thread_local can't be used for non-static class members so they are static.
  // Given: same queue can be used from different threads but thread can't use
  // different queues at the same moment; that means that we actually need per
  // thread storage and static thread_local KernelData should be able to perform
  // as expected.

  template <typename KernelName = detail::auto_name, typename KernelType>
  event single_task(const std::vector<event> &depEvents,
                    const KernelType &kernelFunc) {
    static_assert(
        (detail::check_fn_signature<std::remove_reference_t<KernelType>,
                                    void()>::value),
        "sycl::queue::single_task() requires a kernel instead of command "
        "group. ");

    setKernelParameters(depEvents);
            using NameT =
      typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    kernel_single_task<NameT, KernelType>(kernelFunc);
    return getLastEvent();
  }

    template <typename KernelName = detail::auto_name, int Dims, typename... Rest>
    event parallel_for(range<Dims> numWorkItems, Rest &&...rest) {
      return parallel_for<KernelName>(numWorkItems, {}, rest...);
    }

    template <typename KernelName = detail::auto_name, int Dims, typename... Rest>
    event parallel_for(range<Dims> numWorkItems, event depEvent, Rest
    &&...rest) {
      return parallel_for<KernelName>(numWorkItems, {depEvent}, rest...);
    }

    template <typename KernelName = detail::auto_name, int Dims, typename... Rest>
    event parallel_for(range<Dims> numWorkItems,
                       const std::vector<event> &depEvents, Rest &&...rest) {
      if constexpr (sizeof...(Rest) != 1)
        throw sycl::exception(errc::feature_not_supported, "Reductions are not supported.");

  #ifndef __SYCL_DEVICE_ONLY__
      //detail::checkValueRange<Dims>(numWorkItems);
  #endif
      setKernelParameters(depEvents, numWorkItems);

      using KernelType =
        std::decay_t<detail::nth_type_t<sizeof...(Rest) - 1, Rest...>>;
    using LambdaArgType = sycl::detail::lambda_arg_type<KernelType, item<Dims>>;
    using TransformedArgType = std::conditional_t<
        std::is_integral<LambdaArgType>::value && Dims == 1, item<Dims>,
        typename detail::TransformUserItemType<Dims, LambdaArgType>::type>;
              using NameT =
      typename detail::get_kernel_name_t<KernelName, KernelType>::name;
      kernel_parallel_for<NameT, TransformedArgType, KernelType>(rest...);
      return getLastEvent();
    }

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
  void kernel_single_task(const KernelType KernelFunc) {
    KernelFunc();
  }

  template <typename KernelName, typename ElementType, typename KernelType>
  _LIBSYCL_ENTRY_POINT_ATTR__(KernelName) void
  kernel_parallel_for(const KernelType KernelFunc) {
    #ifdef __SYCL_DEVICE_ONLY__
    KernelFunc(detail::Builder::getElement(detail::declptr<ElementType>()));
    #endif
    (void)KernelFunc;
  }

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
