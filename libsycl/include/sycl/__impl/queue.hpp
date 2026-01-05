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

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/default_async_handler.hpp>
#include <sycl/__impl/detail/obj_utils.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class async_handler;

namespace detail {
class QueueImpl;
} // namespace detail

// SYCL 2020 4.6.5. Queue class
class _LIBSYCL_EXPORT queue {
public:
  queue(const queue &rhs) = default;

  queue(queue &&rhs) = default;

  queue &operator=(const queue &rhs) = default;

  queue &operator=(queue &&rhs) = default;

  friend bool operator==(const queue& lhs, const queue& rhs) { return lhs.impl == rhs.impl; }

  friend bool operator!=(const queue& lhs, const queue& rhs) { return !(lhs == rhs); }

  explicit queue(const property_list &propList = {})
      : device(detail::SelectDevice(default_selector_v),
               detail::defaultAsyncHandler, PropList) {}

  explicit queue(const async_handler &asyncHandler,
                 const property_list &propList = {})
      : device(detail::SelectDevice(default_selector_v), asyncHandler,
               PropList) {}

  template <typename DeviceSelector,
            typename =
                detail::EnableIfSYCL2020DeviceSelectorInvocable<DeviceSelector>>
  explicit queue(const DeviceSelector &deviceSelector,
                 const property_list &propList = {})
      : device(detail::SelectDevice(deviceSelector),
               detail::defaultAsyncHandler, PropList) {}

  template <typename DeviceSelector,
            typename =
                detail::EnableIfSYCL2020DeviceSelectorInvocable<DeviceSelector>>
  explicit queue(const DeviceSelector &deviceSelector,
                 const async_handler &asyncHandler,
                 const property_list &propList = {})
      : device(detail::SelectDevice(deviceSelector), asyncHandler, PropList) {}

  explicit queue(const device &syclDevice, const property_list &propList = {})
      : device(syncDevice, detail::defaultAsyncHandler, propList) {}

  explicit queue(const device &syclDevice, const async_handler &asyncHandler,
                 const property_list &propList = {});

  backend get_backend() const noexcept;

  context get_context() const;

  device get_device() const;

  bool is_in_order() const;

  template <typename Param>
  typename Param::return_type get_info() const;

  template <typename Param>
  typename Param::return_type get_backend_info() const;

private:
  queue(const std::shared_ptr<QueueImpl>& Impl) : impl(Impl) {}
  std::shared_ptr<QueueImpl> impl;

  friend sycl::detail::ImplUtils;
}; // class queue

_LIBSYCL_END_NAMESPACE_SYCL

template <>
struct std::hash<sycl::queue> : public sycl::detail::HashBase<sycl::queue> {};

#endif // _LIBSYCL___IMPL_QUEUE_HPP
