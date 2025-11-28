//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the standard device selectors
/// (4.6.1.1. Device selector) included with all SYCL implementations.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_DEVICE_SELECTOR_HPP
#define _LIBSYCL___IMPL_DEVICE_SELECTOR_HPP

#include <sycl/__impl/detail/config.hpp>

#include <functional>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class device;
class context;
enum class aspect;

namespace detail {

// 4.6.1.1. Device selector:
// The interface for a device selector is any object that meets the C++ named
// requirement Callable, taking a parameter of type const device & and returning
// a value that is implicitly convertible to int.
using DeviceSelectorInvocableType = std::function<int(const sycl::device &)>;

template <typename DeviceSelector>
using EnableIfDeviceSelectorIsInvocable = std::enable_if_t<
    std::is_invocable_r_v<int, DeviceSelector &, const device &>>;

_LIBSYCL_EXPORT device
SelectDevice(const DeviceSelectorInvocableType &DeviceSelector);

_LIBSYCL_EXPORT device SelectDevice(
    const DeviceSelectorInvocableType &DeviceSelector, const context &Context);

} // namespace detail

_LIBSYCL_EXPORT int default_selector_v(const device &dev);
_LIBSYCL_EXPORT int gpu_selector_v(const device &dev);
_LIBSYCL_EXPORT int cpu_selector_v(const device &dev);
_LIBSYCL_EXPORT int accelerator_selector_v(const device &dev);
_LIBSYCL_EXPORT detail::DeviceSelectorInvocableType
aspect_selector(const std::vector<aspect> &RequireList,
                const std::vector<aspect> &DenyList = {});

template <typename... AspectListT>
detail::DeviceSelectorInvocableType aspect_selector(AspectListT... AspectList) {
  std::vector<aspect> RequireList;
  RequireList.reserve(sizeof...(AspectList));
  (RequireList.emplace_back(AspectList), ...);

  return aspect_selector(RequireList, {});
}

template <aspect... AspectList>
detail::DeviceSelectorInvocableType aspect_selector() {
  return aspect_selector({AspectList...}, {});
}

_LIBSYCL_END_NAMESPACE_SYCL

#endif //_LIBSYCL___IMPL_DEVICE_SELECTOR_HPP