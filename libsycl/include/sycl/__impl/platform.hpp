//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL platform class, which
/// encapsulates a single platform on which kernel functions may be executed.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_PLATFORM_HPP
#define _LIBSYCL___IMPL_PLATFORM_HPP

#include <sycl/__impl/backend.hpp>       // sycl::backend
#include <sycl/__impl/detail/config.hpp> // namespace macro
#include <sycl/__impl/detail/impl_utils.hpp>
#include <sycl/__impl/info/platform.hpp> // info::platform entries

#include <memory> // std::shared_ptr for impl
#include <vector> // required in get_platforms
// #include <string>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

// class device;
namespace detail {
class platform_impl;
} // namespace detail

// 4.6.2. Platform class
class _LIBSYCL_EXPORT platform {
public:
  /// Constructs a platform object that is a copy of the platform which contains
  /// the device returned by default_selector_v.
  platform();

  /// Constructs a platform object that is a copy of the platform which contains
  /// the device that is selected by selector.
  /// \param DeviceSelectorInstance is SYCL 2020 Device Selector, a simple
  /// callable taking a device reference and returning an integer rank.
  // template <typename DeviceSelector>
  // explicit platform(const DeviceSelector& DeviceSelectorInstance);

  /// Returns the backend associated with this platform.
  ///
  /// \return the backend associated with this platform
  backend get_backend() const noexcept;

  /// Returns all SYCL devices associated with this platform.
  ///
  /// If there are no devices that match given device
  /// type, resulting vector is empty.
  ///
  /// \param DeviceType is a SYCL device type.
  /// \return a vector of SYCL devices.
  // std::vector<device>
  //     get_devices(info::device_type DeviceType = info::device_type::all)
  //     const;

  /// Queries this SYCL platform for info.
  ///
  /// The return type depends on information being queried.
  // template <typename Param>
  // typename detail::is_platform_info_desc<Param>::return_type get_info()
  // const;

  template <typename Param>
  typename detail::is_backend_info_desc<Param>::return_type
  get_backend_info() const;

  /// Indicates if all of the SYCL devices on this platform have the
  /// given feature.
  ///
  /// \param Aspect is one of the values in Table 4.20 of the SYCL 2020
  /// Provisional Spec.
  ///
  /// \return true if all of the SYCL devices on this platform have the
  /// given feature.
  // bool has(aspect Aspect) const;

  /// Checks if platform supports specified extension.
  ///
  /// \param ExtensionName is a string containing extension name.
  /// \return true if specified extension is supported by this SYCL platform.
  // __SYCL2020_DEPRECATED(
  //     "use platform::has() function with aspects APIs instead")
  // bool has_extension(const std::string& ExtensionName) const; // Deprecated

  /// Returns all SYCL platforms from all backends that are available in the
  /// system.
  ///
  /// \return A std::vector containing all of the platforms from all backends
  /// that are available in the system.
  static std::vector<platform> get_platforms();

private:
  detail::platform_impl *impl = nullptr;

  platform(detail::platform_impl *Impl) : impl(Impl) {}

  // impl extraction utils:
  template <class T>
  friend T detail::createSyclObjFromImpl(
      std::add_rvalue_reference_t<decltype(T::impl)> ImplObj);
  template <class T>
  friend T detail::createSyclObjFromImpl(
      std::add_lvalue_reference_t<const decltype(T::impl)> ImplObj);
  template <class Obj>
  friend const decltype(Obj::impl) &
  detail::getSyclObjImpl(const Obj &SyclObject);
}; // class platform

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_PLATFORM_HPP