//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL device class, which
/// represents a single SYCL device on which kernels can be executed.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_DEVICE_HPP
#define _LIBSYCL___IMPL_DEVICE_HPP


_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {
class device_impl;
} // namespace detail

class platform;
enum class aspect;

// 4.6.4. Device class
class  _LIBSYCL_EXPORT device
    : public detail::ObjBase<detail::device_impl&, device> {
public:
  /// Constructs a SYCL device instance using the default device.
  device();

  /// Constructs a SYCL device instance using the device
  /// identified by the device selector provided.
  /// \param DeviceSelector is SYCL 2020 Device Selector, a simple callable that
  /// takes a device and returns an int
  template <typename DeviceSelector,
            typename =
                detail::EnableIfSYCL2020DeviceSelectorInvocable<DeviceSelector>>
  explicit device(const DeviceSelector &deviceSelector)
      : device(detail::select_device(deviceSelector)) {}

  /// Returns the backend associated with this device.
  ///
  /// \return the backend associated with this device.
  backend get_backend() const noexcept;

  /// Check if device is a CPU device
  ///
  /// \return true if SYCL device is a CPU device
  bool is_cpu() const;

  /// Check if device is a GPU device
  ///
  /// \return true if SYCL device is a GPU device
  bool is_gpu() const;

  /// Check if device is an accelerator device
  ///
  /// \return true if SYCL device is an accelerator device
  bool is_accelerator() const;

  /// Get associated SYCL platform
  ///
  /// \return The associated SYCL platform.
  platform get_platform() const;
  
  /// Queries this SYCL device for information requested by the template
  /// parameter param
  ///
  /// \return device info of type described in 4.6.4.4. Information descriptors.
  // template <typename Param>
  // typename detail::is_device_info_desc<Param>::return_type get_info() const {
  //   return detail::convert_from_abi_neutral(get_info_impl<Param>());
  // }

  /// Queries this SYCL device for SYCL backend-specific information.
  ///
  /// The return type depends on information being queried.
  // typename detail::is_backend_info_desc<Param>::return_type
  //     get_backend_info() const;

  /// Queries which optional features this device supports (if any).
  /// \return true if this device has the given aspect
  // bool has(aspect asp) const;

  /// Check SYCL extension support by device
  ///
  /// \param extension_name is a name of queried extension.
  /// \return true if SYCL device supports the extension.
  // __SYCL2020_DEPRECATED("use device::has() function with aspects APIs instead")
  // bool has_extension(const std::string& extension) const;

  /// Query available SYCL devices
  ///
  /// \param deviceType is one of the values described in A.3 of SYCL Spec
  /// \return a std::vector containing all SYCL devices available in the system
  /// of the device type specified
  static std::vector<device>
  get_devices(info::device_type deviceType = info::device_type::all);

private:
  device(const detail::device_impl& Impl) : ObjBase(Impl) {}
  friend detail::ObjBase<detail::device_impl&, device>;
};

_LIBSYCL_END_NAMESPACE_SYCL

template <>
struct std::hash<sycl::device>
    : public sycl::detail::HashBase<sycl::device> {};

#endif // _LIBSYCL___IMPL_DEVICE_HPP
