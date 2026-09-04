//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the helpers for device images and
/// programs.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_DEVICE_IMAGE_WRAPPER
#define _LIBSYCL_DEVICE_IMAGE_WRAPPER

#include <sycl/__impl/detail/config.hpp>

#include <llvm/Object/OffloadBinary.h>

#include <OffloadAPI.h>

#include <memory>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

class DeviceImageManager;

/// A wrapper of liboffload program handle to manage its lifetime.
class ProgramWrapper {
public:
  /// Constructs ProgramWrapper by creating a liboffload program with the
  /// provided arguments.
  ///
  /// \param Context is the context to use for program creation.
  /// \param Device is the device to use for program creation.
  /// \param DevImage is the device image to use for program creation.
  /// \throw sycl::exception with sycl::errc::runtime when failed to create the
  /// program.
  ProgramWrapper(ol_context_handle_t Context, ol_device_handle_t Device,
                 const DeviceImageManager &DevImage);

  /// Releases the corresponding liboffload program handle by calling
  /// olDestroyProgram.
  ~ProgramWrapper();

  ProgramWrapper(const ProgramWrapper &) = delete;
  ProgramWrapper &operator=(const ProgramWrapper &) = delete;
  ProgramWrapper(ProgramWrapper &&) = delete;
  ProgramWrapper &operator=(ProgramWrapper &&) = delete;

  /// \return the corresponding liboffload program handle.
  ol_program_handle_t getOLHandle() { return MProgram; }

private:
  ol_program_handle_t MProgram{};
};

/// This class manages data parsing of device images.
class DeviceImageManager {
public:
  DeviceImageManager(std::unique_ptr<llvm::object::OffloadBinary> Bin)
      : MBin(std::move(Bin)) {}
  // Explicitly delete copy constructor/operator= to avoid unintentional copies.
  DeviceImageManager(const DeviceImageManager &) = delete;
  DeviceImageManager &operator=(const DeviceImageManager &) = delete;

  DeviceImageManager(DeviceImageManager &&) = default;
  DeviceImageManager &operator=(DeviceImageManager &&) = default;

  ~DeviceImageManager() = default;

  /// \return a reference to the corresponding parsed OffloadBinary object.
  const llvm::object::OffloadBinary &getOffloadBinary() const { return *MBin; }

protected:
  std::unique_ptr<llvm::object::OffloadBinary> MBin;
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_DEVICE_IMAGE_WRAPPER
