//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/device_image_wrapper.hpp>

#include <detail/offload/offload_utils.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

ProgramWrapper::ProgramWrapper(ol_context_handle_t Context,
                               ol_device_handle_t Device,
                               const DeviceImageManager &DevImage) {
  assert(Context);
  assert(Device);

  llvm::StringRef Image = DevImage.getOffloadBinary().getImage();
  callAndThrow(olCreateProgram, Context, Device, Image.data(), Image.size(),
               &MProgram);
}

ProgramWrapper::~ProgramWrapper() {
  assert(MProgram);
  std::ignore = olDestroyProgram(MProgram);
  // TODO: define a way to report errors from dtors.
}

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
