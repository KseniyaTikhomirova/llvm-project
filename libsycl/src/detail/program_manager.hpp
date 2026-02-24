//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_PROGRAM_MANAGER
#define _LIBSYCL_PROGRAM_MANAGER

#include <sycl/__impl/detail/config.hpp>

#include <detail/device_binary_structures.hpp>
#include <detail/device_image_wrapper.hpp>
#include <detail/kernel_id.hpp>

#include <OffloadAPI.h>

#include <unordered_map>
#include <vector>

// +++ Entry points referenced by the offload wrapper object {

/// Executed as a part of current module's (.exe, .dll) static initialization.
/// Registers device executable images with the runtime.
extern "C" _LIBSYCL_EXPORT void __sycl_register_lib(__sycl_tgt_bin_desc *desc);

/// Executed as a part of current module's (.exe, .dll) static
/// de-initialization.
/// Unregisters device executable images with the runtime.
extern "C" _LIBSYCL_EXPORT void
__sycl_unregister_lib(__sycl_tgt_bin_desc *desc);

// +++ }

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

class ProgramWrapper {
public:
  ProgramWrapper(ol_device_handle_t Device, DeviceImageWrapper *DevImage);

  ~ProgramWrapper() {
    assert(MProgram);
    ol_result_t Result = olDestroyProgram(MProgram);
    // if (!isSuccess(Result))
    //   throw;
  }

  ol_program_handle_t getHandle() { return MProgram; }

private:
  ol_program_handle_t MProgram{};
};

class ProgramManager {

public:
  static ProgramManager &getInstance() {
    static ProgramManager PM{};
    return PM;
  }

  ol_symbol_handle_t getOrCreateKernel(const char *KernelName,
                                       ol_device_handle_t Device) {
    auto KernelIDIt = MKernelNameToID.find(KernelName);
    if (KernelIDIt == MKernelNameToID.end())
      // throw?
      return nullptr;

    auto Kernel = getKernel(KernelIDIt->second, Device);
    if (Kernel)
      return Kernel;

    // DeviceImageWrapper *DevImage =
    //     getDeviceImage(KernelID, Device(Type));
    // if (!DevImage)
    //   throw;

    // ol_program_handle_t Program = getOrCreateProgram(DevImage, Device);
    // assert(Program);
    // Kernel = createKernel(Program, KernelName);
    assert(Kernel);
    return Kernel;
  }

  void addImages(__sycl_tgt_bin_desc *FatbinDesc);

private:
  ProgramManager() = default;
  ~ProgramManager() = default;
  ProgramManager(ProgramManager const &) = delete;
  ProgramManager &operator=(ProgramManager const &) = delete;

  // void addImages(__sycl_tgt_bin_desc *FatbinDesc) {}

  // void removeImages(__sycl_tgt_bin_desc *FatbinDesc) {}

  // DeviceImageWrapper *getDeviceImage(kernel_id KernelID, DeviceImpl &Device)
  // {
  //   // search
  //   return nullptr;
  // }

  // ol_program_handle_t getOrCreateProgram(ol_device_handle_t Device,
  //                                        DeviceImageWrapper *DevImage) {
  //   auto ProgramIt = MPrograms.find({DevImage, Device});
  //   if (ProgramIt != MPrograms.end())
  //     return ProgramIt->second;

  //   std::unique_ptr<ProgramWrapper> NewProgramWrapper(
  //       new ProgramWrapper(Device, DevImage));
  //   auto Program = NewProgramWrapper->getHandle();
  //   // lock
  //   {
  //     MPrograms.insert(std::make_pair(DevImage.get(), Device), Program);
  //     MProgramWrappers.push_back(std::move(NewProgramWrapper));
  //   }

  //   return Program;
  // }

  // ol_symbol_handle_t createKernel(ol_program_handle_t Program, const
  // kernel_id& KernelID,
  //                                 const char *KernelName) {
  //   ol_symbol_handle_t Kernel{};

  //   ol_result_t Result =
  //       olGetSymbol(Program, KernelName, OL_SYMBOL_KIND_KERNEL, &Kernel);
  //   if (!isSuccess(Result))
  //     throw;
  //   // lock
  //   MKernels.insert(std::make_pair(Program, KernelID), Kernel);
  //   return Kernel;
  // }

  ol_symbol_handle_t getKernel(const kernel_id &KernelID,
                               ol_device_handle_t Device) {
    // lock
    auto Range = MKernels.equal_range(KernelID);
    for (auto Kernels = Range.first; Kernels != Range.second; ++Kernels) {
      auto &[KernelDevice, KernelSymbol] = Kernels->second;
      if (KernelDevice == Device) {
        assert(KernelSymbol && "Built kernel symbol can't be null");
        return KernelSymbol;
      }
    }
    return nullptr;
  }

static inline bool isDeviceImageCompressed(const __sycl_tgt_device_image& Bin) {
  return Bin.ImageFormat == SYCL_DEVICE_BINARY_TYPE_COMPRESSED_NONE;
}

  // add Mutexes

  // filled by addImages(...);
  std::unordered_map<std::string_view, kernel_id> MKernelNameToID;
  // std::unordered_multimap<kernel_id, DeviceImageWrapper *>
  //     MKernelIDToDevImageJIT;

  // Filled by getOrCreateKernel and everything it calls inside.
  // std::map<std::pair<DeviceImageWrapper *, ol_device_handle_t>,
  //          ol_program_handle_t>
  //     MPrograms;
  std::unordered_multimap<kernel_id,
                          std::pair<ol_device_handle_t, ol_symbol_handle_t>>
      MKernels;

  // controls lifetime
  std::vector<std::unique_ptr<DeviceImageWrapper>> MDeviceImageWrappers;
  // std::vector<std::unique_ptr<ProgramWrapper>> MProgramWrappers;
};

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_PROGRAM_MANAGER
