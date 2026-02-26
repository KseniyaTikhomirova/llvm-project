//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/exception.hpp>

#include <detail/device_impl.hpp>
#include <detail/program_manager.hpp>

#include <cstring>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

// static inline bool isDeviceImageCompressed(const __sycl_tgt_device_image
// &Bin) {
//   return Bin.ImageFormat == LIBSYCL_DEVICE_BINARY_TYPE_COMPRESSED_NONE;
// }

ProgramWrapper::ProgramWrapper(ol_device_handle_t Device,
                               DeviceImageWrapper *DevImage) {
  assert(Device);
  assert(DevImage);

  ol_result_t Result =
      olCreateProgram(Device, DevImage->getRawData().ImageStart,
                      DevImage->getSize(), &MProgram);
  if (!isSuccess(Result))
    throw;
}

ol_symbol_handle_t ProgramManager::getOrCreateKernel(const char *KernelName,
                                                     DeviceImpl &Device) {
  auto KernelIDIt = MKernelNameToID.find(KernelName);
  if (KernelIDIt == MKernelNameToID.end())
    // throw?
    return nullptr;

  auto Kernel = getKernel(KernelIDIt->second, Device);
  if (Kernel)
    return Kernel;

  DeviceImageWrapper *DevImage =
      getDeviceImage(KernelName, KernelIDIt->second, Device);
  if (!DevImage)
    throw;

  ol_program_handle_t Program = getOrCreateProgram(Device, DevImage);
  assert(Program);
  Kernel = createKernel(Program, KernelIDIt->second, KernelName, Device);
  assert(Kernel);
  return Kernel;
}

ol_program_handle_t
ProgramManager::getOrCreateProgram(DeviceImpl &Device,
                                   DeviceImageWrapper *DevImage) {
  if (auto DevToProgramIt = MPrograms.find(DevImage);
      DevToProgramIt != MPrograms.end()) {
    auto ProgramIt = DevToProgramIt->second.find(Device.getHandle());
    if (ProgramIt != DevToProgramIt->second.end())
      return ProgramIt->second;
  }

  std::unique_ptr<ProgramWrapper> NewProgramWrapper(
      new ProgramWrapper(Device.getHandle(), DevImage));
  auto Program = NewProgramWrapper->getHandle();
  // lock
  {
    MPrograms[DevImage].insert(std::make_pair(Device.getHandle(), Program));
    MProgramWrappers.insert(std::make_pair(NewProgramWrapper->getHandle(),
                                           std::move(NewProgramWrapper)));
  }

  return Program;
}

ol_symbol_handle_t ProgramManager::createKernel(ol_program_handle_t Program,
                                                const kernel_id &KernelID,
                                                const char *KernelName,
                                                DeviceImpl &Device) {
  ol_symbol_handle_t Kernel{};

  ol_result_t Result =
      olGetSymbol(Program, KernelName, OL_SYMBOL_KIND_KERNEL, &Kernel);
  if (!isSuccess(Result))
    throw;
  // lock
  MKernels.insert(
      std::make_pair(KernelID, std::make_pair(Device.getHandle(), Kernel)));
  return Kernel;
}

ol_symbol_handle_t ProgramManager::getKernel(const kernel_id &KernelID,
                                             DeviceImpl &Device) {
  // lock
  auto Range = MKernels.equal_range(KernelID);
  for (auto Kernels = Range.first; Kernels != Range.second; ++Kernels) {
    auto &[KernelDevice, KernelSymbol] = Kernels->second;
    if (KernelDevice == Device.getHandle()) {
      assert(KernelSymbol && "Built kernel symbol can't be null");
      return KernelSymbol;
    }
  }
  return nullptr;
}

static inline bool
checkFatBinVersion(const __sycl_tgt_bin_desc *const FatbinDesc) {
  // may be we still need a copy if they change this
  return FatbinDesc->Version == LIBSYCL_SUPPORTED_OFFLOAD_BINARY_VERSION;
}

static inline bool
checkDeviceImageValidness(const __sycl_tgt_device_image &DeviceImage) {
  return (DeviceImage.Version == 3 /*why*?*/) &&
         (DeviceImage.OffloadKind == OFK_SYCL) &&
         (DeviceImage.ImageFormat == IMG_SPIRV /*what else*/);
}

void ProgramManager::addImages(__sycl_tgt_bin_desc *FatbinDesc) {
  assert(FatbinDesc && "Device images descriptor can't be nullptr");

  if (!checkFatBinVersion(FatbinDesc))
    throw;
  // todo add locks
  for (int I = 0; I < FatbinDesc->NumDeviceBinaries; I++) {
    const auto &RawDeviceImage = FatbinDesc->DeviceImages[I];
    if (!checkDeviceImageValidness(RawDeviceImage))
      throw;

    const EntryTy *EntriesB = RawDeviceImage.EntriesBegin;
    const EntryTy *EntriesE = RawDeviceImage.EntriesEnd;
    // Ignore "empty" device image
    if (EntriesB == EntriesE)
      continue;

    // if (isDeviceImageCompressed(RawDeviceImage))
    //   throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
    //                         "Recieved a compressed device image, but "
    //                         "SYCL RT doesn't support compressed format."
    //                         "Aborting. ");

    std::unique_ptr<DeviceImageWrapper> NewImageWrapper =
        std::make_unique<DeviceImageWrapper>(RawDeviceImage);

    for (auto EntriesIt = EntriesB; EntriesIt != EntriesE; EntriesIt++) {

      auto Name = EntriesIt->SymbolName;
      auto KernelIDIt = MKernelNameToID.find(Name);
      if (KernelIDIt == MKernelNameToID.end()) {
        sycl::kernel_id KernelID =
            detail::createSyclObjFromImpl<sycl::kernel_id>(
                std::make_shared<detail::KernelIdImpl>(Name));
        KernelIDIt = MKernelNameToID.insert(
            MKernelNameToID.end(),
            std::make_pair(std::string_view(Name), KernelID));
      }

      MKernelIDToDevImageJIT.insert(
          std::make_pair(KernelIDIt->second, NewImageWrapper.get()));
    }

    MDeviceImageWrappers.insert(
        std::make_pair(&RawDeviceImage, std::move(NewImageWrapper)));
  }
}

void ProgramManager::removeImages(__sycl_tgt_bin_desc *FatbinDesc) {
  assert(FatbinDesc && "Device images descriptor can't be nullptr");
  if (FatbinDesc->NumDeviceBinaries == 0)
    return;

  if (!checkFatBinVersion(FatbinDesc))
    throw;

  // todo add locks
  for (int I = 0; I < FatbinDesc->NumDeviceBinaries; I++) {
    const auto &RawDeviceImage = FatbinDesc->DeviceImages[I];
    if (!checkDeviceImageValidness(RawDeviceImage))
      throw;

    auto DevImageIt = MDeviceImageWrappers.find(&RawDeviceImage);
    if (DevImageIt == MDeviceImageWrappers.end())
      continue;

    const EntryTy *EntriesB = RawDeviceImage.EntriesBegin;
    const EntryTy *EntriesE = RawDeviceImage.EntriesEnd;
    // Ignore "empty" device image
    if (EntriesB == EntriesE)
      continue;

    if (auto ProgramIt = MPrograms.find(DevImageIt->second.get());
        ProgramIt != MPrograms.end()) {
      for (auto &[DeviceHandle, ProgramHandle] : ProgramIt->second) {
        MProgramWrappers.erase(ProgramHandle);
      }
      MPrograms.erase(ProgramIt);
    }

    for (auto EntriesIt = EntriesB; EntriesIt != EntriesE; EntriesIt++) {

      if (auto KernelIDIt = MKernelNameToID.find(EntriesIt->SymbolName);
          KernelIDIt != MKernelNameToID.end()) {
        MKernels.erase(KernelIDIt->second);
        MKernelIDToDevImageJIT.erase(KernelIDIt->second);
        MKernelNameToID.erase(KernelIDIt);
      }
    }

    MDeviceImageWrappers.erase(DevImageIt);
  }
}

static bool isImageTargetCompatible(const DeviceImageWrapper &Image,
                                    const DeviceImpl &Device) {
  sycl::backend BE = Device.getBackend();
  const char *Target = Image.getRawData().TripleString;

  return (strcmp(Target, _LIBSYCL_DEVICE_BINARY_TARGET_SPIRV64) == 0) &&
         (BE == sycl::backend::level_zero);
}

DeviceImageWrapper *ProgramManager::getDeviceImage(const char *KernelName,
                                                   kernel_id KernelID,
                                                   DeviceImpl &Device) {
  auto [Begin, End] = MKernelIDToDevImageJIT.equal_range(KernelID);
  if (Begin != End) {
    ol_result_t Result{};
    bool IsValid{};
    // With AOT (not implemented yet), we need to analize and check
    // olIsValidBinary for AOT binaries first.
    for (auto It = Begin; It != End; ++It) {
      if (isImageTargetCompatible(*It->second, Device)) {
        Result = olIsValidBinary(Device.getHandle(),
                                 It->second->getRawData().ImageStart,
                                 It->second->getSize(), &IsValid);
        if (!isSuccess(Result))
          throw; // add info
        if (IsValid)
          return It->second;
      }
    }
  }

  throw exception(make_error_code(errc::runtime),
                  "No kernel named " + std::string(KernelName) + " was found");
}

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL

extern "C" void __sycl_register_lib(__sycl_tgt_bin_desc *FatbinDesc) {
  sycl::detail::ProgramManager::getInstance().addImages(FatbinDesc);
}

extern "C" void __sycl_unregister_lib(__sycl_tgt_bin_desc *FatbinDesc) {
  sycl::detail::ProgramManager::getInstance().removeImages(FatbinDesc);
}
