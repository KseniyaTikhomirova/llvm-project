//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/exception.hpp>

#include <detail/program_manager.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

ProgramWrapper::ProgramWrapper(ol_device_handle_t Device,
                               DeviceImageWrapper *DevImage) {
  assert(Device);
  assert(DevImage);

  // ol_result_t Result = olCreateProgram(Device, DevImage->getProgData(),
  //                                      DevImage->getProgDataSize(),
  //                                      &MProgram);
  // if (!isSuccess(Result))
  //   throw;
}

// void registerAllExportedSymbols(DeviceImageWrapper *Image) {
//   for (const sycl_device_binary_property &ESProp : Img->getExportedSymbols())
//   {
//     m_ExportedSymbolImages.insert({ESProp->Name, Img.get()});
//   }
// }

// void registerKernelsForImage(DeviceImageWrapper *Image) {

//   std::shared_ptr<std::vector<kernel_id>> &KernelIDs =
//       m_BinImg2KernelIDs[Img.get()];
//   KernelIDs.reset(new std::vector<kernel_id>);

//   for (sycl_offload_entry EntriesIt = EntriesB; EntriesIt != EntriesE;
//        EntriesIt = EntriesIt->Increment()) {

//     auto name = EntriesIt->GetName();

//     // Skip creating device kernel information if it is an exported device
//     // function. Exported device functions appear in the offload entries
//     // among kernels, but are identifiable by being listed in properties.
//     if (m_ExportedSymbolImages.find(name) != m_ExportedSymbolImages.end())
//       continue;

//     auto It = m_DeviceKernelInfoMap.find(std::string_view(name));
//     if (It == m_DeviceKernelInfoMap.end()) {
//       sycl::kernel_id KernelID =
//       detail::createSyclObjFromImpl<sycl::kernel_id>(
//           std::make_shared<detail::kernel_id_impl>(name));
//       CompileTimeKernelInfoTy DefaultCompileTimeInfo{std::string_view(name)};
//       It = m_DeviceKernelInfoMap.emplace_hint(
//           It, std::piecewise_construct, std::forward_as_tuple(name),
//           std::forward_as_tuple(DefaultCompileTimeInfo, KernelID));
//     }
//     m_KernelIDs2BinImage.insert(
//         std::make_pair(It->second.getKernelID(), Img.get()));
//     KernelIDs->push_back(It->second.getKernelID());

//     // Keep track of image to kernel name reference count for cleanup.
//     m_KernelNameRefCount[name]++;
//   }

//   // Sort kernel ids for faster search
//   std::sort(KernelIDs->begin(), KernelIDs->end(), LessByHash<kernel_id>{});
// }

void ProgramManager::addImages(__sycl_tgt_bin_desc *FatbinDesc) {
  // todo add locks
  for (int I = 0; I < FatbinDesc->NumDeviceBinaries; I++) {
    const auto &RawDeviceImage = FatbinDesc->DeviceImages[I];

    const EntryTy *EntriesB = RawDeviceImage.EntriesBegin;
    const EntryTy *EntriesE = RawDeviceImage.EntriesEnd;
    // Ignore "empty" device image
    if (EntriesB == EntriesE)
      return;

    if (isDeviceImageCompressed(RawDeviceImage))
      throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                            "Recieved a compressed device image, but "
                            "SYCL RT doesn't support compressed format."
                            "Aborting. ");

    std::unique_ptr<DeviceImageWrapper> NewImageWrapper =
        std::make_unique<DeviceImageWrapper>(RawDeviceImage);

    // registerAllExportedSymbols(NewImageWrapper.get());
    // registerKernels(NewImageWrapper.get());

    MDeviceImageWrappers.push_back(std::move(NewImageWrapper));
  }
}

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL

extern "C" void __sycl_register_lib(__sycl_tgt_bin_desc *FatbinDesc) {
  sycl::detail::ProgramManager::getInstance().addImages(FatbinDesc);
}

extern "C" void __sycl_unregister_lib(__sycl_tgt_bin_desc *FatbinDesc) {
  // sycl::detail::ProgramManager::getInstance().removeImages(FatbinDesc);
}