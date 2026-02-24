//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/device_image_wrapper.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

// void RTDeviceBinaryImage::PropertyRange::init(sycl_device_binary Bin,
//                                               const char *PropSetName) {
//   assert(!this->Begin && !this->End && "already initialized");
//   sycl_device_binary_property_set PS = nullptr;

//   for (PS = Bin->PropertySetsBegin; PS != Bin->PropertySetsEnd; ++PS) {
//     assert(PS->Name && "nameless property set - bug in the offload
//     wrapper?"); if (!strcmp(PropSetName, PS->Name))
//       break;
//   }
//   if (PS == Bin->PropertySetsEnd) {
//     Begin = End = nullptr;
//     return;
//   }
//   Begin = PS->PropertiesBegin;
//   End = Begin ? PS->PropertiesEnd : nullptr;
// }

// void RTDeviceBinaryImage::print() const {
//   std::cerr << "  --- Image " << Bin << "\n";
//   if (!Bin)
//     return;
//   std::cerr << "    Version  : " << (int)Bin->Version << "\n";
//   std::cerr << "    Kind     : " << (int)Bin->Kind << "\n";
//   std::cerr << "    Format   : " << (int)Bin->Format << "\n";
//   std::cerr << "    Target   : " << Bin->DeviceTargetSpec << "\n";
//   std::cerr << "    Bin size : "
//             << ((intptr_t)Bin->BinaryEnd - (intptr_t)Bin->BinaryStart) <<
//             "\n";
//   std::cerr << "    Compile options : "
//             << (Bin->CompileOptions ? Bin->CompileOptions : "NULL") << "\n";
//   std::cerr << "    Link options    : "
//             << (Bin->LinkOptions ? Bin->LinkOptions : "NULL") << "\n";
//   std::cerr << "    Entries  : ";

//   for (sycl_offload_entry EntriesIt = Bin->EntriesBegin;
//        EntriesIt != Bin->EntriesEnd; EntriesIt = EntriesIt->Increment())
//     std::cerr << EntriesIt->GetName() << " ";
//   std::cerr << "\n";
//   std::cerr << "    Properties [" << Bin->PropertySetsBegin << "-"
//             << Bin->PropertySetsEnd << "]:\n";

//   for (sycl_device_binary_property_set PS = Bin->PropertySetsBegin;
//        PS != Bin->PropertySetsEnd; ++PS) {
//     std::cerr << "      Category " << PS->Name << " [" << PS->PropertiesBegin
//               << "-" << PS->PropertiesEnd << "]:\n";

//     for (sycl_device_binary_property P = PS->PropertiesBegin;
//          P != PS->PropertiesEnd; ++P) {
//       std::cerr << "        " << DeviceBinaryProperty(P) << "\n";
//     }
//   }
// }

// void RTDeviceBinaryImage::dump(std::ostream &Out) const {
//   size_t ImgSize = getSize();
//   Out.write(reinterpret_cast<const char *>(Bin->BinaryStart), ImgSize);
// }

// sycl_device_binary_property
// RTDeviceBinaryImage::getProperty(const char *PropName) const {
//   if (!Misc.isAvailable())
//     return nullptr;
//   auto It = std::find_if(Misc.begin(), Misc.end(),
//                          [=](sycl_device_binary_property Prop) {
//                            return !strcmp(PropName, Prop->Name);
//                          });
//   if (It == Misc.end())
//     return nullptr;

//   return *It;
// }

DeviceImageWrapper::DeviceImageWrapper(const __sycl_tgt_device_image &Bin)
    : MBin(Bin) {
  // ImageId = ImageCounter++;

  // If there was no binary, we let the owner handle initialization as they see
  // fit. This is used when merging binaries, e.g. during linking.
  // if (!Bin)
  //   return;

  // // Bin != nullptr is guaranteed here.
  // this->Bin = Bin;
  // // If device binary image format wasn't set by its producer, then can't
  // change
  // // now, because 'Bin' data is part of the executable image loaded into
  // memory
  // // which can't be modified (easily).
  // // TODO clang driver + ClangOffloadWrapper can figure out the format and
  // set
  // // it when invoking the offload wrapper job
  // Format = static_cast<ur::DeviceBinaryType>(Bin->Format);

  // // For compressed images, we delay determining the format until the image
  // is
  // // decompressed.
  // if (Format == SYCL_DEVICE_BINARY_TYPE_NONE)
  //   // try to determine the format; may remain "NONE"
  //   Format = ur::getBinaryImageFormat(Bin->BinaryStart, getSize());

  // SpecConstIDMap.init(Bin, __SYCL_PROPERTY_SET_SPEC_CONST_MAP);
  // SpecConstDefaultValuesMap.init(
  //     Bin, __SYCL_PROPERTY_SET_SPEC_CONST_DEFAULT_VALUES_MAP);
  // DeviceLibMetadata.init(Bin, __SYCL_PROPERTY_SET_DEVICELIB_METADATA);
  // KernelParamOptInfo.init(Bin, __SYCL_PROPERTY_SET_KERNEL_PARAM_OPT_INFO);
  // ImplicitLocalArg.init(Bin, __SYCL_PROPERTY_SET_SYCL_IMPLICIT_LOCAL_ARG);
  // ProgramMetadata.init(Bin, __SYCL_PROPERTY_SET_PROGRAM_METADATA);
  // // Convert ProgramMetadata into the UR format
  // for (const auto &Prop : ProgramMetadata) {
  //   ProgramMetadataUR.push_back(
  //       ur::mapDeviceBinaryPropertyToProgramMetadata(Prop));
  // }
  // KernelNames.init(Bin, __SYCL_PROPERTY_SET_SYCL_KERNEL_NAMES);
  // ExportedSymbols.init(Bin, __SYCL_PROPERTY_SET_SYCL_EXPORTED_SYMBOLS);
  // ImportedSymbols.init(Bin, __SYCL_PROPERTY_SET_SYCL_IMPORTED_SYMBOLS);
  // DeviceGlobals.init(Bin, __SYCL_PROPERTY_SET_SYCL_DEVICE_GLOBALS);
  // DeviceRequirements.init(Bin, __SYCL_PROPERTY_SET_SYCL_DEVICE_REQUIREMENTS);
  // VirtualFunctions.init(Bin, __SYCL_PROPERTY_SET_SYCL_VIRTUAL_FUNCTIONS);
  // RegisteredKernels.init(Bin, __SYCL_PROPERTY_SET_SYCL_REGISTERED_KERNELS);
  // Misc.init(Bin, __SYCL_PROPERTY_SET_SYCL_MISC_PROP);
}

// std::atomic<uintptr_t> RTDeviceBinaryImage::ImageCounter = 1;

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
