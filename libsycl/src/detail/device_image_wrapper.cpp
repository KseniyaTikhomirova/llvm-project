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

DeviceImageWrapper::DeviceImageWrapper(const __sycl_tgt_device_image &Bin)
    : MBin(&Bin) {}

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
