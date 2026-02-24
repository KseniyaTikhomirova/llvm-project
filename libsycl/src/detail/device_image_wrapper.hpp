//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_DEVICE_BINARY_WRAPPER
#define _LIBSYCL_DEVICE_BINARY_WRAPPER

#include <sycl/__impl/detail/config.hpp>

#include <detail/device_binary_structures.hpp>

#include <OffloadAPI.h>

#include <memory>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

class DeviceImageWrapper {
public:
  //   // Represents a range of properties to enable iteration over them.
  //   // Implements the standard C++ STL input iterator interface.
  //   class PropertyRange {
  //   public:
  //     using ValTy = std::remove_pointer<sycl_device_binary_property>::type;

  //     class ConstIterator {
  //       sycl_device_binary_property Cur;

  //     public:
  //       using iterator_category = std::input_iterator_tag;
  //       using value_type = ValTy;
  //       using difference_type = ptrdiff_t;
  //       using pointer = const sycl_device_binary_property;
  //       using reference = sycl_device_binary_property;

  //       ConstIterator(sycl_device_binary_property Cur = nullptr) : Cur(Cur)
  //       {} ConstIterator &operator++() {
  //         Cur++;
  //         return *this;
  //       }
  //       ConstIterator operator++(int) {
  //         ConstIterator Ret = *this;
  //         ++(*this);
  //         return Ret;
  //       }
  //       bool operator==(ConstIterator Other) const { return Cur == Other.Cur;
  //       } bool operator!=(ConstIterator Other) const { return !(*this ==
  //       Other); } reference operator*() const { return Cur; }
  //     };
  //     ConstIterator begin() const { return ConstIterator(Begin); }
  //     ConstIterator end() const { return ConstIterator(End); }
  //     size_t size() const { return std::distance(begin(), end()); }
  //     bool empty() const { return begin() == end(); }
  //     friend class RTDeviceBinaryImage;
  //     friend class DynRTDeviceBinaryImage;
  //     bool isAvailable() const { return !(Begin == nullptr); }

  //   private:
  //     PropertyRange() : Begin(nullptr), End(nullptr) {}
  //     // Searches for a property set with given name and constructs a
  //     // PropertyRange spanning all its elements. If property set is not
  //     found,
  //     // the range will span zero elements.
  //     PropertyRange(sycl_device_binary Bin, const char *PropSetName)
  //         : PropertyRange() {
  //       init(Bin, PropSetName);
  //     };
  //     void init(sycl_device_binary Bin, const char *PropSetName);
  //     sycl_device_binary_property Begin;
  //     sycl_device_binary_property End;
  //   };

public:
  // RTDeviceBinaryImage() : Bin(nullptr) {}
  DeviceImageWrapper(const __sycl_tgt_device_image &Bin);
  //   // Explicitly delete copy constructor/operator= to avoid unintentional
  //   copies RTDeviceBinaryImage(const RTDeviceBinaryImage &) = delete;
  //   RTDeviceBinaryImage &operator=(const RTDeviceBinaryImage &) = delete;
  //   // Explicitly retain move constructors to facilitate potential moves
  //   across
  //   // collections
  //   RTDeviceBinaryImage(RTDeviceBinaryImage &&) = default;
  //   RTDeviceBinaryImage &operator=(RTDeviceBinaryImage &&) = default;

  //   virtual ~RTDeviceBinaryImage() {}

  //   bool supportsSpecConstants() const {
  //     return getFormat() == SYCL_DEVICE_BINARY_TYPE_SPIRV;
  //   }

  //   const sycl_device_binary_struct &getRawData() const { return *get(); }

  //   virtual void print() const;
  //   virtual void dump(std::ostream &Out) const;

  //   virtual size_t getSize() const {
  //     assert(Bin && "binary image data not set");
  //     return static_cast<size_t>(Bin->BinaryEnd - Bin->BinaryStart);
  //   }

  //   const char *getCompileOptions() const {
  //     assert(Bin && "binary image data not set");
  //     return Bin->CompileOptions;
  //   }

  //   const char *getLinkOptions() const {
  //     assert(Bin && "binary image data not set");
  //     return Bin->LinkOptions;
  //   }

  //   /// Returns the format of the binary image
  //   ur::DeviceBinaryType getFormat() const {
  //     assert(Bin && "binary image data not set");
  //     return Format;
  //   }

  //   /// Returns a single property from SYCL_MISC_PROP category.
  //   sycl_device_binary_property getProperty(const char *PropName) const;

  //   const PropertyRange &getKernelNames() const { return KernelNames; }
  //   const PropertyRange &getExportedSymbols() const { return ExportedSymbols;
  //   } const PropertyRange &getDeviceRequirements() const {
  //     return DeviceRequirements;
  //   }

  //   std::uintptr_t getImageID() const {
  //     assert(Bin && "Image ID is not available without a binary image.");
  //     return ImageId;
  //   }

protected:
  const __sycl_tgt_device_image &get() const { return MBin; }

  const __sycl_tgt_device_image &MBin;

  //   ur::DeviceBinaryType Format = SYCL_DEVICE_BINARY_TYPE_NONE;

  //   RTDeviceBinaryImage::PropertyRange KernelNames;
  //   RTDeviceBinaryImage::PropertyRange ExportedSymbols;
  //   RTDeviceBinaryImage::PropertyRange DeviceRequirements;

  //   std::vector<ur_program_metadata_t> ProgramMetadataUR;

  // private:
  //   static std::atomic<uintptr_t> ImageCounter;
  //   uintptr_t ImageId = 0;
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_DEVICE_BINARY_WRAPPER
