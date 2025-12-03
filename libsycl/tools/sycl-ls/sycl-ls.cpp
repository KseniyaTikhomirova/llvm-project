//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The "sycl-ls" utility lists all platforms & devices discovered by SYCL.
//
// There are two types of output:
//   concise (default) and
//   verbose (enabled with --verbose).
//
#include <sycl/sycl.hpp>

#include "llvm/Support/CommandLine.h"

#include <iostream>

using namespace sycl;
using namespace std::literals;

std::string getDeviceTypeName(const device &Device) {
  auto DeviceType = Device.get_info<info::device::device_type>();
  switch (DeviceType) {
  case info::device_type::cpu:
    return "cpu";
  case info::device_type::gpu:
    return "gpu";
  case info::device_type::host:
    return "host";
  case info::device_type::accelerator:
    return "fpga";
  default:
    return "unknown";
  }
}

static void printDeviceInfo(const device &Device, bool Verbose,
                            const std::string &Prepend) {
  auto DeviceName = Device.get_info<info::device::name>();
  auto DeviceVendor = Device.get_info<info::device::vendor>();
  auto DeviceDriverVersion = Device.get_info<info::device::driver_version>();

  if (Verbose) {
    std::cout << Prepend << "Type              : " << getDeviceTypeName(Device)
              << std::endl;
    std::cout << Prepend << "Name              : " << DeviceName << std::endl;
    std::cout << Prepend << "Vendor            : " << DeviceVendor << std::endl;
    std::cout << Prepend << "Driver            : " << DeviceDriverVersion
              << std::endl;
  } else {
    std::cout << Prepend << ", " << DeviceName << " [" << DeviceDriverVersion
              << "]" << std::endl;
  }
}

int main(int argc, char **argv) {
  llvm::cl::opt<bool> Verbose(
      "verbose",
      llvm::cl::desc("Verbosely prints all the discovered devices"));
  llvm::cl::alias VerboseShort("v", llvm::cl::desc("Alias for -verbose"),
                               llvm::cl::aliasopt(Verbose));
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "This program lists all devices discovered by SYCL");

  try {
    const auto &Platforms = platform::get_platforms();

    if (Platforms.size() == 0) {
      std::cout << "No platforms found." << std::endl;
    }

    for (const auto &Platform : Platforms) {
      backend Backend = Platform.get_backend();
      auto PlatformName = Platform.get_info<info::platform::name>();
      const auto &Devices = Platform.get_devices();

      for (const auto &Device : Devices) {
        std::cout << "[" << detail::get_backend_name(Backend) << ":"
                  << getDeviceTypeName(Device) << "]";
        std::cout << " ";
        // Verbose parameter is set to false to print regular devices output
        // first
        printDeviceInfo(Device, false, PlatformName);
      }
    }

    if (Verbose) {
      std::cout << "\nPlatforms: " << Platforms.size() << std::endl;
      uint32_t PlatformNum = 0;
      for (const auto &Platform : Platforms) {
        ++PlatformNum;
        auto PlatformVersion = Platform.get_info<info::platform::version>();
        auto PlatformName = Platform.get_info<info::platform::name>();
        auto PlatformVendor = Platform.get_info<info::platform::vendor>();
        std::cout << "Platform [#" << PlatformNum << "]:" << std::endl;
        std::cout << "    Version  : " << PlatformVersion << std::endl;
        std::cout << "    Name     : " << PlatformName << std::endl;
        std::cout << "    Vendor   : " << PlatformVendor << std::endl;

        const auto &Devices = Platform.get_devices();
        std::cout << "    Devices  : " << Devices.size() << std::endl;
        for (const auto &Device : Devices) {
          // ktikhomi prepend - can be done better?
          printDeviceInfo(Device, true, "        ");
        }
      }
    } else {
      return EXIT_SUCCESS;
    }
  } catch (sycl::exception &e) {
    std::cerr << "SYCL Exception encountered: " << e.what() << std::endl
              << std::endl;
  }

  return EXIT_SUCCESS;
}
