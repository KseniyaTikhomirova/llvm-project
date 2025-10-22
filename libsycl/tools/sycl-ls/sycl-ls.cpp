//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The "sycl-ls" utility lists all platforms discovered by SYCL.
//
// There are two types of output:
//   concise (default) and
//   verbose (enabled with --verbose).
//
#include <sycl/sycl.hpp>

#include <iostream>

using namespace sycl;
using namespace std::literals;

// Controls verbose output vs. concise.
bool verbose{};

static int printUsageAndExit() {
  std::cout << "Usage: sycl-ls [--verbose]" << std::endl;
  std::cout << "This program lists all backends discovered by SYCL."
            << std::endl;
  std::cout << "\n Options:" << std::endl;
  std::cout << "\t --verbose "
            << "\t Verbosely prints all the discovered platforms. "
            << std::endl;

  return EXIT_FAILURE;
}

int main(int argc, char **argv) {
  if (argc == 1) {
    verbose = false;
  } else {
    // Parse CLI options.
    for (int i = 1; i < argc; i++) {
      if (argv[i] == "--verbose"sv)
        verbose = true;
      else
        return printUsageAndExit();
    }
  }

  try {
    const auto &Platforms = platform::get_platforms();

    if (Platforms.size() == 0) {
      std::cout << "No platforms found." << std::endl;
    }

    for (const auto &Platform : Platforms) {
      backend Backend = Platform.get_backend();
      std::cout << "[" << detail::get_backend_name(Backend) << ":"
                << "unknown" << "]" << std::endl;
    }

    if (verbose) {
      std::cout << "\nPlatforms: " << Platforms.size() << std::endl;
      uint32_t PlatformNum = 0;
      for (const auto &Platform : Platforms) {
        backend Backend = Platform.get_backend();
        ++PlatformNum;
        auto PlatformVersion = Platform.get_info<info::platform::version>();
        auto PlatformName = Platform.get_info<info::platform::name>();
        auto PlatformVendor = Platform.get_info<info::platform::vendor>();
        std::cout << "Platform [#" << PlatformNum << "]:" << std::endl;
        std::cout << "    Version  : " << PlatformVersion << std::endl;
        std::cout << "    Name     : " << PlatformName << std::endl;
        std::cout << "    Vendor   : " << PlatformVendor << std::endl;

        std::cout << "    Devices  : " << "unknown" << std::endl;
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