// RUN: %{build} -Wno-deprecated-declarations -o %t.out
// RUN: %{run} %t.out
//
// Tests platform::get_platforms and get_backend.

#include <sycl/platform.hpp>

std::string BackendToString(sycl::backend Backend) {
  switch (Backend) {
  case sycl::backend::opencl:
    return "opencl";
  case sycl::backend::level_zero:
    return "level_zero";
  case sycl::backend::cuda:
    return "cuda";
  case sycl::backend::all:
    return "all";
  case sycl::backend::hip:
    return "hip";
  default:
    return "UNKNOWN";
  }
}

int main() {
  for (sycl::platform P : sycl::platform::get_platforms()) {
    std::cout << "Checking platform with backend "
              << BackendToString(P.get_backend()) << std::endl;
  }
  return 0;
}