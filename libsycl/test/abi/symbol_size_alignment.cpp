// RUN: %clangxx %sycl_options %s -o %t

// Changing symbol size or alignment is an ABI breaking change.

#include <sycl/sycl.hpp>

using namespace sycl;

template <int newSize, int oldSize> void check_size() {
  static_assert(newSize == oldSize, "Symbol size has changed.");
}

template <int newAlignment, int oldAlignment> void check_alignment() {
  static_assert(newAlignment == oldAlignment, "Alignment has changed");
}

template <typename T, size_t oldSize, size_t oldAlignment> void check() {
  check_size<sizeof(T), oldSize>();
  check_alignment<alignof(T), oldAlignment>();
}

int main() {
  // sycl::platform and sycl::device contain only raw pointer
  check<platform, 8, 8>();
  check<device, 8, 8>();

  return 0;
}