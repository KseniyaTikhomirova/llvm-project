// REQUIRES: any-device
// UNSUPPORTED: *
// RUN: %clangxx -fsycl %sycl_options %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

int *p = nullptr;

int main() {
  sycl::queue q;
  p = sycl::malloc_shared<int>(1, q);
  *p = 0;
  q.single_task<class test>([=]() { *p = 42; }).wait();
  assert(*p == 42);
  sycl::free(p, q);
  return 0;
}