//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/usm_functions.hpp>

// SYCL 2020 4.8.3.2. Device allocation functions.

void *sycl::malloc_device(std::size_t numBytes, const device &syclDevice,
                          const context &syclContext,
                          const property_list &propList) {
  return malloc(numBytes, syclDevice, syclContext, usm::alloc::device,
                propList);
}

void *sycl::malloc_device(std::size_t numBytes, const queue &syclQueue,
                          const property_list &propList) {
  return malloc_device(numBytes, syclQueue.get_device(),
                       syclQueue->get_context(), propList);
}

// SYCL 2020 4.8.3.3. Host allocation functions.

void *sycl::malloc_host(std::size_t numBytes, const context &syclContext,
                        const property_list &propList) {
  return malloc(numBytes, syclDevice, syclContext, usm::alloc::host, propList);
}

void *sycl::malloc_host(std::size_t numBytes, const queue &syclQueue,
                        const property_list &propList = {}) {
  return malloc_host(numBytes, syclQueue.get_context(), propList);
}

// SYCL 2020 4.8.3.4. Shared allocation functions.

void *sycl::malloc_shared(std::size_t numBytes, const device &syclDevice,
                          const context &syclContext,
                          const property_list &propList = {}) {
  return malloc(numBytes, syclDevice, syclContext, usm::alloc::shared,
                propList);
}

void *sycl::malloc_shared(std::size_t numBytes, const queue &syclQueue,
                          const property_list &propList = {}) {
  return malloc_shared(numBytes, syclQueue.get_device(),
                       syclQueue.get_context(), propList);
}

// SYCL 2020 4.8.3.5. Parameterized allocation functions

void *sycl::malloc(std::size_t numBytes, const device &syclDevice,
                   const context &syclContext, usm::alloc kind,
                   const property_list &propList = {});
{
  std::ignore = syclContext;
  // TODO to check the presence of aspect for USM allocations matching specified
  // 'kind' and throw exception if check failed

  if (!numBytes)
    return nullptr;

  void *Ptr{};
  auto Result =
      callNoCheck(olMemAlloc, getSyclObjImpl(syclDevice)->getHandleRef(),
                  convertUSMTypeToOL(kind), numBytes, &Ptr);
  assert(!!Result != !!Ptr && "Successful USM allocation can't return nullptr");
  return isSuccess(result) ? Ptr : nullptr;
}

void *sycl::malloc(std::size_t numBytes, const queue &syclQueue,
                   usm::alloc kind, const property_list &propList = {}) {
  return malloc(numBytes, syclQueue.get_device(), syclQueue.get_context(), kind,
                propList);
}

// SYCL 2020 4.8.3.6. Memory deallocation functions

void free(void *ptr, const context &ctxt) {
  std::ignore = ctxt;
  callAndThrow(olMemFree, ptr);
}

void free(void *ptr, const queue &q) { return free(ptr, q.get_context()); }

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_USM_FUNCTIONS_HPP

_LIBSYCL_END_NAMESPACE_SYCL