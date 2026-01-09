//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_USM_FUNCTIONS_HPP
#define _LIBSYCL___IMPL_USM_FUNCTIONS_HPP

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/macro_definitions.hpp>

#include <sycl/__impl/context.hpp>
#include <sycl/__impl/queue.hpp>
#include <sycl/__impl/usm_alloc_type.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

// SYCL 2020 4.8.3.2. Device allocation functions.

void *_LIBSYCL_EXPORT malloc_device(std::size_t numBytes,
                                    const device &syclDevice,
                                    const context &syclContext,
                                    const property_list &propList = {});
template <typename T>
T *malloc_device(std::size_t count, const device &syclDevice,
                 const context &syclContext,
                 const property_list &propList = {}) {
  // TODO: to rewrite with aligned_malloc_device once it's supported in
  // liboffload.
  return static_cast<T *>(
      malloc_device(count * sizeof(T), syclDevice, syclContext, propList));
}

void *malloc_device(std::size_t numBytes, const queue &syclQueue,
                    const property_list &propList = {});

template <typename T>
T *malloc_device(std::size_t count, const queue &syclQueue,
                 const property_list &propList = {}) {
  return malloc_device<T>(count, syclQueue.get_device(),
                          syclQueue.get_context(), propList);
}

// SYCL 2020 4.8.3.3. Host allocation functions.

void *malloc_host(std::size_t numBytes, const context &syclContext,
                  const property_list &propList = {});
template <typename T>
T *malloc_host(std::size_t count, const context &syclContext,
               const property_list &propList = {}) {
  // TODO: to rewrite with aligned_malloc_host once it's supported in
  // liboffload.
  return static_cast<T *>(
      malloc_host(count * sizeof(T), syclContext, propList));
}

void *malloc_host(std::size_t numBytes, const queue &syclQueue,
                  const property_list &propList = {});

template <typename T>
T *malloc_host(std::size_t count, const queue &syclQueue,
               const property_list &propList = {}) {
  return malloc_host<T>(count, syclQueue.get_context(), propList);
}

// SYCL 2020 4.8.3.4. Shared allocation functions.

void *malloc_shared(std::size_t numBytes, const device &syclDevice,
                    const context &syclContext,
                    const property_list &propList = {});
template <typename T>
T *malloc_shared(std::size_t count, const device &syclDevice,
                 const context &syclContext,
                 const property_list &propList = {}) {
  // TODO: to rewrite with aligned_malloc_shared once it's supported in
  // liboffload.
  return static_cast<T *>(
      malloc_shared(count * sizeof(T), syclDevice, syclContext, propList));
}

void *malloc_shared(std::size_t numBytes, const queue &syclQueue,
                    const property_list &propList = {});

template <typename T>
T *malloc_shared(std::size_t count, const queue &syclQueue,
                 const property_list &propList = {}) {
  return malloc_shared<T>(count, syclQueue.get_device(),
                          syclQueue.get_context(), propList);
}

// SYCL 2020 4.8.3.5. Parameterized allocation functions

void *malloc(std::size_t numBytes, const device &syclDevice,
             const context &syclContext, usm::alloc kind,
             const property_list &propList = {});

template <typename T>
T *malloc(std::size_t count, const device &syclDevice,
          const context &syclContext, usm::alloc kind,
          const property_list &propList = {}) {
  // TODO: to rewrite with aligned_malloc once it's supported in liboffload.
  return static_cast<T *>(
      malloc(count * sizeof(T), syclDevice, syclContext, kind, propList));
}

void *malloc(std::size_t numBytes, const queue &syclQueue, usm::alloc kind,
             const property_list &propList = {});

template <typename T>
T *malloc(std::size_t count, const queue &syclQueue, usm::alloc kind,
          const property_list &propList = {}) {
  return malloc<T>(count, syclQueue.get_device(), syclQueue.get_context(), kind,
                   propList);
}

// SYCL 2020 4.8.3.6. Memory deallocation functions

void free(void *ptr, const context &ctxt);
void free(void *ptr, const queue &q);

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_USM_FUNCTIONS_HPP
