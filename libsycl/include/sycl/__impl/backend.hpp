//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL enum class backend that is
/// implementation-defined and is populated with a unique identifier for each
/// SYCL backend that the SYCL implementation can support.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_BACKEND_HPP
#define _LIBSYCL___IMPL_BACKEND_HPP

#include <sycl/__impl/detail/config.hpp> // namespace macro

_LIBSYCL_BEGIN_NAMESPACE_SYCL

// 4.1. Backends
enum class backend : char {
  opencl = 1,
  level_zero = 2,
  cuda = 3,
  all = 4,
  hip = 6,
};

// 4.5.1.1. Type traits backend_traits
template <backend Backend> class backend_traits;

template <backend Backend, typename SYCLObjectT>
using backend_input_t =
    typename backend_traits<Backend>::template input_type<SYCLObjectT>;
template <backend Backend, typename SYCLObjectT>
using backend_return_t =
    typename backend_traits<Backend>::template return_type<SYCLObjectT>;

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_BACKEND_HPP
