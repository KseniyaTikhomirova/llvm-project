//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_USM_ALLOC_TYPE_HPP
#define _LIBSYCL___IMPL_USM_ALLOC_TYPE_HPP

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/macro_definitions.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace usm {

enum class alloc : char { host = 0, device = 1, shared = 2, unknown = 3 };

}

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_USM_ALLOC_TYPE_HPP
