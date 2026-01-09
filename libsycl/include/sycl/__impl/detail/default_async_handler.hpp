//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains definition of an implementation-defined default
/// async_handler which is invoked when an asynchronous error occurs in a queue
/// or context that has no user-supplied asynchronous error handler object (see
/// SYCL 2020 4.13.1.2).
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_DETAIL_DEFAULT_ASYNC_HANDLER_HPP
#define _LIBSYCL___IMPL_DETAIL_DEFAULT_ASYNC_HANDLER_HPP

#include <sycl/__impl/exception.hpp>

#include <iostream>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

// ktikhomi: TODO implement exclude from ABI stuff
inline void defaultAsyncHandler(exception_list ExceptionList) {
  std::cerr
      << "Implementation-defined default async_handler caught exceptions:";
  for (auto &Exception : ExceptionList) {
    try {
      if (Exception) {
        std::rethrow_exception(Exception);
      }
    } catch (const std::exception &E) {
      std::cerr << "\n\t" << E.what();
    }
  }
  std::cerr << std::endl;
  std::terminate();
}

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_DETAIL_DEFAULT_ASYNC_HANDLER_HPP
