//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of the SYCL 2020 exception class interface
/// (4.13.2.)
///
//===----------------------------------------------------------------------===//

// 4.9.2 Exception Class Interface
// #include <detail/global_handler.hpp>
// #include <sycl/context.hpp>
#include <sycl/exception.hpp>

// #include <cstring>
// #include <sstream>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail{
class SYCLCategory : public std::error_category {
public:
  const char *name() const noexcept override { return "sycl"; }
  std::string message(int) const override { return "SYCL Error"; }
};
} //namespace detail

// Free functions
const std::error_category &sycl_category() noexcept {
  static const detail::SYCLCategory SYCLCategoryObj;
  return SYCLCategoryObj;
}

std::error_code make_error_code(sycl::errc Err) noexcept {
  return {Err, sycl_category()};
}

// Exception methods implementation
exception::exception(std::error_code EC, const char *Msg)
    : exception(EC, nullptr, Msg) {}

exception::exception(std::error_code EC) : exception(EC, nullptr, "") {}

exception::exception(int EV, const std::error_category &ECat,
                     const char *WhatArg)
    // ktikhomi char* to string looks suspicious
    : exception({EV, ECat}, nullptr, std::string(WhatArg)) {}

exception::exception(int EV, const std::error_category &ECat)
    : exception({EV, ECat}, nullptr, "") {}

// protected base constructor for all SYCL 2020 constructors
// exception::exception(std::error_code EC, std::shared_ptr<context>
// SharedPtrCtx,
//                      const char *WhatArg)
//     : MMsg(std::make_shared<detail::string>(WhatArg)),
//       MErr(UR_RESULT_ERROR_INVALID_VALUE), MContext(SharedPtrCtx), MErrC(EC)
//       {
// }

exception::~exception() {}

const std::error_code &exception::code() const noexcept { return MErrC; }

const std::error_category &exception::category() const noexcept {
  return code().category();
}

const char *exception::what() const noexcept { return MMsg.data(); }

bool exception::has_context() const noexcept { return (MContext != nullptr); }

context exception::get_context() const {
  if (!has_context())
    throw sycl::exception(sycl::errc::invalid);

  return *MContext;
}

_LIBSYCL_END_NAMESPACE_SYCL
