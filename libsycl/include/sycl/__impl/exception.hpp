//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL 2020 Exception class
/// interface (4.13.2.)
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_EXCEPTION_HPP
#define _LIBSYCL___IMPL_EXCEPTION_HPP

#include <sycl/__impl/detail/config.hpp> // namespace macro

#include <exception> // for exception
// #include <memory>       // for allocator, shared_ptr, make...
#include <string>       // for string, basic_string, opera...
#include <system_error> // for error_code, error_category
#include <type_traits>  // for true_type

_LIBSYCL_BEGIN_NAMESPACE_SYCL

// Forward declaration
class context;

enum class errc : int {
  success = 0,
  runtime = 1,
  kernel = 2,
  accessor = 3,
  nd_range = 4,
  event = 5,
  kernel_argument = 6,
  build = 7,
  invalid = 8,
  memory_allocation = 9,
  platform = 10,
  profiling = 11,
  feature_not_supported = 12,
  kernel_not_supported = 13,
  backend_mismatch = 14,
};

/// Constructs an error code using e and sycl_category()
__SYCL_EXPORT std::error_code make_error_code(sycl::errc E) noexcept;

/// Obtains a reference to the static error category object for SYCL errors.
__SYCL_EXPORT const std::error_category &sycl_category() noexcept;

// Derive from std::exception so uncaught exceptions are printed in c++ default
// exception handler.
// 4.13.2. Exception class interface
/// \ingroup sycl_api
class __SYCL_EXPORT exception : public virtual std::exception {
public:
  exception(std::error_code, const char *);
  exception(std::error_code Ec, const std::string &Msg)
      : exception(Ec, nullptr, Msg.c_str()) {}

  exception(std::error_code);
  exception(int EV, const std::error_category &ECat, const std::string &WhatArg)
      : exception(EV, ECat, WhatArg.c_str()) {}
  exception(int, const std::error_category &, const char *);
  exception(int, const std::error_category &);

  // context.hpp depends on exception.hpp but we can't define these ctors in
  // exception.hpp while context is still an incomplete type.
  // So, definition of ctors that require a context parameter are moved to
  // context.hpp.
  exception(context, std::error_code, const std::string &);
  exception(context, std::error_code, const char *);
  exception(context, std::error_code);
  exception(context, int, const std::error_category &, const std::string &);
  exception(context, int, const std::error_category &, const char *);
  exception(context, int, const std::error_category &);

  virtual ~exception();

  const std::error_code &code() const noexcept;
  const std::error_category &category() const noexcept;

  const char *what() const noexcept final;

  bool has_context() const noexcept;

  context get_context() const;

private:
  // ktikhoim: to check Exceptions must be noexcept copy constructible
  std::vector<char> MMsg;
  std::error_code MErrC = make_error_code(sycl::errc::invalid);
  // ktikhomi: why is it shptr to sycl context, not impl? I don't like it
  std::shared_ptr<context> MContext;
protected:
  // base constructor for all SYCL 2020 constructors
  // exception(context *, std::error_code, const std::string);
  // exception(std::error_code Ec, std::shared_ptr<context> SharedPtrCtx,
  //           const std::string &what_arg)
  //     : exception(Ec, SharedPtrCtx, what_arg.c_str()) {}
  // exception(std::error_code Ec, std::shared_ptr<context> SharedPtrCtx,
  //           const char *WhatArg);
};

/// Used as a container for a list of asynchronous exceptions
///
/// \ingroup sycl_api
class __SYCL_EXPORT exception_list {
public:
  using value_type = std::exception_ptr;
  using reference = value_type &;
  using const_reference = const value_type &;
  using size_type = std::size_t;
  using iterator = std::vector<std::exception_ptr>::const_iterator;
  using const_iterator = std::vector<std::exception_ptr>::const_iterator;

  size_type size() const;
  // first asynchronous exception
  iterator begin() const;
  // refer to past-the-end last asynchronous exception
  iterator end() const;

private:
  // friend class detail::queue_impl;
  // void PushBack(const_reference Value);
  // void PushBack(value_type &&Value);
  // void Clear() noexcept;
  std::vector<std::exception_ptr> MList;
};

_LIBSYCL_END_NAMESPACE_SYCL

namespace std {
template <> struct is_error_code_enum<sycl::errc> : true_type {};
} // namespace std

#endif // _LIBSYCL___IMPL_EXCEPTION_HPP
