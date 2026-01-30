//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL event class, that represents
/// the status of an operation that is being executed by the SYCL runtime
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_EVENT_HPP
#define _LIBSYCL___IMPL_EVENT_HPP

#include <sycl/__impl/backend.hpp>
#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/obj_utils.hpp>

#include <memory>
#include <vector>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {
class EventImpl;
}

class event {
public:
  event(const event &rhs) = default;

  event(event &&rhs) = default;

  event &operator=(const event &rhs) = default;

  event &operator=(event &&rhs) = default;

  friend bool operator==(const event &lhs, const event &rhs) {
    return lhs.impl == rhs.impl;
  }

  friend bool operator!=(const event &lhs, const event &rhs) {
    return !(lhs == rhs);
  }

  backend get_backend() const noexcept;

  void wait();

  static void wait(const std::vector<event> &eventList);

  template <typename Param> typename Param::return_type get_info() const;

  template <typename Param>
  typename Param::return_type get_backend_info() const;

private:
  event(std::shared_ptr<detail::EventImpl> Impl) : impl(Impl) {}
  std::shared_ptr<detail::EventImpl> impl;

  friend sycl::detail::ImplUtils;
};

_LIBSYCL_END_NAMESPACE_SYCL

template <>
struct std::hash<sycl::event> : public sycl::detail::HashBase<sycl::event> {};

#endif // _LIBSYCL___IMPL_EVENT_HPP
