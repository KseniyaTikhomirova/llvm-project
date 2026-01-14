//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_CONTEXT_IMPL
#define _LIBSYCL_CONTEXT_IMPL

#include <sycl/__impl/context.hpp>
#include <sycl/__impl/detail/config.hpp>

#include <OffloadAPI.h>

#include <functional>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

class platform_impl;
class device_impl;

class ContextImpl : public std::enable_shared_from_this<ContextImpl> {
  struct Private {
    explicit Private() = default;
  };

public:
  ContextImpl(platform_impl &Platform, Private) : MPlatform(Platform) {}

  template <typename... Ts>
  static std::shared_ptr<ContextImpl> create(Ts &&...args) {
    return std::make_shared<ContextImpl>(std::forward<Ts>(args)..., Private{});
  }

  platform_impl &getPlatformImpl() const { return MPlatform; }

  void iterateDevices(const std::function<void(device_impl *)> &callback) const;

  backend getBackend() const;

private:
  platform_impl &MPlatform;

  // ol_device_handle_t MContext{};
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_CONTEXT_IMPL
