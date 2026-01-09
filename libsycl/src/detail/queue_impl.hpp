//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_QUEUE_IMPL
#define _LIBSYCL_QUEUE_IMPL

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/queue.hpp>

#include <detail/device_impl.hpp>

#include <OffloadAPI.h>

#include <memory>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

class ContextImpl;

class QueueImpl : public std::enable_shared_from_this<QueueImpl> {
  struct PrivateTag {
    explicit PrivateTag() = default;
  };

public:
  ~QueueImpl() = default;

  explicit QueueImpl(const device_impl &deviceImpl,
                     const async_handler &asyncHandler,
                     const property_list &propList, PrivateTag)
      : MIsInorder(false), MAsyncHandler(asyncHandler), MPropList(propList),
        MDevice(deviceImpl),
        MContext(MDevice.getPlatformImpl().getDefaultContext()) {}

  template <typename... Ts>
  static std::shared_ptr<QueueImpl> create(Ts &&...args) {
    return std::make_shared<QueueImpl>(std::forward<Ts>(args)..., PrivateTag{});
  }

  backend getBackend() const noexcept { return MDevice.getBackend(); }

  ContextImpl &getContext() const { return MContext; }

  device_impl &getDevice() const { return MDevice; }

  bool isInOrder() const { return MIsInorder; }

private:
  // ol_queue_handle_t MOffloadQueue = {};
  const bool MIsInorder;
  const async_handler MAsyncHandler;
  const property_list MPropList;
  const device_impl &MDevice;
  const ContextImpl &MContext;
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_QUEUE_IMPL
