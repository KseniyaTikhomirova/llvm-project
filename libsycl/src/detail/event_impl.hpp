//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_EVENT_IMPL
#define _LIBSYCL_EVENT_IMPL

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/queue.hpp>

#include <OffloadAPI.h>

#include <memory>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

class PlatformImpl;

class EventImpl {
  struct PrivateTag {
    explicit PrivateTag() = default;
  };

public:
  EventImpl(ol_event_handle_t Event, PlatformImpl &Platform, PrivateTag)
      : MOffloadEvent(Event), MPlatform(Platform) {}

  // static std::shared_ptr<EventImpl> createDefaultEvent() {
  //   return std::make_shared<EventImpl>(PrivateTag{});
  // }

  static std::shared_ptr<EventImpl>
  createEventWithHandle(ol_event_handle_t Event, PlatformImpl &Queue) {
    return std::make_shared<EventImpl>(Event, Queue, PrivateTag{});
  }

  ~EventImpl() {
    // consider where to report errors
    if (MOffloadEvent)
      std::ignore = olDestroyEvent(MOffloadEvent);
  }

  backend getBackend() const noexcept;

  void wait() {
    // MOffloadEvent == nullptr when event is default constructed. Default
    // constructed event is immediately  ready.
    if (!MOffloadEvent)
      return;

    // No error reporting is declared for sycl::event::wait failures.
    std::ignore = olSyncEvent(MOffloadEvent);
  }

  ol_event_handle_t getHandle() { return MOffloadEvent; }

private:
  ol_event_handle_t MOffloadEvent{};
  PlatformImpl &MPlatform;
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_EVENT_IMPL
