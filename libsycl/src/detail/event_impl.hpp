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


class EventImpl {
  struct PrivateTag {
    explicit PrivateTag() = default;
  };

public:
  event_impl(private_tag)
  {  }

  event_impl(ol_event_handle_t Event, QueueImpl &Queue, private_tag): MOffloadEvent(Event), MQueue(Queue)
  { }

  static std::shared_ptr<event_impl> createDefaultEvent() {
    return std::make_shared<event_impl>(private_tag{});
  }

  event_impl::~event_impl() {
    // consider where to report errors
    if (MOffloadEvent)
      std::ignore = olDestroyEvent(MOffloadEvent);
}

  backend getBackend() const noexcept
  {
    // to handle default cosntructed
    //  The event is constructed as though it were created from a default-constructed queue. Therefore, its backend is the same as the backend of the device selected by default_selector_v.
    return MContext->getBackend();
  }

  void wait()
  {
    // MOffloadEvent == nullptr when event is default constructed. Default constructed event is immediately  ready.
    if (!MOffloadEvent)
      return;

      // No error reporting is declared for sycl::event::wait failures.
      std::ignore = olSyncEvent(MOFfloadEvent);
  }

private:
  ol_event_handle_t MOffloadEvent{};
  std::weak_ptr<QueueImpl> MQueue;
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_EVENT_IMPL
