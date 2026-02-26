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

#include <OffloadAPI.h>

#include <memory>
#include <mutex>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

class ContextImpl;
class DeviceImpl;
class EventImpl;

using EventImplPtr = std::shared_ptr<EventImpl>;

class QueueImpl : public std::enable_shared_from_this<QueueImpl> {
  struct PrivateTag {
    explicit PrivateTag() = default;
  };

public:
  ~QueueImpl() {
    // consider where to report errors
    if (MOffloadQueue)
      std::ignore = olDestroyQueue(MOffloadQueue);
  }

  /// Constructs a SYCL queue from a device using an asyncHandler and
  /// propList provided.
  ///
  /// \param deviceImpl is a SYCL device that is used to dispatch tasks
  /// submitted to the queue.
  /// \param asyncHandler is a SYCL asynchronous exception handler.
  /// \param propList is a list of properties to use for queue construction.
  explicit QueueImpl(DeviceImpl &deviceImpl, const async_handler &asyncHandler,
                     const property_list &propList, PrivateTag);

  /// Constructs a QueueImpl with a provided arguments. Variadic helper.
  /// Restrics ways of QueueImpl creation.
  template <typename... Ts>
  static std::shared_ptr<QueueImpl> create(Ts &&...args) {
    return std::make_shared<QueueImpl>(std::forward<Ts>(args)..., PrivateTag{});
  }

  /// Returns backend this queue is associated with.
  ///
  /// \return SYCL backend.
  backend getBackend() const noexcept;

  /// Returns context this queue is associated with.
  ///
  /// \return context implementation object.
  ContextImpl &getContext() { return MContext; }

  /// Returns device this queue is associated with.
  ///
  /// \return device implementation object.
  DeviceImpl &getDevice() { return MDevice; }

  /// Returns whether the queue is in order or out of order.
  ///
  /// \return true if queue is in order.
  bool isInOrder() const { return MIsInorder; }

  void submitKernelImpl(const char *KernelName,
                        detail::ArgCollection &TypelessArgs);

  EventImplPtr getLastEvent() { 
    assert(MCurrentSubmitInfo.LastEvent && "getLastEvent must be called after enqueue");
    return MCurrentSubmitInfo.LastEvent; }

  void setKernelParameters(std::vector<const EventImplPtr *> &&Events,
                           const detail::UnifiedRangeView &Range);

  // extend for parallel_for
  static void setKernelLaunchArgs(ol_kernel_launch_size_args_t &ArgsToSet,
                                  const detail::UnifiedRangeView &Range) {
    ArgsToSet.Dimensions = Range.MDims;
    // set WG from range
    ArgsToSet.NumGroups.x = 1;
    ArgsToSet.NumGroups.y = 1;
    ArgsToSet.NumGroups.z = 1;
    ArgsToSet.GroupSize.x = 1;
    ArgsToSet.GroupSize.y = 1;
    ArgsToSet.GroupSize.z = 1;
    ArgsToSet.DynSharedMemory = 0;
  }

  // used for demonstration only. Without accessor and host task there is no
  // valid scenario to delay enqueue.
  static bool canEnqueueDirectly() { return true; }

  void wait() {
    auto Result = olSyncQueue(MOffloadQueue);
    if (!isSuccess(Result))
      throw; // or async?
  }

private:
  // Queue features
  ol_queue_handle_t MOffloadQueue = {};
  const bool MIsInorder;
  const async_handler MAsyncHandler;
  const property_list MPropList;
  DeviceImpl &MDevice;
  ContextImpl &MContext;

  // Submit data
  struct KernelSubmitInfo
  {
   EventImplPtr LastEvent;
   // range to be replaced with copy or OL data
   ol_kernel_launch_size_args_t Range;
   // consider storing EventImplPtr here, it will work with plain handle only
   // because submission is done within queue::submit call. Otherwise we need to
   // ensure that event handle is still alive by keeping our own copy of
   // EventImpl.
   std::vector<ol_event_handle_t> DepEvents;
  };
  inline static thread_local KernelSubmitInfo MCurrentSubmitInfo = {};
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_QUEUE_IMPL
