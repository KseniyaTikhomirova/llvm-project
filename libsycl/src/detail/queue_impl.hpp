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
  /// a propList.
  ///
  /// \param deviceImpl is a SYCL device that is used to dispatch tasks
  /// submitted to the queue.
  /// \param asyncHandler is a SYCL asynchronous exception handler.
  /// \param propList is a list of properties to use for queue construction.
  explicit QueueImpl(DeviceImpl &deviceImpl, const async_handler &asyncHandler,
                     const property_list &propList, PrivateTag);

  /// Constructs a QueueImpl with the provided arguments. Variadic helper.
  /// Restricts QueueImpl creation to std::shared_ptr allocations.
  template <typename... Ts>
  static std::shared_ptr<QueueImpl> create(Ts &&...args) {
    return std::make_shared<QueueImpl>(std::forward<Ts>(args)..., PrivateTag{});
  }

  /// \return the SYCL backend this queue is associated with.
  backend getBackend() const noexcept;

  /// \return the context implementation object this queue is associated with.
  ContextImpl &getContext() { return MContext; }

  /// \return the device implementation object this queue is associated with.
  DeviceImpl &getDevice() { return MDevice; }

  /// \return true if and only if the queue is in order.
  bool isInOrder() const { return MIsInorder; }

  void submitKernelImpl(const char *KernelName,
                        detail::ArgCollection &TypelessArgs);

  EventImplPtr getLastEvent() {
    assert(MCurrentSubmitInfo.LastEvent &&
           "getLastEvent must be called after enqueue");
    return MCurrentSubmitInfo.LastEvent;
  }

  void setKernelParameters(std::vector<const EventImplPtr *> &&Events,
                           const detail::UnifiedRangeView &Range);

  // extend for parallel_for
  static void setKernelLaunchArgs(ol_kernel_launch_size_args_t &ArgsToSet,
                                  const detail::UnifiedRangeView &Range) {
  size_t GlobalSize[3] = {1, 1, 1};
  if (Range.MGlobalSize) {
    for (uint32_t I = 0; I < Range.MDims; I++) {
      GlobalSize[I] = Range.MGlobalSize[I];
    }
  }

  size_t GroupSize[3] = {1, 1, 1};
  if (Range.MLocalSize) {
    for (uint32_t I = 0; I < Range.MDims; I++) {
      GroupSize[I] = Range.MLocalSize[I];
    }
  }

  ol_kernel_launch_size_args_t LaunchArgs;
  LaunchArgs.Dimensions = Range.MDims;
  LaunchArgs.NumGroups.x = GlobalSize[0] / GroupSize[0];
  LaunchArgs.NumGroups.y = GlobalSize[1] / GroupSize[1];
  LaunchArgs.NumGroups.z = GlobalSize[2] / GroupSize[2];
  LaunchArgs.GroupSize.x = GroupSize[0];
  LaunchArgs.GroupSize.y = GroupSize[1];
  LaunchArgs.GroupSize.z = GroupSize[2];
  LaunchArgs.DynSharedMemory = 0;
  }

  // used for demonstration only. Without accessor and host task there is no
  // valid scenario to delay enqueue.
  static bool canEnqueueDirectly() { return true; }

  void wait() {
    auto Result = olSyncQueue(MOffloadQueue);
    if (isFailed(Result))
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
  struct KernelSubmitInfo {
    EventImplPtr LastEvent;
    // range to be replaced with copy or OL data
    ol_kernel_launch_size_args_t Range;
    // consider storing EventImplPtr here, it will work with plain handle only
    // because submission is done within queue::submit call. Otherwise we need
    // to ensure that event handle is still alive by keeping our own copy of
    // EventImpl.
    std::vector<ol_event_handle_t> DepEvents;
  };
  inline static thread_local KernelSubmitInfo MCurrentSubmitInfo = {};
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_QUEUE_IMPL
