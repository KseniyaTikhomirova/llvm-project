//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/device_impl.hpp>
#include <detail/queue_impl.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

QueueImpl::QueueImpl(DeviceImpl &deviceImpl, const async_handler &asyncHandler,
                     const property_list &propList, PrivateTag)
    : MIsInorder(false), MAsyncHandler(asyncHandler), MPropList(propList),
      MDevice(deviceImpl),
      MContext(MDevice.getPlatformImpl().getDefaultContext()) {}

backend QueueImpl::getBackend() const noexcept { return MDevice.getBackend(); }

// used for demonstration only. Without accessor and host task there is no valid
// scenario to delay enqueue.
static bool canEnqueueDirectly() { return true; }

// extend for parallel_for
static void setKernelLaunchArgs(ol_kernel_launch_size_args_t &ArgsToSet) {
  ArgsToSet.Dimensions = 1;
  ArgsToSet.NumGroups.x = 1;
  ArgsToSet.NumGroups.y = 1;
  ArgsToSet.NumGroups.z = 1;
  ArgsToSet.GroupSize.x = 1;
  ArgsToSet.GroupSize.y = 1;
  ArgsToSet.GroupSize.z = 1;
  ArgsToSet.DynSharedMemory = 0;
}
// rvalue to ArgCollection?
void submitKernelImpl(const char *KernelName,
                      detail::ArgCollection &TypelessArgs) {
  // to create progrma & kernel
  ol_symbol_handle_t Kernel;

  ol_kernel_launch_size_args_t LaunchArgs{};
  setKernelLaunchArgs(LaunchArgs);

  // canEnqueueDirectly should check accessor presence & host task dependency
  // (incl. streams).

  // add env var to check copy and validness
  //  deep copy + move enqueue to getLastEvent for verification
  if (!canEnqueueDirectly()) {
    TypelessArgs.deepCopy();
    // keep it saved in command and schedule command for enqueue
  }

  ol_event_handle_t NewEvent{};
  {
    std::lock_guard<std::mutex> Guard(MMutex);
    if (!MDepEvents.empty()) {
      std::vector<EventImplPtr> SwappedDepEvents;
      SwappedDepEvents.swap(MDepEvents);
      // do conversion to offload event handles
      if (!isSuccess(olWaitEvents(MOffloadQueue, SwappedDepEvents.data(),
                                  SwappedDepEvents.size())))
        throw;
    }

    auto Result = olLaunchKernel(MOffloadQueue, MDevice, Kernel,
                                 TypelessArgs.getArgumentsArray(),
                                 TypelessArgs.getSizesArray(),
                                 TypelessArgs.getArgumentCount(), &LaunchArgs);
    if (!isSuccess(Result))
      throw;

    Result = olCreateEvent(MOffloadQueue, &NewEvent);
    if (!isSuccess(Result))
      throw;
  }
  MLastEvent = new EventImpl(NewEvent, MOffloadQueue);
}

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
