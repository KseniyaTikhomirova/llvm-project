//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/device_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/program_manager.hpp>
#include <detail/queue_impl.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

QueueImpl::QueueImpl(DeviceImpl &deviceImpl, const async_handler &asyncHandler,
                     const property_list &propList, PrivateTag)
    : MIsInorder(false), MAsyncHandler(asyncHandler), MPropList(propList),
      MDevice(deviceImpl),
      MContext(MDevice.getPlatformImpl().getDefaultContext()) {
  auto Result = olCreateQueue(MDevice.getHandle(), &MOffloadQueue);
  if (!isSuccess(Result))
    throw;
      }

backend QueueImpl::getBackend() const noexcept { return MDevice.getBackend(); }

void QueueImpl::setKernelParameters(std::vector<const EventImplPtr *> &&Events,
                                    const detail::UnifiedRangeView &Range) {
  //   "The input events can be from any queue on any device provided by the
  //   same platform as `Queue`.",
  // check

  // This convertion and storing only offload events is possible only while we
  // don't have host tasks (and featured based on host tasks, like streams).
  // With them - it is very likely we should copy EventImplPtr (shared_ptr) and
  // keep it here. Although it may differ if host tasks will be implemented on
  // offload level (no data now).
  assert(MCurrentSubmitInfo.DepEvents.empty() &&
         "Kernel submission must clean up.");
  MCurrentSubmitInfo.DepEvents.reserve(Events.size());
  for (auto &Event : Events) {
    assert(Event && "Event impl object can't be nullptr");
    MCurrentSubmitInfo.DepEvents.push_back((*Event)->getHandle());
  }
  setKernelLaunchArgs(MCurrentSubmitInfo.Range, Range);
}

// rvalue to ArgCollection?
void QueueImpl::submitKernelImpl(const char *KernelName,
                                 detail::ArgCollection &TypelessArgs) {
  // to create progrma & kernel
  ol_symbol_handle_t Kernel =
      detail::ProgramManager::getInstance().getOrCreateKernel(
          KernelName, MDevice.getHandle());
  assert(Kernel);

  // canEnqueueDirectly should check accessor presence & host task dependency
  // (incl. streams).

  // add env var to check copy and validness
  //  deep copy + move enqueue to getLastEvent for verification
  if (!canEnqueueDirectly()) {
    TypelessArgs.deepCopy();
    // keep it saved in command and schedule command for enqueue
  }

  ol_event_handle_t NewEvent{};
  if (!MCurrentSubmitInfo.DepEvents.empty()) {
    if (!isSuccess(olWaitEvents(MOffloadQueue, MCurrentSubmitInfo.DepEvents.data(),
                                MCurrentSubmitInfo.DepEvents.size())))
      throw;
  }
  ol_kernel_launch_prop_t Props[2];
  Props[0].type = OL_KERNEL_LAUNCH_PROP_TYPE_SIZE;
  Props[0].data = TypelessArgs.getSizesArray();
  Props[1] = OL_KERNEL_LAUNCH_PROP_END;
  auto Result = olLaunchKernel(MOffloadQueue, MDevice.getHandle(), Kernel,
                               TypelessArgs.getArgumentsArray(),
                               TypelessArgs.getArgumentCount() * sizeof(void *),
                               &MCurrentSubmitInfo.Range, Props);
  // clean up current kernel submit data to prepare structures for next
  // submission.
  MCurrentSubmitInfo.DepEvents.clear();
  MCurrentSubmitInfo.Range = {};
  if (!isSuccess(Result))
    throw;

  Result = olCreateEvent(MOffloadQueue, &NewEvent);
  if (!isSuccess(Result))
    throw;

  MCurrentSubmitInfo.LastEvent =
      EventImpl::createEventWithHandle(NewEvent, MDevice.getPlatformImpl());
}

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
