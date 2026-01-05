//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_LIBSYCL_BEGIN_NAMESPACE_SYCL

explicit queue(const device &syclDevice, const async_handler &asyncHandler,
               const property_list &propList) {
  impl = detail::QueueImpl::create(*detail::getSyclObjImpl(syclDevice),
                                   asyncHandler, PropList);
}

backend get_backend() const noexcept { return impl->getBackend(); }

context get_context() const {
  return detail::createSyclObjFromImpl<context>(impl->getContext());
}

device get_device() const {
  return detail::createSyclObjFromImpl<device>(impl->getDevice());
}

bool is_in_order() const { return impl->isInOrder(); }

template <typename Param> typename Param::return_type get_info() const;

_LIBSYCL_END_NAMESPACE_SYCL
