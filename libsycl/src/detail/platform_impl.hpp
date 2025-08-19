//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/backend.hpp>       // sycl::backend
#include <sycl/__impl/detail/config.hpp> // namespace macro
#include <sycl/__impl/platform.hpp>      // sycl::platform

#include <memory> // std::enable_shared_from_this
#include <vector> // std::vector

#include <detail/ur/adapter_impl.hpp> // adapter_impl

#ifndef _LIBSYCL_PLATFORM_IMPL
#  define _LIBSYCL_PLATFORM_IMPL

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

class platform_impl : public std::enable_shared_from_this<platform_impl> {
public:
  /// Constructs platform_impl from a platform handle.
  ///
  /// \param Platform is a raw offload library handle representing platform.
  /// \param Adapter is an offload library handle representing Adapter.
  //
  // Platforms can only be created under `GlobalHandler`'s ownership via
  // `platform_impl::getOrMakePlatformImpl` method.
  explicit platform_impl(ur_platform_handle_t Platform, adapter_impl &Adapter,
                         const backend &Backend);

  ~platform_impl() = default;

  /// Returns the backend associated with this platform.
  backend getBackend() const noexcept { return MBackend; }

  /// Returns all SYCL platforms from all backends that are available in the
  /// system.
  static std::vector<platform> getPlatforms();

  /// Returns raw underlying offload platform handle.
  ///
  /// Unlike get() method, this method does not retain handle. It is caller
  /// responsibility to make sure that platform stays alive while raw handle
  /// is in use.
  ///
  /// \return a raw plug-in platform handle.
  const ur_platform_handle_t &getHandleRef() const { return MPlatform; }

  /// Queries the cache to see if the specified offloading RT platform has been
  /// seen before.  If so, return the cached platform_impl, otherwise create a
  /// new one and cache it.
  ///
  /// \param UrPlatform is the offloading RT Platform handle representing the
  /// platform
  /// \param Adapter is the offloading RT adapter providing the backend for the
  /// platform
  /// \return the platform_impl representing the offloading RT platform
  static platform_impl *getOrMakePlatformImpl(ur_platform_handle_t UrPlatform,
                                              adapter_impl &Adapter);

private:
  // Helper to get the vector of platforms supported by a given UR adapter
  static std::vector<platform> getAdapterPlatforms(adapter_impl *&Adapter);

  ur_platform_handle_t MPlatform{};
  adapter_impl &MAdapter;
  backend MBackend;
};

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_PLATFORM_IMPL
