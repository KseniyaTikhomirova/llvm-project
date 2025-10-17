//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/detail/config.hpp>   // namespace macro
#include <sycl/__impl/detail/spinlock.hpp> // class SpinLock
#include <detail/offload/offload_topology.hpp>

#include <mutex>
#include <vector>
#include <memory>
#include <array>

#ifndef _LIBSYCL_GLOBAL_HANDLER
#  define _LIBSYCL_GLOBAL_HANDLER

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {
class platform_impl;

/// Wrapper class for global data structures with non-trivial destructors.
///
/// As user code can call SYCL Runtime functions from destructor of global
/// objects, it is not safe for the runtime library to have global objects with
/// non-trivial destructors. Such destructors can be called any time after
/// exiting main, which may result in user application crashes. Instead,
/// complex global objects must be wrapped into GlobalHandler. Its instance
/// is stored on heap, and deallocated when the runtime library is being
/// unloaded.
///
/// There's no need to store trivial globals here, as no code for their
/// construction or destruction is generated anyway.
class GlobalHandler {
public:
  /// \return a reference to a GlobalHandler singleton instance. Memory for
  /// storing objects is allocated on first call. The reference is valid as long
  /// as runtime library is loaded (i.e. untill `DllMain` or
  /// `__attribute__((destructor))` is called).
  static GlobalHandler &instance();

  GlobalHandler(const GlobalHandler &) = delete;
  GlobalHandler(GlobalHandler &&) = delete;
  GlobalHandler &operator=(const GlobalHandler &) = delete;

  // Offload topologies (one per backend) discovered from liboffload.
  std::array<detail::OffloadTopology, OL_PLATFORM_BACKEND_LAST> &
  getOffloadTopologies();

  std::mutex &getPlatformMapMutex();
  std::vector<std::shared_ptr<platform_impl>> &getPlatformCache();

private:
  static GlobalHandler *&getInstancePtr();
  static SpinLock MInstancePtrProtector;

  friend void shutdown_late();

  // Constructor and destructor are declared out-of-line to allow incomplete
  // types as template arguments to unique_ptr.
  GlobalHandler();
  ~GlobalHandler();

  template <typename T> struct InstWithLock {
    std::unique_ptr<T> Inst;
    SpinLock Lock;
  };

  template <typename T, typename... Types>
  T &getOrCreate(InstWithLock<T> &IWL, Types &&...Args);

  InstWithLock<std::array<detail::OffloadTopology, OL_PLATFORM_BACKEND_LAST>>
      MOffloadTopologies;
  InstWithLock<std::vector<std::shared_ptr<platform_impl>>> MPlatformCache;
  InstWithLock<std::mutex> MPlatformMapMutex;
};
} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_GLOBAL_HANDLER