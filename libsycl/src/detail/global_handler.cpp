//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/global_handler.hpp>
#include <detail/platform_impl.hpp>

#ifdef _WIN32
#  include <windows.h>
#endif

#include <vector>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

using LockGuard = std::lock_guard<SpinLock>;

GlobalHandler::GlobalHandler() = default;
GlobalHandler::~GlobalHandler() = default;

GlobalHandler *&GlobalHandler::getInstancePtr() {
  static GlobalHandler *RTGlobalObjHandler = new GlobalHandler();
  return RTGlobalObjHandler;
}

GlobalHandler &GlobalHandler::instance() {
  GlobalHandler *RTGlobalObjHandler = GlobalHandler::getInstancePtr();
  assert(RTGlobalObjHandler && "Handler must not be deallocated earlier");
  return *RTGlobalObjHandler;
}

template <typename T, typename... Types>
T &GlobalHandler::getOrCreate(InstWithLock<T> &IWL, Types &&...Args) {
  const LockGuard Lock{IWL.Lock};

  if (!IWL.Inst)
    IWL.Inst = std::make_unique<T>(std::forward<Types>(Args)...);

  return *IWL.Inst;
}

std::array<detail::OffloadTopology, OL_PLATFORM_BACKEND_LAST> &
GlobalHandler::getOffloadTopologies() {
  static std::array<detail::OffloadTopology, OL_PLATFORM_BACKEND_LAST>
      &Topologies = getOrCreate(MOffloadTopologies);
  return Topologies;
}

std::vector<std::shared_ptr<platform_impl>> &GlobalHandler::getPlatformCache() {
  static std::vector<std::shared_ptr<platform_impl>> &PlatformCache =
      getOrCreate(MPlatformCache);
  return PlatformCache;
}

std::mutex &GlobalHandler::getPlatformMapMutex() {
  static std::mutex &PlatformMapMutex = getOrCreate(MPlatformMapMutex);
  return PlatformMapMutex;
}

void shutdown_late() {
  GlobalHandler *&Handler = GlobalHandler::getInstancePtr();
  if (!Handler)
    return;

  // First, release resources, that may access offload lib.
  Handler->MPlatformCache.Inst.reset(nullptr);

  // No error reporting in shutdown
  std::ignore = olShutDown();

  // Release the rest of global resources.
  delete Handler;
  Handler = nullptr;
}

#ifdef _WIN32
extern "C" _LIBSYCL_EXPORT BOOL WINAPI DllMain(HINSTANCE hinstDLL,
                                               DWORD fdwReason,
                                               LPVOID lpReserved) {
  // Perform actions based on the reason for calling.
  switch (fdwReason) {
  case DLL_PROCESS_DETACH:
    try {
      shutdown_late();
    } catch (std::exception &e) {
      // report
    }

    break;
  case DLL_PROCESS_ATTACH:
    break;
  case DLL_THREAD_ATTACH:
    break;
  case DLL_THREAD_DETACH:
    break;
  }
  return TRUE; // Successful DLL_PROCESS_ATTACH.
}
#else
// Setting low priority on destructor ensures it runs after all other global
// destructors. Priorities 0-100 are reserved by the compiler. The priority
// value 110 allows SYCL users to run their destructors after runtime library
// deinitialization.
__attribute__((destructor(110))) static void syclUnload() { shutdown_late(); }
#endif
} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL