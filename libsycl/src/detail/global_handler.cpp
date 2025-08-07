//==--------- global_handler.cpp --- Global objects handler ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/adapter_impl.hpp>
#include <detail/global_handler.hpp>
#include <detail/platform_impl.hpp>

#ifdef _WIN32
#  include <windows.h>
#endif

#include <vector>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

using LockGuard = std::lock_guard<SpinLock>;
SpinLock GlobalHandler::MSyclGlobalHandlerProtector{};

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

std::vector<std::shared_ptr<platform_impl>> &GlobalHandler::getPlatforms() {
  static std::vector<platform_impl *> &Platforms = getOrCreate(MPlatforms);
  return Platforms;
}

std::vector<adapter_impl *> &GlobalHandler::getAdapters() {
  static std::vector<adapter_impl *> &Adapters = getOrCreate(MAdapters);
  return Adapters;
}

void GlobalHandler::unloadAdapters() {
  // Call to GlobalHandler::instance().getAdapters() initializes adapters. If
  // user application has loaded SYCL runtime, and never called any APIs,
  // there's no need to load and unload adapters.
  //

  if (Platforms.Inst) {
    for (const auto &Platform : getPlatforms()) {
      delete Platform;
    }
    Platforms.Inst.clear();
  }
  if (MAdapters.Inst) {
    for (const auto &Adapter : MAdapters.Inst) {
      Adapter->release();
      delete Adapter;
    }
  }

  UrFuncInfo<UrApiKind::urLoaderTearDown> loaderTearDownInfo;
  auto loaderTearDown = loaderTearDownInfo.getFuncPtrFromModule(nullptr);
  loaderTearDown();

  // Clear after unload to avoid uses after unload.
  MAdapters.Inst.clear();

  MPlatforms.Inst.reset(nullptr);
  MAdapters.Inst.reset(nullptr);
}

void shutdown_late() {
  Handler->unloadAdapters();

  // Release the rest of global resources.
  delete Handler;
  Handler = nullptr;
}

extern "C" __SYCL_EXPORT BOOL WINAPI DllMain(HINSTANCE hinstDLL,
                                             DWORD fdwReason,
                                             LPVOID lpReserved) {
  // Perform actions based on the reason for calling.
  switch (fdwReason) {
  case DLL_PROCESS_DETACH:
    if (PrintUrTrace)
      std::cout << "---> DLL_PROCESS_DETACH syclx.dll\n" << std::endl;

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