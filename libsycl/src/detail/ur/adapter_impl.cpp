//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/global_handler.hpp>
#include <detail/ur/adapter_impl.hpp>

#include <algorithm> // std::find
#include <iostream>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

#define __SYCL_BACKEND_ERROR_REPORT(backend)                                   \
  std::string(sycl::detail::get_backend_name(backend)) +                       \
      " backend failed with error: "

void adapter_impl::queryLastErrorAndThrow(const sycl::errc &Error) const {
  const char *message = nullptr;
  int32_t adapter_error = 0;
  ur_result_t Result = call_nocheck<UrApiKind::urAdapterGetLastError>(
      MAdapter, &message, &adapter_error);
  throw sycl::exception(sycl::make_error_code(Error),
                        __SYCL_BACKEND_ERROR_REPORT(MBackend) +
                            sycl::detail::codeToString(Result) +
                            (message ? "\n" + std::string(message) +
                                           "(adapter error )" +
                                           std::to_string(adapter_error) + "\n"
                                     : std::string{}));
}

std::vector<ur_platform_handle_t> &adapter_impl::getUrPlatforms() {
  std::call_once(MPlatformsPopulated, [&]() {
    uint32_t PlatformCount = 0;
    call<UrApiKind::urPlatformGet>(MAdapter, 0u, nullptr, &PlatformCount);
    MUrPlatforms.resize(PlatformCount);
    if (PlatformCount) {
      call<UrApiKind::urPlatformGet>(MAdapter, PlatformCount,
                                     MUrPlatforms.data(), nullptr);
      MLastDeviceIds.resize(PlatformCount);
    }
  });
  return MUrPlatforms;
}

void adapter_impl::release() {
  auto Res = call_nocheck<UrApiKind::urAdapterRelease>(MAdapter);
  if (Res == UR_RESULT_ERROR_ADAPTER_SPECIFIC) {
    // We can't query the adapter for the error message because the adapter
    // has been released
    throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                          __SYCL_BACKEND_ERROR_REPORT(MBackend) +
                              "Adapter failed to be released and reported "
                              "`UR_RESULT_ERROR_ADAPTER_SPECIFIC`. This should "
                              "never happen, please file a bug.");
  }
  // method for excpetion
  MAdapter = nullptr;
  checkUrResult(Res);
}

int adapter_impl::getPlatformId(ur_platform_handle_t Platform) {
  auto It = std::find(MUrPlatforms.begin(), MUrPlatforms.end(), Platform);
  assert(It != MUrPlatforms.end());
  return It - MUrPlatforms.begin();
}

int adapter_impl::getStartingDeviceId(ur_platform_handle_t Platform) {
  int PlatformId = getPlatformId(Platform);
  return PlatformId == 0 ? 0 : MLastDeviceIds[PlatformId - 1];
}

void adapter_impl::setLastDeviceId(ur_platform_handle_t Platform, int Id) {
  int PlatformId = getPlatformId(Platform);
  MLastDeviceIds[PlatformId] = Id;
}

void adapter_impl::adjustLastDeviceId(ur_platform_handle_t Platform) {
  int PlatformId = getPlatformId(Platform);
  if (PlatformId > 0 &&
      MLastDeviceIds[PlatformId] < MLastDeviceIds[PlatformId - 1])
    MLastDeviceIds[PlatformId] = MLastDeviceIds[PlatformId - 1];
}

bool adapter_impl::containsUrPlatform(ur_platform_handle_t Platform) {
  auto It = std::find(MUrPlatforms.begin(), MUrPlatforms.end(), Platform);
  return It != MUrPlatforms.end();
}

std::vector<adapter_impl *> &
adapter_impl::getAdapters(ur_loader_config_handle_t LoaderConfig) {
  // This uses static variable initialization to work around a gcc bug with
  // std::call_once and exceptions.
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=66146
  static bool Initialized = [=]() {
    initializeAdapters(GlobalHandler::instance().getAdapters(), LoaderConfig);
    return true;
  }();
  std::ignore = Initialized;

  return GlobalHandler::instance().getAdapters();
}

void adapter_impl::initializeAdapters(std::vector<adapter_impl *> &Adapters,
                                      ur_loader_config_handle_t LoaderConfig) {
#define CHECK_UR_SUCCESS(Call)                                                 \
  {                                                                            \
    if (ur_result_t error = Call) {                                            \
      std::cerr << "UR adapter initialization failed: "                        \
                << sycl::detail::codeToString(error) << std::endl;             \
    }                                                                          \
  }

  UrFuncInfo<UrApiKind::urLoaderConfigCreate> loaderConfigCreateInfo;
  auto loaderConfigCreate =
      loaderConfigCreateInfo.getFuncPtrFromModule(nullptr);
  // UrFuncInfo<UrApiKind::urLoaderConfigEnableLayer>
  // loaderConfigEnableLayerInfo;
  //  auto loaderConfigEnableLayer =
  //      loaderConfigEnableLayerInfo.getFuncPtrFromModule(nullptr);
  UrFuncInfo<UrApiKind::urLoaderConfigRelease> loaderConfigReleaseInfo;
  auto loaderConfigRelease =
      loaderConfigReleaseInfo.getFuncPtrFromModule(nullptr);
  // UrFuncInfo<UrApiKind::urLoaderConfigSetCodeLocationCallback>
  //     loaderConfigSetCodeLocationCallbackInfo;
  // auto loaderConfigSetCodeLocationCallback =
  //     loaderConfigSetCodeLocationCallbackInfo.getFuncPtrFromModule(nullptr);
  UrFuncInfo<UrApiKind::urLoaderInit> loaderInitInfo;
  auto loaderInit = loaderInitInfo.getFuncPtrFromModule(nullptr);
  UrFuncInfo<UrApiKind::urAdapterGet> adapterGet_Info;
  auto adapterGet = adapterGet_Info.getFuncPtrFromModule(nullptr);
  UrFuncInfo<UrApiKind::urAdapterGetInfo> adapterGetInfoInfo;
  auto adapterGetInfo = adapterGetInfoInfo.getFuncPtrFromModule(nullptr);
  // UrFuncInfo<UrApiKind::urAdapterSetLoggerCallback>
  //     adapterSetLoggerCallbackInfo;
  // auto adapterSetLoggerCallback =
  //     adapterSetLoggerCallbackInfo.getFuncPtrFromModule(nullptr);

  bool OwnLoaderConfig = false;
  // If we weren't provided with a custom config handle create our own.
  if (!LoaderConfig) {
    CHECK_UR_SUCCESS(loaderConfigCreate(&LoaderConfig))
    OwnLoaderConfig = true;
  }

  ur_device_init_flags_t device_flags = 0;
  CHECK_UR_SUCCESS(loaderInit(device_flags, LoaderConfig));

  if (OwnLoaderConfig) {
    CHECK_UR_SUCCESS(loaderConfigRelease(LoaderConfig));
  }

  uint32_t adapterCount = 0;
  CHECK_UR_SUCCESS(adapterGet(0, nullptr, &adapterCount));
  std::vector<ur_adapter_handle_t> adapters(adapterCount);
  CHECK_UR_SUCCESS(adapterGet(adapterCount, adapters.data(), nullptr));

  auto UrToSyclBackend = [](ur_backend_t backend) -> sycl::backend {
    switch (backend) {
    case UR_BACKEND_LEVEL_ZERO:
      return backend::level_zero;
    case UR_BACKEND_OPENCL:
      return backend::opencl;
    case UR_BACKEND_CUDA:
      return backend::cuda;
    case UR_BACKEND_HIP:
      return backend::hip;
    default:
      // Throw an exception, this should be unreachable.
      CHECK_UR_SUCCESS(UR_RESULT_ERROR_INVALID_ENUMERATION)
      return backend::all;
    }
  };

  for (const auto &UrAdapter : adapters) {
    ur_backend_t adapterBackend = UR_BACKEND_UNKNOWN;
    CHECK_UR_SUCCESS(adapterGetInfo(UrAdapter, UR_ADAPTER_INFO_BACKEND,
                                    sizeof(adapterBackend), &adapterBackend,
                                    nullptr));
    auto syclBackend = UrToSyclBackend(adapterBackend);
    Adapters.emplace_back(new adapter_impl(UrAdapter, syclBackend));
  }
#undef CHECK_UR_SUCCESS
}

// Get the adapter serving given backend.
template <backend BE> adapter_impl *&adapter_impl::getAdapter() {
  static adapter_impl *adapterPtr = nullptr;
  if (adapterPtr)
    return adapterPtr;

  std::vector<adapter_impl *> Adapters = getAdapters();
  for (auto &P : Adapters)
    if (P->hasBackend(BE)) {
      adapterPtr = P;
      return adapterPtr;
    }

  throw exception(errc::runtime, "ur::getAdapter couldn't find adapter");
}

template <> adapter_impl *&adapter_impl::getAdapter<backend::opencl>();
template <> adapter_impl *&adapter_impl::getAdapter<backend::level_zero>();
template <> adapter_impl *&adapter_impl::getAdapter<backend::cuda>();
template <> adapter_impl *&adapter_impl::getAdapter<backend::hip>();

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
