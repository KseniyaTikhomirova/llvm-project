//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/ur/adapter_impl.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

void adapter_impl::queryLastErrorAndThrow(const sycl::errc &Error) {
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
      LastDeviceIds.resize(PlatformCount);
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
  auto It = std::find(UrPlatforms.begin(), UrPlatforms.end(), Platform);
  assert(It != UrPlatforms.end());
  return It - UrPlatforms.begin();
}

int adapter_impl::getStartingDeviceId(ur_platform_handle_t Platform) {
  int PlatformId = getPlatformId(Platform);
  return PlatformId == 0 ? 0 : LastDeviceIds[PlatformId - 1];
}

void adapter_impl::setLastDeviceId(ur_platform_handle_t Platform, int Id) {
  int PlatformId = getPlatformId(Platform);
  LastDeviceIds[PlatformId] = Id;
}

void adapter_impl::adjustLastDeviceId(ur_platform_handle_t Platform) {
  int PlatformId = getPlatformId(Platform);
  if (PlatformId > 0 &&
      LastDeviceIds[PlatformId] < LastDeviceIds[PlatformId - 1])
    LastDeviceIds[PlatformId] = LastDeviceIds[PlatformId - 1];
}

bool adapter_impl::containsUrPlatform(ur_platform_handle_t Platform) {
  auto It = std::find(UrPlatforms.begin(), UrPlatforms.end(), Platform);
  return It != UrPlatforms.end();
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
      loaderConfigCreateInfo.getFuncPtrFromModule(ur::getURLoaderLibrary());
  UrFuncInfo<UrApiKind::urLoaderConfigEnableLayer> loaderConfigEnableLayerInfo;
  auto loaderConfigEnableLayer =
      loaderConfigEnableLayerInfo.getFuncPtrFromModule(
          ur::getURLoaderLibrary());
  UrFuncInfo<UrApiKind::urLoaderConfigRelease> loaderConfigReleaseInfo;
  auto loaderConfigRelease =
      loaderConfigReleaseInfo.getFuncPtrFromModule(ur::getURLoaderLibrary());
  UrFuncInfo<UrApiKind::urLoaderConfigSetCodeLocationCallback>
      loaderConfigSetCodeLocationCallbackInfo;
  auto loaderConfigSetCodeLocationCallback =
      loaderConfigSetCodeLocationCallbackInfo.getFuncPtrFromModule(
          ur::getURLoaderLibrary());
  UrFuncInfo<UrApiKind::urLoaderInit> loaderInitInfo;
  auto loaderInit =
      loaderInitInfo.getFuncPtrFromModule(ur::getURLoaderLibrary());
  UrFuncInfo<UrApiKind::urAdapterGet> adapterGet_Info;
  auto adapterGet =
      adapterGet_Info.getFuncPtrFromModule(ur::getURLoaderLibrary());
  UrFuncInfo<UrApiKind::urAdapterGetInfo> adapterGetInfoInfo;
  auto adapterGetInfo =
      adapterGetInfoInfo.getFuncPtrFromModule(ur::getURLoaderLibrary());
  UrFuncInfo<UrApiKind::urAdapterSetLoggerCallback>
      adapterSetLoggerCallbackInfo;
  auto adapterSetLoggerCallback =
      adapterSetLoggerCallbackInfo.getFuncPtrFromModule(
          ur::getURLoaderLibrary());

  bool OwnLoaderConfig = false;
  // If we weren't provided with a custom config handle create our own.
  if (!LoaderConfig) {
    CHECK_UR_SUCCESS(loaderConfigCreate(&LoaderConfig))
    OwnLoaderConfig = true;
  }

  const char *LogOptions = "level:info;output:stdout;flush:info";
  if (trace(TraceLevel::TRACE_CALLS)) {
#ifdef _WIN32
    _putenv_s("UR_LOG_TRACING", LogOptions);
#else
    setenv("UR_LOG_TRACING", LogOptions, 1);
#endif
    CHECK_UR_SUCCESS(loaderConfigEnableLayer(LoaderConfig, "UR_LAYER_TRACING"));
  }

  if (trace(TraceLevel::TRACE_BASIC)) {
#ifdef _WIN32
    _putenv_s("UR_LOG_LOADER", LogOptions);
#else
    setenv("UR_LOG_LOADER", LogOptions, 1);
#endif
  }

  CHECK_UR_SUCCESS(loaderConfigSetCodeLocationCallback(
      LoaderConfig, codeLocationCallback, nullptr));

  switch (ProgramManager::getInstance().kernelUsesSanitizer()) {
  case SanitizerType::AddressSanitizer:
    CHECK_UR_SUCCESS(loaderConfigEnableLayer(LoaderConfig, "UR_LAYER_ASAN"));
    break;
  case SanitizerType::MemorySanitizer:
    CHECK_UR_SUCCESS(loaderConfigEnableLayer(LoaderConfig, "UR_LAYER_MSAN"));
    break;
  case SanitizerType::ThreadSanitizer:
    CHECK_UR_SUCCESS(loaderConfigEnableLayer(LoaderConfig, "UR_LAYER_TSAN"));
    break;
  default:
    break;
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
      return backend::ext_oneapi_level_zero;
    case UR_BACKEND_OPENCL:
      return backend::opencl;
    case UR_BACKEND_CUDA:
      return backend::ext_oneapi_cuda;
    case UR_BACKEND_HIP:
      return backend::ext_oneapi_hip;
    case UR_BACKEND_NATIVE_CPU:
      return backend::ext_oneapi_native_cpu;
    case UR_BACKEND_OFFLOAD:
      return backend::ext_oneapi_offload;
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

    const char *env_value = std::getenv("UR_LOG_CALLBACK");
    if (env_value == nullptr || std::string(env_value) != "disabled") {
      CHECK_UR_SUCCESS(adapterSetLoggerCallback(UrAdapter, urLoggerCallback,
                                                nullptr, UR_LOGGER_LEVEL_WARN));
    }
  }
#undef CHECK_UR_SUCCESS
}

// Get the adapter serving given backend.
template <backend BE> static adapter_impl *&adapter_impl::getAdapter() {
  static adapter_impl *adapterPtr = nullptr;
  if (adapterPtr)
    return adapterPtr;

  std::vector<adapter_impl *> Adapters = ur::initializeUr();
  for (auto &P : Adapters)
    if (P->hasBackend(BE)) {
      adapterPtr = P;
      return adapterPtr;
    }

  throw exception(errc::runtime, "ur::getAdapter couldn't find adapter");
}

template adapter_impl *&getAdapter<backend::opencl>();
template adapter_impl *&getAdapter<backend::level_zero>();
template adapter_impl *&getAdapter<backend::cuda>();
template adapter_impl *&getAdapter<backend::hip>();

private:
ur_adapter_handle_t MAdapter = nullptr;
backend MBackend;
// Mutex to guard UrPlatforms and LastDeviceIds.
std::shared_ptr<std::mutex> MAdapterMutex;
// vector of UrPlatforms that belong to this adapter
std::once_flag MPlatformsPopulated;
std::vector<ur_platform_handle_t> MUrPlatforms;
// represents the unique ids of the last device of each platform
// index of this vector corresponds to the index in UrPlatforms vector.
std::vector<int> MLastDeviceIds;

UrFuncPtrMapT MUrFuncPtrs;
}; // class adapter_impl

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
