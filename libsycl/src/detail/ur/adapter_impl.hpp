//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/detail/config.hpp> // namespace macro

#include <ur_api.h>

#include <mutex> // std::call_once

#ifndef _LIBSYCL_ADAPTER_IMPL
#  define _LIBSYCL_ADAPTER_IMPL

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

enum class backend : char;

/// The adapter class provides a unified interface to the underlying low-level
/// runtimes for the device-agnostic SYCL runtime.
class adapter_impl {
public:
  adapter_impl() = delete;

  adapter_impl(ur_adapter_handle_t Adapter, backend Backend)
      : MAdapter(Adapter), MBackend(Backend),
        MAdapterMutex(std::make_shared<std::mutex>()) {}

  // Disallow accidental copies of adapters
  adapter_impl &operator=(const adapter_impl &) = delete;
  adapter_impl(const adapter_impl &) = delete;
  adapter_impl &operator=(adapter_impl &&other) noexcept = delete;
  adapter_impl(adapter_impl &&other) noexcept = delete;

  ~adapter_impl() = default;

  void queryLastErrorAndThrow(const sycl::errc &Error);

  template <sycl::errc errc = sycl::errc::runtime>
  void checkUrResult(ur_result_t InitialResult) const {
    if (InitialResult == UR_RESULT_ERROR_ADAPTER_SPECIFIC) {
      assert(MAdapter);
      queryLastErrorAndThrow();
    }
    if (InitialResult != UR_RESULT_SUCCESS) {
      throw sycl::exception(sycl::make_error_code(errc),
                            __SYCL_BACKEND_ERROR_REPORT(MBackend) +
                                sycl::detail::codeToString(InitialResult));
    }
  }

  std::vector<ur_platform_handle_t> &getUrPlatforms();

  ur_adapter_handle_t getUrAdapter() { return MAdapter; }

  /// Calls the UR Api, traces the call, and returns the result.
  ///
  /// Usage:
  /// \code{cpp}
  /// ur_result_t Err = Adapter->call<UrApiKind::urEntryPoint>(Args);
  /// Adapter->checkUrResult(Err);
  /// // Checks Result and throws a runtime_error exception.
  /// \endcode
  ///
  /// \sa adapter::checkUrResult
  template <UrApiKind UrApiOffset, typename... ArgsT>
  ur_result_t call_nocheck(ArgsT &&...Args) const {
    ur_result_t R = UR_RESULT_SUCCESS;
    if (MAdapter) {
      detail::UrFuncInfo<UrApiOffset> UrApiInfo;
      auto F = UrApiInfo.getFuncPtr(&UrFuncPtrs);
      R = F(std::forward<ArgsT>(Args)...);
    }
    return R;
  }

  /// Calls the API, traces the call, checks the result
  ///
  /// \throw sycl::runtime_exception if the call was not successful.
  template <UrApiKind UrApiOffset, typename... ArgsT>
  void call(ArgsT &&...Args) const {
    auto Err = call_nocheck<UrApiOffset>(std::forward<ArgsT>(Args)...);
    checkUrResult(Err);
  }

  /// Calls the API, traces the call, checks the result
  ///
  /// \throw sycl::exceptions(errc) if the call was not successful.
  template <sycl::errc errc, UrApiKind UrApiOffset, typename... ArgsT>
  void call(ArgsT &&...Args) const {
    auto Err = call_nocheck<UrApiOffset>(std::forward<ArgsT>(Args)...);
    checkUrResult<errc>(Err);
  }

  /// Returns the backend reported by the adapter.
  backend getBackend() const { return MBackend; }

  /// Tells if this adapter can serve specified backend.
  /// For example, Unified Runtime adapter will be able to serve
  /// multiple backends as determined by the platforms reported by the adapter.
  bool hasBackend(backend Backend) const { return Backend == MBackend; }

  void release();

  // Return the index of a UR platform.
  // Platform must belong to the current adapter.
  // The function is expected to be called in a thread safe manner.
  int getPlatformId(ur_platform_handle_t Platform);

  // Device ids are consecutive across platforms within a adapter.
  // We need to return the same starting index for the given platform.
  // So, instead of returing the last device id of the given platform,
  // return the last device id of the predecessor platform.
  // The function is expected to be called in a thread safe manner.
  int getStartingDeviceId(ur_platform_handle_t Platform);

  // set the id of the last device for the given platform
  // The function is expected to be called in a thread safe manner.
  void setLastDeviceId(ur_platform_handle_t Platform, int Id);

  // Adjust the id of the last device for the given platform.
  // Involved when there is no device on that platform at all.
  // The function is expected to be called in a thread safe manner.
  void adjustLastDeviceId(ur_platform_handle_t Platform);

  bool containsUrPlatform(ur_platform_handle_t Platform);

  std::shared_ptr<std::mutex> getAdapterMutex() { return MAdapterMutex; }

  static std::vector<adapter_impl *> &
  getAdapters(ur_loader_config_handle_t LoaderConfig = nullptr);

  static void initializeAdapters(std::vector<adapter_impl *> &Adapters,
                                 ur_loader_config_handle_t LoaderConfig);

  // Get the adapter serving given backend.
  template <backend BE> static adapter_impl *&getAdapter();

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

#endif // _LIBSYCL_ADAPTER_IMPL