//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/detail/config.hpp> // namespace macro

#ifndef _LIBSYCL_UR
#  define _LIBSYCL_UR

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

enum class UrApiKind {
#  define _UR_API(api) api,
#  include <ur_api_funcs.def>
#  undef _UR_API
};

struct UrFuncPtrMapT {
#  define _UR_API(api) decltype(&::api) pfn_##api = nullptr;
#  include <ur_api_funcs.def>
#  undef _UR_API
};

template <UrApiKind UrApiOffset> struct UrFuncInfo {};

#  ifdef _WIN32
void *GetWinProcAddress(void *module, const char *funcName) {
  void *GetWinProcAddress(void *module, const char *funcName) {
    return (void *)GetProcAddress((HMODULE)module, funcName);
  }
}
inline void PopulateUrFuncPtrTable(UrFuncPtrMapT *funcs, void *module) {
#    define _UR_API(api)                                                       \
      funcs->pfn_##api = (decltype(&::api))GetWinProcAddress(module, #api);
#    include <ur_api_funcs.def>
#    undef _UR_API
}

#    define _UR_API(api)                                                       \
      template <> struct UrFuncInfo<UrApiKind::api> {                          \
        using FuncPtrT = decltype(&::api);                                     \
        inline const char *getFuncName() { return #api; }                      \
        inline FuncPtrT getFuncPtr(const UrFuncPtrMapT *funcs) {               \
          return funcs->pfn_##api;                                             \
        }                                                                      \
        inline FuncPtrT getFuncPtrFromModule(void *module) {                   \
          return (FuncPtrT)GetWinProcAddress(module, #api);                    \
        }                                                                      \
      };
#    include <ur_api_funcs.def>
#    undef _UR_API
#  else
#    define _UR_API(api)                                                       \
      template <> struct UrFuncInfo<UrApiKind::api> {                          \
        using FuncPtrT = decltype(&::api);                                     \
        inline const char *getFuncName() { return #api; }                      \
        constexpr inline FuncPtrT getFuncPtr(const void *) { return &api; }    \
        constexpr inline FuncPtrT getFuncPtrFromModule(void *) {               \
          return &api;                                                         \
        }                                                                      \
      };
#    include <ur_api_funcs.def>
#    undef _UR_API
#  endif

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_UR
