//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of SYCL 2020 platform info types.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_INFO_PLATFORM_HPP
#define _LIBSYCL___IMPL_INFO_PLATFORM_HPP

#include <cstdint>
#include <string> // info::platform::extensions
#include <vector> // info::platform::extensions

#include <sycl/__impl/detail/config.hpp>            // namespace macro
#include <sycl/__impl/detail/macro_definitions.hpp> // __SYCL2020_DEPRECATED
#include <ur_api.h> // all info::platform entries are mapped to UR code

_LIBSYCL_BEGIN_NAMESPACE_SYCL

// A.1. Platform information descriptors

namespace info {
// ktikhomi: to be moved to a common place
#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, UrCode)              \
  struct Desc {                                                                \
    using return_type = ReturnT;                                               \
  };

// 4.6.2.4. Information descriptors
namespace platform {

#define __SYCL_PARAM_TRAITS_DEPRECATED(Desc, Message)                          \
  struct __SYCL2020_DEPRECATED(Message) Desc;

#include <sycl/__impl/info/platform.def>
#include <sycl/__impl/info/platform_deprecated_2020.def>

#undef __SYCL_PARAM_TRAITS_DEPRECATED
} // namespace platform

#undef __SYCL_PARAM_TRAITS_SPEC
} // namespace info

namespace detail {
template <typename T> struct is_platform_info_desc : std::false_type {};

// ktikhomi: to be moved to a common place
#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, UrCode)              \
  template <>                                                                  \
  struct is_##DescType##_info_desc<info::DescType::Desc> : std::true_type {    \
    using return_type = info::DescType::Desc::return_type;                     \
  };

#include <sycl/__impl/info/platform.def>

#undef __SYCL_PARAM_TRAITS_SPEC
} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_INFO_PLATFORM_HPP