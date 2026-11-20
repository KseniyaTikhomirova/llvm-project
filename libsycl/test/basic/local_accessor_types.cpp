//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -Wno-error=deprecated-declarations %s -fsyntax-only \
// RUN: -Xclang -verify -Xclang -verify-ignore-unexpected=note

#include <sycl/sycl.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

template <int Dims, typename AccT> struct FullSubscriptType;
template <typename AccT> struct FullSubscriptType<1, AccT> {
  using type = decltype(std::declval<AccT>()[0]);
};
template <typename AccT> struct FullSubscriptType<2, AccT> {
  using type = decltype(std::declval<AccT>()[0][0]);
};
template <typename AccT> struct FullSubscriptType<3, AccT> {
  using type = decltype(std::declval<AccT>()[0][0][0]);
};

template <typename DataT, int Dims, typename AccT> void checkTypes() {
  static_assert(std::is_same_v<typename AccT::value_type, DataT>);
  static_assert(std::is_same_v<typename AccT::reference, DataT &>);
  static_assert(std::is_same_v<typename AccT::const_reference, const DataT &>);
  static_assert(std::is_same_v<typename AccT::size_type, std::size_t>);
  static_assert(std::is_same_v<typename AccT::difference_type,
                               typename std::iterator_traits<
                                   typename AccT::iterator>::difference_type>);
  static_assert(std::is_same_v<decltype(std::declval<AccT>().begin()),
                               typename AccT::iterator>);
  static_assert(std::is_same_v<decltype(std::declval<AccT>().cbegin()),
                               typename AccT::const_iterator>);
  static_assert(std::is_same_v<decltype(std::declval<AccT>().rbegin()),
                               typename AccT::reverse_iterator>);
  static_assert(std::is_same_v<decltype(std::declval<AccT>().crbegin()),
                               typename AccT::const_reverse_iterator>);

  if constexpr (Dims == 0) {
    static_assert(std::is_convertible_v<const AccT &, DataT &>);
  } else {
    static_assert(
        std::is_same_v<typename FullSubscriptType<Dims, AccT>::type, DataT &>);
    static_assert(
        std::is_same_v<decltype(std::declval<AccT>()[sycl::id<Dims>{}]),
                       DataT &>);
    static_assert(std::is_same_v<decltype(std::declval<AccT>().get_range()),
                                 sycl::range<Dims>>);
  }
}

template <typename DataT> void checkAllDims() {
  checkTypes<DataT, 0, sycl::local_accessor<DataT, 0>>();
  checkTypes<DataT, 1, sycl::local_accessor<DataT, 1>>();
  checkTypes<DataT, 2, sycl::local_accessor<DataT, 2>>();
  checkTypes<DataT, 3, sycl::local_accessor<DataT, 3>>();
}

template <int Dims> void checkDeprecatedAccessor() {
  // expected-warning@+1 {{'local' is deprecated: use 'local_accessor' instead}}
  constexpr sycl::target Local = sycl::target::local;
  using AccT = sycl::accessor<int, Dims, sycl::access_mode::read_write, Local>;
  checkTypes<int, Dims, AccT>();
  static_assert(
      std::is_same_v<decltype(std::declval<AccT>().get_size()), std::size_t>);
  static_assert(
      std::is_same_v<decltype(std::declval<AccT>().get_count()), std::size_t>);
}

void checkAll() {
  checkAllDims<int>();
  checkAllDims<const int>();
  checkAllDims<double>();

  checkDeprecatedAccessor<1>();
  checkDeprecatedAccessor<2>();
  checkDeprecatedAccessor<3>();

  static_assert(std::is_convertible_v<sycl::local_accessor<int, 2>,
                                      sycl::local_accessor<const int, 2>>);
  static_assert(!std::is_convertible_v<sycl::local_accessor<const int, 1>,
                                       sycl::local_accessor<int, 1>>);
  static_assert(!std::is_convertible_v<sycl::local_accessor<int, 1>,
                                       sycl::local_accessor<int, 2>>);
  static_assert(std::is_default_constructible_v<sycl::local_accessor<int, 1>>);
  static_assert(
      !std::is_constructible_v<sycl::local_accessor<int, 1>, sycl::handler &>);
}

void checkMaxSizeDeprecated(const sycl::local_accessor<int, 1> &Acc) {
  // expected-warning@+1 {{'max_size' is deprecated: max_size() is deprecated by SYCL 2020}}
  (void)Acc.max_size();
}
