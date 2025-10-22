//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains helper class SpinLock.
/// SpinLock is a synchronization primitive, that uses atomic variable and
/// causes thread trying acquire lock wait in loop while repeatedly check if
/// the lock is available.
///
/// One important feature of this implementation is that std::atomic<bool> can
/// be zero-initialized. This allows SpinLock to have trivial constructor and
/// destructor, which makes it possible to use it in global context (unlike
/// std::mutex, that doesn't provide such guarantees).
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_SPINLOCK_HPP
#define _LIBSYCL___IMPL_SPINLOCK_HPP

#include <sycl/__impl/detail/config.hpp> // namespace macro

#include <atomic>
#include <thread>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {
class SpinLock {
public:
  bool try_lock() { return !MLock.test_and_set(std::memory_order_acquire); }

  void lock() {
    while (MLock.test_and_set(std::memory_order_acquire))
      std::this_thread::yield();
  }
  void unlock() { MLock.clear(std::memory_order_release); }

private:
  std::atomic_flag MLock = ATOMIC_FLAG_INIT;
};
} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_SPINLOCK_HPP
