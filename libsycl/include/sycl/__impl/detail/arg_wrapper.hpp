//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains helper functions used to wrap kernel argumentss to
/// typeless collection.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_DETAIL_ARG_WRAPPER_HPP
#define _LIBSYCL___IMPL_DETAIL_ARG_WRAPPER_HPP

#include <sycl/__impl/detail/config.hpp>

#include <algorithm>
#include <cassert>
#include <iterator>
#include <memory>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

class ArgWrapperBase {
public:
  ArgWrapperBase() = default;
  ArgWrapperBase(const ArgWrapperBase &) = delete;
  ArgWrapperBase &operator=(const ArgWrapperBase &) = delete;
  virtual ~ArgWrapperBase() = default;

  virtual void deepCopy() = 0;
  virtual size_t getSize() const = 0;
  virtual void *getPtr() const = 0;
};

template <typename Type> class ArgWrapper : public ArgWrapperBase {
  static constexpr bool isUSM = std::is_pointer_v<Type>;
  using RawType = std::remove_pointer_t<Type>;

public:
  ArgWrapper(Type &Arg) {
    if constexpr (isUSM)
      Ptr = Arg;
    else
      Ptr = &Arg;
    // is device copyable
  }
  ArgWrapper(const ArgWrapper &) = delete;
  ArgWrapper &operator=(const ArgWrapper &) = delete;

  size_t getSize() const override { return sizeof(RawType); }

  void *getPtr() const override {
    assert((!DeepCopy || (DeepCopy.get()) == Ptr) &&
           "Incorrect state of copied argument");
    return Ptr;
  }

  void deepCopy() override {
    if constexpr (isUSM)
      return;
    if (DeepCopy)
      return;

    DeepCopy.reset(new RawType(*Ptr));
    Ptr = &DeepCopy.get();
  }

private:
  RawType *Ptr;
  std::unique_ptr<RawType> DeepCopy;
};

class ArgCollection {
public:
  ArgCollection(size_t Size) { MArgs.reserve(Size); }

  template <typename Type> void addArg(Type &Arg, int Index = -1) {
    assert(MArgs.capacity() > MArgs.size() && "Unexpected arguments count.");
    // is device copyable

    size_t IndexToUse = Index > 0 ? Index : MArgs.size();
    if (IndexToUse > MArgs.capacity())
      throw sycl::exception(make_error_code(errc::invalid),
                            "Index is out of range.");
    // This is more like warning than assert since it is an allowed usage.
    // But this usage must be intentional.
    assert(IndexToUse <= MArgs.size() && "Insertion is not sequential. ");

    MArgs.insert(MArgs.begin() + IndexToUse, Arg);
  }

  void **getArgumentsArray() {
    if (MPtrs.size() != MArgs.size()) {
      MPtrs.clear();
      std::transform(MArgs.cbegin(), MArgs.cend(), MPtrs.begin(),
                     [](const ArgWrapperBase &Arg) { return Arg.getPtr(); });
    }
    return MPtrs.data();
  }

  int64_t *getSizesArray() {
    if (MSizes.size() != MArgs.size()) {
      MSizes.clear();
      std::transform(MArgs.cbegin(), MArgs.cend(), MSizes.begin(),
                     [](const ArgWrapperBase &Arg) {
                       return static_cast<int64_t>(Arg.getSize());
                     });
    }
    return MSizes.data();
  }

  size_t getArgumentCount() { return MArgs.size(); }

  void deepCopy() {
    for (auto &Arg : MArgs)
      Arg.deepCopy();
  }

private:
  std::vector<ArgWrapperBase> MArgs;
  std::vector<int64_t> MSizes;
  std::vector<void *> MPtrs;
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_DETAIL_ARG_WRAPPER_HPP
