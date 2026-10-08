//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the shared implementation of the SYCL local accessors:
/// sycl::local_accessor and the deprecated specialization of sycl::accessor
/// for access::target::local.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_DETAIL_LOCAL_ACCESSOR_BASE_HPP
#define _LIBSYCL___IMPL_DETAIL_LOCAL_ACCESSOR_BASE_HPP

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/linearization.hpp>
#include <sycl/__impl/exception.hpp>
#include <sycl/__impl/index_space_classes.hpp>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

/// \return a process-wide unique non-zero identifier for a newly constructed
/// local accessor.
_LIBSYCL_EXPORT std::uint64_t createLocalAccessorId();

/// Helper for operator[] of a multi-dimensional accessor. Each subscript
/// consumes one dimension, the last one yields the element reference.
template <typename DataT, int Dimensions, int CurDim> class AccessorSubscript {
  static_assert(Dimensions > CurDim,
                "Current dimension must be less than total dimensions");
  static constexpr bool IsLastDim = CurDim + 1 == Dimensions;

public:
  AccessorSubscript(DataT *Ptr, const range<Dimensions> &Range,
                    std::size_t LinearIndex)
      : MPtr(Ptr), MRange(Range), MLinearIndex(LinearIndex) {}

  decltype(auto) operator[](std::size_t Index) const {
    const std::size_t LinearIndex = MLinearIndex * MRange[CurDim] + Index;
    if constexpr (IsLastDim)
      return MPtr[LinearIndex];
    else
      return AccessorSubscript<DataT, Dimensions, CurDim + 1>(MPtr, MRange,
                                                              LinearIndex);
  }

private:
  DataT *MPtr;
  range<Dimensions> MRange;
  std::size_t MLinearIndex;
};

/// Provides access to work-group local memory allocated for a kernel.
///
/// The class has a single definition and a single layout for host and device.
/// On host MData is always nullptr, element access is not available and the
/// object only records the requested allocation size and the identity of the
/// accessor. On device the object is updated with device memory pointer.
///
/// The base is parameterized on DataT and Dimensions only, so that both
/// sycl::local_accessor and the deprecated sycl::accessor specialization for
/// access::target::local share a single instantiation.
template <typename DataT, int Dimensions> class LocalAccessorBase {
  static_assert(Dimensions >= 0 && Dimensions <= 3,
                "A local accessor can only be 0-, 1-, 2-, or 3-dimensional.");

public:
  using value_type = DataT;
  using reference = value_type &;
  using const_reference = const DataT &;
  // TODO: add accessor_ptr, get_pointer and get_multi_ptr once multi_ptr is
  // supported.

  // TODO: SYCL 2020 Table 51 requires the pointer underlying the iterator of a
  // local accessor to be address space qualified. A plain pointer is a
  // conforming choice for the unspecified iterator type only because the
  // qualifier is dropped on conversion to the generic address space.
  using iterator = value_type *;
  using const_iterator = const value_type *;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using difference_type =
      typename std::iterator_traits<iterator>::difference_type;
  using size_type = std::size_t;

  size_type byte_size() const noexcept { return size() * sizeof(DataT); }

  size_type size() const noexcept { return MRange.size(); }

  __SYCL2020_DEPRECATED("max_size() is deprecated by SYCL 2020")
  size_type max_size() const noexcept {
    return empty() ? 0 : (std::numeric_limits<difference_type>::max)();
  }

  bool empty() const noexcept { return size() == 0; }

  template <int Dims = Dimensions, std::enable_if_t<(Dims > 0), bool> = true>
  range<Dimensions> get_range() const {
    return MRange;
  }

  // SYCL 2020 defines no properties for a local accessor, so there is nothing
  // to store and nothing to report.
  template <typename Property> bool has_property() const noexcept {
    return false;
  }

  template <typename Property> Property get_property() const {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::invalid),
        "a local accessor doesn't have the requested property");
  }

  // For accessor and local_accessor, the following API  functions may only be
  // called from within a command.

  template <int Dims = Dimensions, std::enable_if_t<Dims == 0, bool> = true>
  operator reference() const {
    return *getPtr();
  }

  template <int Dims = Dimensions, std::enable_if_t<(Dims > 0), bool> = true>
  reference operator[](id<Dimensions> Index) const {
    return getPtr()[linearizeId(Index, MRange)];
  }

  template <int Dims = Dimensions, std::enable_if_t<Dims == 1, bool> = true>
  reference operator[](std::size_t Index) const {
    return getPtr()[Index];
  }

  template <int Dims = Dimensions, std::enable_if_t<(Dims > 1), bool> = true>
  auto operator[](std::size_t Index) const {
    return AccessorSubscript<DataT, Dimensions, 1>(getPtr(), MRange, Index);
  }

  iterator begin() const noexcept { return getPtr(); }
  iterator end() const noexcept { return begin() + size(); }

  const_iterator cbegin() const noexcept { return const_iterator(begin()); }
  const_iterator cend() const noexcept { return const_iterator(end()); }

  reverse_iterator rbegin() const noexcept { return reverse_iterator(end()); }
  reverse_iterator rend() const noexcept { return reverse_iterator(begin()); }

  const_reverse_iterator crbegin() const noexcept {
    return const_reverse_iterator(end());
  }
  const_reverse_iterator crend() const noexcept {
    return const_reverse_iterator(begin());
  }

protected:
  static constexpr int AdjustedDims = Dimensions == 0 ? 1 : Dimensions;

  using LocalPtrT = DataT _LIBSYCL_LOCAL_AS *;

  LocalAccessorBase() = default;

  explicit LocalAccessorBase(range<AdjustedDims> AllocationSize)
      : MId(createLocalAccessorId()), MRange(AllocationSize) {}

  template <typename OtherDataT>
  LocalAccessorBase(
      const LocalAccessorBase<OtherDataT, Dimensions> &Other) noexcept
      : MOffset(Other.MOffset), MData(Other.MData), MId(Other.MId),
        MRange(Other.MRange) {}

  // void initialize(LocalPtrT BlockBase, range<AdjustedDims> Range,
  //                 std::size_t Offset) {
  //   MOffset = Offset;
  //   MData = BlockBase;
  //   MRange = Range;
  // }

  template <typename OtherDataT>
  bool
  isEqual(const LocalAccessorBase<OtherDataT, Dimensions> &Rhs) const noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    // On device an accessor is identified by the work-group local memory
    // region it refers to. All local accessors of a kernel share MData, so the
    // region is the pair of the block base and the offset within the block.
    return MData == Rhs.MData && MOffset == Rhs.MOffset;
#else
    // MData is not available on host, so MId identifies the accessor.
    return MId == Rhs.MId;
#endif
  }

  void swapBase(LocalAccessorBase &Other) {
    using std::swap;
    swap(MOffset, Other.MOffset);
    swap(MData, Other.MData);
    swap(MId, Other.MId);
    swap(MRange, Other.MRange);
  }

  std::uint64_t getId() const noexcept { return MId; }

  /// \return a generic pointer to the first element of this accessor's region.
  DataT *getPtr() const noexcept { return getLocalPtr(); }

private:
  LocalPtrT getLocalPtr() const noexcept {
    using ByteT = std::conditional_t<std::is_const_v<DataT>, const char, char>;
    using LocalByteT = ByteT _LIBSYCL_LOCAL_AS *;
    return reinterpret_cast<LocalPtrT>(reinterpret_cast<LocalByteT>(MData) +
                                       MOffset);
  }

  // All local accessors used in the same kernel share a single memory
  // allocation. MOffset identifies the region within that allocation that
  // belongs to this accessor. Together with MData it identifies the accessor
  // on device, see isEqual().
  std::size_t MOffset = 0;
  LocalPtrT MData = nullptr;

  // Identity of the accessor on host. A memory pointer is not available on
  // host but we need something to identify the accessor uniquely and perform a
  // comparison check. Also the value std::hash is computed from.
  std::uint64_t MId = 0;

  range<AdjustedDims> MRange;

  template <typename, int> friend class LocalAccessorBase;
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_DETAIL_LOCAL_ACCESSOR_BASE_HPP
