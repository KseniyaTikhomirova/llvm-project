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
#include <sycl/__impl/property_list.hpp>

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

template <typename DataT, int Dimensions> class LocalAccessorBase;

/// Lays out the local accessors of one kernel invocation in the single block
/// of work-group local memory allocated for that invocation.
///
/// The host only computes the size of the block. The device points each
/// accessor to its region at the start of the kernel. Both walk the accessors
/// in the order the compiler passes them and use appendRegion(), so they agree
/// on the layout.
///
/// The functions take LocalAccessorBase so that template argument deduction
/// accepts both local accessor classes deriving from it.
class LocalAccessorReifier {
public:
  /// \return the size in bytes of the block that holds \p Accs, to request at
  /// launch.
  template <typename... AccTs>
  static std::size_t getBlockSize(const AccTs &...Accs) {
    std::size_t BlockSize = 0;
    (appendRegion(Accs, BlockSize), ...);
    return BlockSize;
  }

  /// Points each of \p Accs to its region of the block, at the start of the
  /// kernel.
  template <typename... AccTs>
  static void bind([[maybe_unused]] AccTs &...Accs) {
#ifdef __SYCL_DEVICE_ONLY__
    // The work-group local memory allocated for the kernel invocation, its size
    // is ol_kernel_launch_size_args_t::DynSharedMemory.
    LocalBytePtrT Block = __builtin_sycl_dynamic_local_memory();
    std::size_t BlockSize = 0;
    (bindOne(Accs, Block, BlockSize), ...);
#endif
  }

private:
  /// \return the offset of the region of \p Acc within the block.
  ///
  /// \param BlockSize the size in bytes of the regions appended so far, it is
  /// increased by the region of \p Acc.
  template <typename DataT, int Dimensions>
  static std::size_t
  appendRegion(const LocalAccessorBase<DataT, Dimensions> &Acc,
               std::size_t &BlockSize) {
    // TODO: the device guarantees a block alignment of only 32 bytes, see
    // SPIRVLowerDynamicLocalMemory. A DataT aligned stricter than that needs
    // an over-allocated block.
    constexpr std::size_t Align = alignof(DataT);
    const std::size_t Offset = (BlockSize + Align - 1) / Align * Align;
    BlockSize = Offset + Acc.byte_size();
    return Offset;
  }

#ifdef __SYCL_DEVICE_ONLY__
  using LocalBytePtrT = unsigned char _LIBSYCL_LOCAL_AS *;

  template <typename DataT, int Dimensions>
  static void bindOne(LocalAccessorBase<DataT, Dimensions> &Acc,
                      LocalBytePtrT Block, std::size_t &BlockSize) {
    using LocalPtrT = typename LocalAccessorBase<DataT, Dimensions>::LocalPtrT;
    Acc.MData =
        reinterpret_cast<LocalPtrT>(Block + appendRegion(Acc, BlockSize));
  }
#endif
};

/// Provides access to work-group local memory allocated for a kernel.
///
/// The class has a single definition and a single layout for host and device.
/// On host MData is always nullptr, element access is not available and the
/// object only records the requested allocation size and the identity of the
/// accessor. At the start of the kernel the device points the accessor to its
/// region of the kernel's local memory block, see LocalAccessorReifier.
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

  // SYCL 2020 defines no properties for a local accessor and the constructors
  // reject any, so there is nothing to store and nothing to report.
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

  LocalAccessorBase(range<AdjustedDims> AllocationSize,
                    const property_list &PropList)
      : MId(createLocalAccessorId()), MRange(AllocationSize) {
    checkPropertyList<LocalAccessorBase>(PropList);
  }

  template <typename OtherDataT>
  LocalAccessorBase(
      const LocalAccessorBase<OtherDataT, Dimensions> &Other) noexcept
      : MData(Other.MData), MId(Other.MId), MRange(Other.MRange) {}

  template <typename OtherDataT>
  bool
  isEqual(const LocalAccessorBase<OtherDataT, Dimensions> &Rhs) const noexcept {
    // TODO: on device, assert that equal accessors refer to the same memory
    // once device assertions are supported.
    return MId == Rhs.MId;
  }

  std::uint64_t getId() const noexcept { return MId; }

  /// \return a generic pointer to the first element of this accessor's region.
  DataT *getPtr() const noexcept { return MData; }

private:
  // The region of the kernel's local memory block that belongs to this
  // accessor. Set on device only, see LocalAccessorReifier.
  LocalPtrT MData = nullptr;

  // Identity of the accessor, copied to device with the kernel object. Used for
  // comparison and as the value std::hash is computed from.
  std::uint64_t MId = 0;

  range<AdjustedDims> MRange;

  template <typename, int> friend class LocalAccessorBase;
  friend class LocalAccessorReifier;
};

template <typename DataT, int Dimensions>
struct ObjectKindOf<LocalAccessorBase<DataT, Dimensions>> {
  static constexpr ObjectKind value = ObjectKind::LocalAccessor;
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_DETAIL_LOCAL_ACCESSOR_BASE_HPP
