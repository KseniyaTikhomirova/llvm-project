//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_INDEX_SPACE_CLASSES_HPP
#define _LIBSYCL___IMPL_INDEX_SPACE_CLASSES_HPP

#include <sycl/__impl/detail/config.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

  class Builder;
template <int Dimensions = 1> class RawArray {
  static_assert(Dimensions >= 1 && Dimensions <= 3,
                "RawArray can only be 1, 2, or 3 Dimensional.");

public:
  template <int N = Dimensions, std::enable_if_t<N == 1, bool> = true>
  RawArray(size_t Dim0 = 0) : MArray{Dim0} {}

  template <int N = Dimensions, std::enable_if_t<N == 2, bool> = true>
  RawArray(size_t Dim0, size_t Dim1) : MArray{Dim0, Dim1} {}

  template <int N = Dimensions, std::enable_if_t<N == 2, bool> = true>
  RawArray() : RawArray(0, 0) {}

  template <int N = Dimensions, std::enable_if_t<N == 3, bool> = true>
  RawArray(size_t Dim0, size_t Dim1, size_t Dim2) : MArray{Dim0, Dim1, Dim2} {}

  template <int N = Dimensions, std::enable_if_t<N == 3, bool> = true>
  RawArray() : RawArray(0, 0, 0) {}

  // Results in undefined behavior if dimension is not in the range [0,
  // Dimensions).
  std::size_t get(int Dimension) const noexcept {
    assert(Dimension >= 0 && Dimension < Dimensions &&
           "Dimention is out of range.");
    return MArray[Dimension];
  }

  // Results in undefined behavior if dimension is not in the range [0,
  // Dimensions).
  std::size_t &operator[](int Dimension) noexcept {
    assert(Dimension >= 0 && Dimension < Dimensions &&
           "Dimention is out of range.");
    return MArray[Dimension];
  }

  // Results in undefined behavior if dimension is not in the range [0,
  // Dimensions).
  std::size_t operator[](int Dimension) const noexcept {
    assert(Dimension >= 0 && Dimension < Dimensions &&
           "Dimention is out of range.");
    return MArray[Dimension];
  }

  RawArray(const RawArray<Dimensions> &rhs) = default;
  RawArray(RawArray<Dimensions> &&rhs) = default;
  RawArray<Dimensions> &operator=(const RawArray<Dimensions> &rhs) = default;
  RawArray<Dimensions> &operator=(RawArray<Dimensions> &&rhs) = default;
  ~RawArray() = default;

  friend bool operator==(const RawArray<Dimensions> &lhs,
                         const RawArray<Dimensions> &rhs) {
    for (int i = 0; i < Dimensions; ++i) {
      if (lhs.MArray[i] != rhs.MArray[i]) {
        return false;
      }
    }
    return true;
  }

  friend bool operator!=(const RawArray<Dimensions> &lhs,
                         const RawArray<Dimensions> &rhs) {
    for (int i = 0; i < Dimensions; ++i) {
      if (lhs.MArray[i] != rhs.MArray[i]) {
        return true;
      }
    }
    return false;
  }

protected:
  size_t MArray[Dimensions];
};
} // namespace detail

template <int Dimensions = 1>
class range : public detail::RawArray<Dimensions> {
  static_assert(Dimensions >= 1 && Dimensions <= 3,
                "range can only be 1, 2, or 3 Dimensional.");
  using Base = detail::RawArray<Dimensions>;

public:
  static constexpr int dimensions = Dimensions;
  range() noexcept = default;
  range(const range<Dimensions> &rhs) = default;
  range(range<Dimensions> &&rhs) = default;
  range<Dimensions> &operator=(const range<Dimensions> &rhs) = default;
  range<Dimensions> &operator=(range<Dimensions> &&rhs) = default;

  /* The following constructor is only available in the range class
specialization where: Dimensions==1 */
  template <int N = Dimensions, std::enable_if_t<N == 1, bool> = true>
  range(std::size_t dim0) noexcept : Base(dim0) {}

  /* The following constructor is only available in the range class
  specialization where: Dimensions==2 */
  template <int N = Dimensions, std::enable_if_t<N == 2, bool> = true>
  range(std::size_t dim0, std::size_t dim1) noexcept : Base(dim0, dim1) {}

  /* The following constructor is only available in the range class
  specialization where: Dimensions==3 */
  template <int N = Dimensions, std::enable_if_t<N == 3, bool> = true>
  range(std::size_t dim0, std::size_t dim1, std::size_t dim2) noexcept
      : Base(dim0, dim1, dim2) {}

  /*
  Declared and implemented in detail::RawArray:
      std::size_t get(int dimension) const noexcept;
      std::size_t& operator[](int dimension) noexcept;
      std::size_t operator[](int dimension) const noexcept;
  */

  std::size_t size() const noexcept {
    std::size_t size = 1;
    for (int i = 0; i < Dimensions; ++i) {
      size *= Base::MArray[i];
    }
    return size;
  }

  // Operators
};

// Deduction guides
#ifdef __cpp_deduction_guides
range(std::size_t) -> range<1>;
range(std::size_t, std::size_t) -> range<2>;
range(std::size_t, std::size_t, std::size_t) -> range<3>;
#endif

template <int Dimensions = 1, bool WithOffset = true> class item;

template <int Dimensions = 1> class id : public detail::RawArray<Dimensions> {
  static_assert(Dimensions >= 1 && Dimensions <= 3,
                "id can only be 1, 2, or 3 Dimensional.");
  using Base = detail::RawArray<Dimensions>;

  /* Helper class for conversion operator. Void type is not suitable. User
   * cannot even try to get address of the operator PrivateTag(). User
   * may try to get an address of operator void() and will get the
   * compile-time error */
  class PrivateTag;
  template <bool Condition, typename T>
  using EnableIfT = std::conditional_t<Condition, T, PrivateTag>;

public:
  static constexpr int dimensions = Dimensions;

  id() noexcept = default;
  id(const id<Dimensions> &rhs) = default;
  id(id<Dimensions> &&rhs) = default;
  id<Dimensions> &operator=(const id<Dimensions> &rhs) = default;
  id<Dimensions> &operator=(id<Dimensions> &&rhs) = default;

  /* The following constructor is only available in the id class
   * specialization where: Dimensions==1 */
  template <int N = Dimensions, std::enable_if_t<N == 1, bool> = true>
  id(std::size_t dim0) noexcept : Base(dim0) {}
  /* The following constructor is only available in the id class
   * specialization where: Dimensions==2 */
  template <int N = Dimensions, std::enable_if_t<N == 2, bool> = true>
  id(std::size_t dim0, std::size_t dim1) noexcept : Base(dim0, dim1) {}
  /* The following constructor is only available in the id class
   * specialization where: Dimensions==3 */
  template <int N = Dimensions, std::enable_if_t<N == 3, bool> = true>
  id(std::size_t dim0, std::size_t dim1, std::size_t dim2) noexcept
      : Base(dim0, dim1, dim2) {}

  template <int N = Dimensions, std::enable_if_t<N == 1, bool> = true>
  id(const range<Dimensions> &range) noexcept : Base(range.get(0)) {}

  template <int N = Dimensions, std::enable_if_t<N == 2, bool> = true>
  id(const range<Dimensions> &range) noexcept
      : Base(range.get(0), range.get(1)) {}

  template <int N = Dimensions, std::enable_if_t<N == 3, bool> = true>
  id(const range<Dimensions> &range) noexcept
      : Base(range.get(0), range.get(1), range.get(2)) {}

  template <int N = Dimensions, std::enable_if_t<N == 1, bool> = true>
  id(const item<Dimensions> &item) noexcept : Base(item.get_id(0)) {}

  template <int N = Dimensions, std::enable_if_t<N == 2, bool> = true>
  id(const item<Dimensions> &item) noexcept
      : Base(item.get_id(0), item.get_id(1)) {}

  template <int N = Dimensions, std::enable_if_t<N == 3, bool> = true>
  id(const item<Dimensions> &item) noexcept
      : Base(item.get_id(0), item.get_id(1), item.get_id(2)) {}
  /*
    Declared and implemented in detail::RawArray:
        std::size_t get(int dimension) const noexcept;
        std::size_t& operator[](int dimension) noexcept;
        std::size_t operator[](int dimension) const noexcept;
    */

  // Only available if Dimensions == 1
  // Template operator is not allowed because it disables further type
  //   conversion. For example, the next code will not work in case of template
  //   conversion:
  //   int a = id<1>(value);
  operator EnableIfT<(Dimensions == 1), std::size_t>() const noexcept {
    return Base::get(0);
  }

  // Operators
};

// Deduction guides
#ifdef __cpp_deduction_guides
id(std::size_t) -> id<1>;
id(std::size_t, std::size_t) -> id<2>;
id(std::size_t, std::size_t, std::size_t) -> id<3>;
#endif

template <int Dimensions /* = 1*/, bool WithOffset /* = true*/> class item {
public:
  static constexpr int dimensions = Dimensions;

  item() = delete;

  item(const item &rhs) = default;

  item(item<Dimensions, WithOffset> &&rhs) = default;

  item &operator=(const item &rhs) = default;

  item &operator=(item &&rhs) = default;

  friend bool operator==(const item<Dimensions, WithOffset> &lhs,
                         const item<Dimensions, WithOffset> &rhs) {
    if constexpr (WithOffset)
      return (lhs.MId == rhs.MId) && (lhs.MRange == rhs.MRange) &&
             (lhs.MOffset == rhs.MOffset);
    else
      return (lhs.MId == rhs.MId) && (lhs.MRange == rhs.MRange);
  }

  friend bool operator!=(const item<Dimensions, WithOffset> &lhs,
                         const item<Dimensions, WithOffset> &rhs) {
    return !(lhs == rhs);
  }

  id<Dimensions> get_id() const noexcept { return MId; }

  // toadd everywhere  __builtin_assume
  std::size_t get_id(int dimension) const noexcept {
    return MId.get(dimension);
  }

  std::size_t operator[](int dimension) const noexcept {
    return MId[dimension];
  }

  range<Dimensions> get_range() const noexcept { return MRange; }

  std::size_t get_range(int dimension) const noexcept {
    return MRange[dimension];
  }

  // Deprecated in SYCL 2020.
  // only available if WithOffset is true
  template <bool HasOffset = WithOffset,
            std::enable_if_t<HasOffset == true, bool> = true>
  id<Dimensions> get_offset() const noexcept {
    return MOffset;
  }

  // Deprecated in SYCL 2020.
  // only available if WithOffset is false
  template <bool HasOffset = WithOffset,
            std::enable_if_t<HasOffset == false, bool> = true>
  operator item<Dimensions, true>() const noexcept {
    return item<Dimensions, true>(MRange, MId, id<Dimensions>{});
  }

  // only available if Dimensions == 1
  operator std::size_t() const noexcept;

  std::size_t get_linear_id() const noexcept {
    if constexpr (WithOffset) {
      if constexpr (1 == Dimensions) {
        return MId;
      }
      if constexpr (2 == Dimensions) {
        return (MId[0] - MOffset[0]) * MRange[1] + (MId[1] - MOffset[1]);
      }
      return ((MId[0] - MOffset[0]) * MRange[1] * MRange[2]) +
             ((MId[1] - MOffset[1]) * MRange[2]) + (MId[2] - MOffset[2]);
    } else {
      if constexpr (1 == Dimensions) {
        return MId[0];
      }
      if constexpr (2 == Dimensions) {
        return MId[0] * MRange[1] + MId[1];
      }
      return (MId[0] * MRange[1] * MRange[2]) + (MId[1] * MRange[2]) + MId[2];
    }
  }

protected:
  template <bool HasOffset = WithOffset,
            std::enable_if_t<HasOffset == true, bool> = true>
  item(const sycl::range<Dimensions> &range, const sycl::id<Dimensions> &id,
       const sycl::id<Dimensions> &offset)
      : MRange(range), MId(id), MOffset(offset) {}

  template <bool HasOffset = WithOffset,
            std::enable_if_t<HasOffset == false, bool> = true>
  item(const range<Dimensions> &range, const id<Dimensions> &id)
      : MRange(range), MId(id), MOffset() {}

private:
  range<Dimensions> MRange;
  id<Dimensions> MId;
  id<Dimensions> MOffset;

    friend class detail::Builder;
};

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_INDEX_SPACE_CLASSES_HPP
