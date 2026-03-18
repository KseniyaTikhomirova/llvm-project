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
#include <sycl/__impl/exception.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

template <int dimensions> class id;
template <int dimensions> class range;
namespace detail {

template <int dimensions = 1> class array {
  static_assert(dimensions >= 1, "Array cannot be 0-dimensional.");

public:
  /* The following constructor is only available in the array struct
   * specialization where: dimensions==1 */
  template <int N = dimensions>
  array(typename std::enable_if_t<(N == 1), size_t> dim0 = 0)
      : common_array{dim0} {}

  /* The following constructors are only available in the array struct
   * specialization where: dimensions==2 */
  template <int N = dimensions>
  array(typename std::enable_if_t<(N == 2), size_t> dim0, size_t dim1)
      : common_array{dim0, dim1} {}

  template <int N = dimensions, std::enable_if_t<(N == 2), size_t> = 0>
  array() : array(0, 0) {}

  /* The following constructors are only available in the array struct
   * specialization where: dimensions==3 */
  template <int N = dimensions>
  array(typename std::enable_if_t<(N == 3), size_t> dim0, size_t dim1,
        size_t dim2)
      : common_array{dim0, dim1, dim2} {}

  template <int N = dimensions, std::enable_if_t<(N == 3), size_t> = 0>
  array() : array(0, 0, 0) {}

  // Conversion operators to derived classes
  operator sycl::id<dimensions>() const {
    sycl::id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result[i] = common_array[i];
    }
    return result;
  }

  size_t get(int dimension) const {
    check_dimension(dimension);
    return common_array[dimension];
  }

  size_t &operator[](int dimension) {
    check_dimension(dimension);
    return common_array[dimension];
  }

  size_t operator[](int dimension) const {
    check_dimension(dimension);
    return common_array[dimension];
  }

  array(const array<dimensions> &rhs) = default;
  array(array<dimensions> &&rhs) = default;
  array<dimensions> &operator=(const array<dimensions> &rhs) = default;
  array<dimensions> &operator=(array<dimensions> &&rhs) = default;

  // Returns true iff all elements in 'this' are equal to
  // the corresponding elements in 'rhs'.
  bool operator==(const array<dimensions> &rhs) const {
    for (int i = 0; i < dimensions; ++i) {
      if (this->common_array[i] != rhs.common_array[i]) {
        return false;
      }
    }
    return true;
  }

  // Returns true iff there is at least one element in 'this'
  // which is not equal to the corresponding element in 'rhs'.
  bool operator!=(const array<dimensions> &rhs) const {
    for (int i = 0; i < dimensions; ++i) {
      if (this->common_array[i] != rhs.common_array[i]) {
        return true;
      }
    }
    return false;
  }

protected:
  size_t common_array[dimensions];
  __SYCL_ALWAYS_INLINE void check_dimension(int dimension) const {
#ifndef __SYCL_DEVICE_ONLY__
    if (dimension >= dimensions || dimension < 0) {
      throw sycl::exception(make_error_code(errc::invalid),
                            "Index out of range");
    }
#endif
    (void)dimension;
  }
};

class Builder;

} // namespace detail

template <int Dimensions = 1> class range : public detail::array<Dimensions> {
public:
  static constexpr int dimensions = Dimensions;

private:
  static_assert(Dimensions >= 1 && Dimensions <= 3,
                "range can only be 1, 2, or 3 Dimensional.");
  using base = detail::array<Dimensions>;
  template <typename N, typename T>
  using IntegralType = std::enable_if_t<std::is_integral_v<N>, T>;

public:
  /* The following constructor is only available in the range class
  specialization where: Dimensions==1 */
  template <int N = Dimensions>
  range(typename std::enable_if_t<(N == 1), size_t> dim0) : base(dim0) {}

  /* The following constructor is only available in the range class
  specialization where: Dimensions==2 */
  template <int N = Dimensions>
  range(typename std::enable_if_t<(N == 2), size_t> dim0, size_t dim1)
      : base(dim0, dim1) {}

  /* The following constructor is only available in the range class
  specialization where: Dimensions==3 */
  template <int N = Dimensions>
  range(typename std::enable_if_t<(N == 3), size_t> dim0, size_t dim1,
        size_t dim2)
      : base(dim0, dim1, dim2) {}

  size_t size() const {
    size_t size = 1;
    for (int i = 0; i < Dimensions; ++i) {
      size *= this->common_array[i];
    }
    return size;
  }

  range(const range<Dimensions> &rhs) = default;
  range(range<Dimensions> &&rhs) = default;
  range<Dimensions> &operator=(const range<Dimensions> &rhs) = default;
  range<Dimensions> &operator=(range<Dimensions> &&rhs) = default;
  range() = default;

// OP is: +, -, *, /, %, <<, >>, &, |, ^, &&, ||, <, >, <=, >=
#define __SYCL_GEN_OPT_BASE(op)                                                \
  friend range<Dimensions> operator op(const range<Dimensions> &lhs,           \
                                       const range<Dimensions> &rhs) {         \
    range<Dimensions> result(lhs);                                             \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      result.common_array[i] = lhs.common_array[i] op rhs.common_array[i];     \
    }                                                                          \
    return result;                                                             \
  }

#ifndef __SYCL_DISABLE_ID_TO_INT_CONV__
  // Enable operators with integral types only
#define __SYCL_GEN_OPT(op)                                                     \
  __SYCL_GEN_OPT_BASE(op)                                                      \
  template <typename T>                                                        \
  friend IntegralType<T, range<Dimensions>> operator op(                       \
      const range<Dimensions> &lhs, const T &rhs) {                            \
    range<Dimensions> result(lhs);                                             \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      result.common_array[i] = lhs.common_array[i] op rhs;                     \
    }                                                                          \
    return result;                                                             \
  }                                                                            \
  template <typename T>                                                        \
  friend IntegralType<T, range<Dimensions>> operator op(                       \
      const T &lhs, const range<Dimensions> &rhs) {                            \
    range<Dimensions> result(rhs);                                             \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      result.common_array[i] = lhs op rhs.common_array[i];                     \
    }                                                                          \
    return result;                                                             \
  }
#else
#define __SYCL_GEN_OPT(op)                                                     \
  __SYCL_GEN_OPT_BASE(op)                                                      \
  friend range<Dimensions> operator op(const range<Dimensions> &lhs,           \
                                       const size_t &rhs) {                    \
    range<Dimensions> result(lhs);                                             \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      result.common_array[i] = lhs.common_array[i] op rhs;                     \
    }                                                                          \
    return result;                                                             \
  }                                                                            \
  friend range<Dimensions> operator op(const size_t &lhs,                      \
                                       const range<Dimensions> &rhs) {         \
    range<Dimensions> result(rhs);                                             \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      result.common_array[i] = lhs op rhs.common_array[i];                     \
    }                                                                          \
    return result;                                                             \
  }
#endif // __SYCL_DISABLE_ID_TO_INT_CONV__

  __SYCL_GEN_OPT(+)
  __SYCL_GEN_OPT(-)
  __SYCL_GEN_OPT(*)
  __SYCL_GEN_OPT(/)
  __SYCL_GEN_OPT(%)
  __SYCL_GEN_OPT(<<)
  __SYCL_GEN_OPT(>>)
  __SYCL_GEN_OPT(&)
  __SYCL_GEN_OPT(|)
  __SYCL_GEN_OPT(^)
  __SYCL_GEN_OPT(&&)
  __SYCL_GEN_OPT(||)
  __SYCL_GEN_OPT(<)
  __SYCL_GEN_OPT(>)
  __SYCL_GEN_OPT(<=)
  __SYCL_GEN_OPT(>=)

#undef __SYCL_GEN_OPT
#undef __SYCL_GEN_OPT_BASE

// OP is: +=, -=, *=, /=, %=, <<=, >>=, &=, |=, ^=
#define __SYCL_GEN_OPT(op)                                                     \
  friend range<Dimensions> &operator op(range<Dimensions> &lhs,                \
                                        const range<Dimensions> &rhs) {        \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      lhs.common_array[i] op rhs[i];                                           \
    }                                                                          \
    return lhs;                                                                \
  }                                                                            \
  friend range<Dimensions> &operator op(range<Dimensions> &lhs,                \
                                        const size_t &rhs) {                   \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      lhs.common_array[i] op rhs;                                              \
    }                                                                          \
    return lhs;                                                                \
  }

  __SYCL_GEN_OPT(+=)
  __SYCL_GEN_OPT(-=)
  __SYCL_GEN_OPT(*=)
  __SYCL_GEN_OPT(/=)
  __SYCL_GEN_OPT(%=)
  __SYCL_GEN_OPT(<<=)
  __SYCL_GEN_OPT(>>=)
  __SYCL_GEN_OPT(&=)
  __SYCL_GEN_OPT(|=)
  __SYCL_GEN_OPT(^=)

#undef __SYCL_GEN_OPT

// OP is unary +, -
#define __SYCL_GEN_OPT(op)                                                     \
  friend range<Dimensions> operator op(const range<Dimensions> &rhs) {         \
    range<Dimensions> result(rhs);                                             \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      result.common_array[i] = (op rhs.common_array[i]);                       \
    }                                                                          \
    return result;                                                             \
  }

  __SYCL_GEN_OPT(+)
  __SYCL_GEN_OPT(-)

#undef __SYCL_GEN_OPT

// OP is prefix ++, --
#define __SYCL_GEN_OPT(op)                                                     \
  friend range<Dimensions> &operator op(range<Dimensions> &rhs) {              \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      op rhs.common_array[i];                                                  \
    }                                                                          \
    return rhs;                                                                \
  }

  __SYCL_GEN_OPT(++)
  __SYCL_GEN_OPT(--)

#undef __SYCL_GEN_OPT

// OP is postfix ++, --
#define __SYCL_GEN_OPT(op)                                                     \
  friend range<Dimensions> operator op(range<Dimensions> &lhs, int) {          \
    range<Dimensions> old_lhs(lhs);                                            \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      op lhs.common_array[i];                                                  \
    }                                                                          \
    return old_lhs;                                                            \
  }

  __SYCL_GEN_OPT(++)
  __SYCL_GEN_OPT(--)

#undef __SYCL_GEN_OPT

private:
  friend class handler;
  friend class detail::Builder;

  // Adjust the first dim of the range
  void set_range_dim0(const size_t dim0) { this->common_array[0] = dim0; }
};

#ifdef __cpp_deduction_guides
range(size_t)->range<1>;
range(size_t, size_t)->range<2>;
range(size_t, size_t, size_t)->range<3>;
#endif

template <int Dimensions, bool with_offset> class item;

template <int Dimensions = 1> class id : public detail::array<Dimensions> {
public:
  static constexpr int dimensions = Dimensions;

private:
  using base = detail::array<Dimensions>;
  static_assert(Dimensions >= 1 && Dimensions <= 3,
                "id can only be 1, 2, or 3 Dimensional.");
  template <int N, int val, typename T>
  using ParamTy = std::enable_if_t<(N == val), T>;

#ifndef __SYCL_DISABLE_ID_TO_INT_CONV__
  /* Helper class for conversion operator. Void type is not suitable. User
   * cannot even try to get address of the operator __private_class(). User
   * may try to get an address of operator void() and will get the
   * compile-time error */
  class __private_class;

  template <typename N, typename T>
  using EnableIfIntegral = std::enable_if_t<std::is_integral_v<N>, T>;
  template <bool B, typename T>
  using EnableIfT = std::conditional_t<B, T, __private_class>;
#endif // __SYCL_DISABLE_ID_TO_INT_CONV__

public:
  id() = default;

  /* The following constructor is only available in the id struct
   * specialization where: Dimensions==1 */
  template <int N = Dimensions> id(ParamTy<N, 1, size_t> dim0) : base(dim0) {}

  template <int N = Dimensions>
  id(ParamTy<N, 1, const range<Dimensions>> &range_size)
      : base(range_size.get(0)) {}

  template <int N = Dimensions, bool with_offset = true>
  id(ParamTy<N, 1, const item<Dimensions, with_offset>> &item)
      : base(item.get_id(0)) {}

  /* The following constructor is only available in the id struct
   * specialization where: Dimensions==2 */
  template <int N = Dimensions>
  id(ParamTy<N, 2, size_t> dim0, size_t dim1) : base(dim0, dim1) {}

  template <int N = Dimensions>
  id(ParamTy<N, 2, const range<Dimensions>> &range_size)
      : base(range_size.get(0), range_size.get(1)) {}

  template <int N = Dimensions, bool with_offset = true>
  id(ParamTy<N, 2, const item<Dimensions, with_offset>> &item)
      : base(item.get_id(0), item.get_id(1)) {}

  /* The following constructor is only available in the id struct
   * specialization where: Dimensions==3 */
  template <int N = Dimensions>
  id(ParamTy<N, 3, size_t> dim0, size_t dim1, size_t dim2)
      : base(dim0, dim1, dim2) {}

  template <int N = Dimensions>
  id(ParamTy<N, 3, const range<Dimensions>> &range_size)
      : base(range_size.get(0), range_size.get(1), range_size.get(2)) {}

  template <int N = Dimensions, bool with_offset = true>
  id(ParamTy<N, 3, const item<Dimensions, with_offset>> &item)
      : base(item.get_id(0), item.get_id(1), item.get_id(2)) {}

#ifndef __SYCL_DISABLE_ID_TO_INT_CONV__
  /* Template operator is not allowed because it disables further type
   * conversion. For example, the next code will not work in case of template
   * conversion:
   * int a = id<1>(value); */

  __SYCL_ALWAYS_INLINE operator EnableIfT<(Dimensions == 1), size_t>() const {
    size_t Result = this->common_array[0];
    __SYCL_ASSUME_INT(Result);
    return Result;
  }
#endif // __SYCL_DISABLE_ID_TO_INT_CONV__

// OP is: ==, !=
#ifndef __SYCL_DISABLE_ID_TO_INT_CONV__
  using detail::array<Dimensions>::operator==;
  // Needed for clang in C++20 mode as the above operator== would be ambigious
  // between regular/reversed call for "Id == Id" case.
  bool operator==(const id<Dimensions> &rhs) const {
    return this->detail::array<Dimensions>::operator==(rhs);
  }
#if __cpp_impl_three_way_comparison < 201907
  using detail::array<Dimensions>::operator!=;
#endif

  /* Enable operators with integral types.
   * Template operators take precedence than type conversion. In the case of
   * non-template operators, ambiguity appears: "id op size_t" may refer
   * "size_t op size_t" and "id op size_t". In case of template operators it
   * will be "id op size_t"*/
#define __SYCL_GEN_OPT(op)                                                     \
  template <typename T>                                                        \
  EnableIfIntegral<T, bool> operator op(const T &rhs) const {                  \
    if (this->common_array[0] != rhs)                                          \
      return false op true;                                                    \
    return true op true;                                                       \
  }                                                                            \
  template <typename T>                                                        \
  friend EnableIfIntegral<T, bool> operator op(const T &lhs,                   \
                                               const id<Dimensions> &rhs) {    \
    if (lhs != rhs.common_array[0])                                            \
      return false op true;                                                    \
    return true op true;                                                       \
  }

  __SYCL_GEN_OPT(==)
  __SYCL_GEN_OPT(!=)

#undef __SYCL_GEN_OPT

#endif // __SYCL_DISABLE_ID_TO_INT_CONV__

// OP is: +, -, *, /, %, <<, >>, &, |, ^, &&, ||, <, >, <=, >=
#define __SYCL_GEN_OPT_BASE(op)                                                \
  friend id<Dimensions> operator op(const id<Dimensions> &lhs,                 \
                                    const id<Dimensions> &rhs) {               \
    id<Dimensions> result;                                                     \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      result.common_array[i] = lhs.common_array[i] op rhs.common_array[i];     \
    }                                                                          \
    return result;                                                             \
  }

#ifndef __SYCL_DISABLE_ID_TO_INT_CONV__
// Enable operators with integral types only
#define __SYCL_GEN_OPT(op)                                                     \
  __SYCL_GEN_OPT_BASE(op)                                                      \
  template <typename T>                                                        \
  friend EnableIfIntegral<T, id<Dimensions>> operator op(                      \
      const id<Dimensions> &lhs, const T &rhs) {                               \
    id<Dimensions> result;                                                     \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      result.common_array[i] = lhs.common_array[i] op rhs;                     \
    }                                                                          \
    return result;                                                             \
  }                                                                            \
  template <typename T>                                                        \
  friend EnableIfIntegral<T, id<Dimensions>> operator op(                      \
      const T &lhs, const id<Dimensions> &rhs) {                               \
    id<Dimensions> result;                                                     \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      result.common_array[i] = lhs op rhs.common_array[i];                     \
    }                                                                          \
    return result;                                                             \
  }
#else
#define __SYCL_GEN_OPT(op)                                                     \
  __SYCL_GEN_OPT_BASE(op)                                                      \
  friend id<Dimensions> operator op(const id<Dimensions> &lhs,                 \
                                    const size_t &rhs) {                       \
    id<Dimensions> result;                                                     \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      result.common_array[i] = lhs.common_array[i] op rhs;                     \
    }                                                                          \
    return result;                                                             \
  }                                                                            \
  friend id<Dimensions> operator op(const size_t &lhs,                         \
                                    const id<Dimensions> &rhs) {               \
    id<Dimensions> result;                                                     \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      result.common_array[i] = lhs op rhs.common_array[i];                     \
    }                                                                          \
    return result;                                                             \
  }
#endif // __SYCL_DISABLE_ID_TO_INT_CONV__

  __SYCL_GEN_OPT(+)
  __SYCL_GEN_OPT(-)
  __SYCL_GEN_OPT(*)
  __SYCL_GEN_OPT(/)
  __SYCL_GEN_OPT(%)
  __SYCL_GEN_OPT(<<)
  __SYCL_GEN_OPT(>>)
  __SYCL_GEN_OPT(&)
  __SYCL_GEN_OPT(|)
  __SYCL_GEN_OPT(^)
  __SYCL_GEN_OPT(&&)
  __SYCL_GEN_OPT(||)
  __SYCL_GEN_OPT(<)
  __SYCL_GEN_OPT(>)
  __SYCL_GEN_OPT(<=)
  __SYCL_GEN_OPT(>=)

#undef __SYCL_GEN_OPT
#undef __SYCL_GEN_OPT_BASE

// OP is: +=, -=, *=, /=, %=, <<=, >>=, &=, |=, ^=
#define __SYCL_GEN_OPT(op)                                                     \
  friend id<Dimensions> &operator op(id<Dimensions> &lhs,                      \
                                     const id<Dimensions> &rhs) {              \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      lhs.common_array[i] op rhs.common_array[i];                              \
    }                                                                          \
    return lhs;                                                                \
  }                                                                            \
  friend id<Dimensions> &operator op(id<Dimensions> &lhs, const size_t &rhs) { \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      lhs.common_array[i] op rhs;                                              \
    }                                                                          \
    return lhs;                                                                \
  }

  __SYCL_GEN_OPT(+=)
  __SYCL_GEN_OPT(-=)
  __SYCL_GEN_OPT(*=)
  __SYCL_GEN_OPT(/=)
  __SYCL_GEN_OPT(%=)
  __SYCL_GEN_OPT(<<=)
  __SYCL_GEN_OPT(>>=)
  __SYCL_GEN_OPT(&=)
  __SYCL_GEN_OPT(|=)
  __SYCL_GEN_OPT(^=)

#undef __SYCL_GEN_OPT

// OP is unary +, -
#define __SYCL_GEN_OPT(op)                                                     \
  friend id<Dimensions> operator op(const id<Dimensions> &rhs) {               \
    id<Dimensions> result;                                                     \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      result.common_array[i] = (op rhs.common_array[i]);                       \
    }                                                                          \
    return result;                                                             \
  }

  __SYCL_GEN_OPT(+)
  __SYCL_GEN_OPT(-)

#undef __SYCL_GEN_OPT

// OP is prefix ++, --
#define __SYCL_GEN_OPT(op)                                                     \
  friend id<Dimensions> &operator op(id<Dimensions> &rhs) {                    \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      op rhs.common_array[i];                                                  \
    }                                                                          \
    return rhs;                                                                \
  }

  __SYCL_GEN_OPT(++)
  __SYCL_GEN_OPT(--)

#undef __SYCL_GEN_OPT

// OP is postfix ++, --
#define __SYCL_GEN_OPT(op)                                                     \
  friend id<Dimensions> operator op(id<Dimensions> &lhs, int) {                \
    id<Dimensions> old_lhs;                                                    \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      old_lhs.common_array[i] = lhs.common_array[i];                           \
      op lhs.common_array[i];                                                  \
    }                                                                          \
    return old_lhs;                                                            \
  }

  __SYCL_GEN_OPT(++)
  __SYCL_GEN_OPT(--)

#undef __SYCL_GEN_OPT
};

namespace detail {
template <int Dimensions>
size_t getOffsetForId(range<Dimensions> Range, id<Dimensions> Id,
                      id<Dimensions> Offset) {
  size_t offset = 0;
  for (int i = 0; i < Dimensions; ++i)
    offset = offset * Range[i] + Offset[i] + Id[i];
  return offset;
}

inline id<1> getDelinearizedId(const range<1> &, size_t Index) {
  return {Index};
}

inline id<2> getDelinearizedId(const range<2> &Range, size_t Index) {
  size_t X = Index % Range[1];
  size_t Y = Index / Range[1];
  return {Y, X};
}

inline id<3> getDelinearizedId(const range<3> &Range, size_t Index) {
  size_t D1D2 = Range[1] * Range[2];
  size_t Z = Index / D1D2;
  size_t ZRest = Index % D1D2;
  size_t Y = ZRest / Range[2];
  size_t X = ZRest % Range[2];
  return {Z, Y, X};
}
} // namespace detail

// C++ feature test macros are supported by all supported compilers
// with the exception of MSVC 1914. It doesn't support deduction guides.
#ifdef __cpp_deduction_guides
id(size_t)->id<1>;
id(size_t, size_t)->id<2>;
id(size_t, size_t, size_t)->id<3>;
#endif
 
namespace detail {
template <int Dims, bool WithOffset> struct ItemBase;

template <int Dims> struct ItemBase<Dims, true> {

  bool operator==(const ItemBase &Rhs) const {
    return (Rhs.MIndex == MIndex) && (Rhs.MExtent == MExtent) &&
           (Rhs.MOffset == MOffset);
  }

  bool operator!=(const ItemBase &Rhs) const { return !((*this) == Rhs); }

  size_t get_linear_id() const {
    if (1 == Dims) {
      return MIndex[0] - MOffset[0];
    }
    if (2 == Dims) {
      return (MIndex[0] - MOffset[0]) * MExtent[1] + (MIndex[1] - MOffset[1]);
    }
    return ((MIndex[0] - MOffset[0]) * MExtent[1] * MExtent[2]) +
           ((MIndex[1] - MOffset[1]) * MExtent[2]) + (MIndex[2] - MOffset[2]);
  }

  range<Dims> MExtent;
  id<Dims> MIndex;
  id<Dims> MOffset;
};

template <int Dims> struct ItemBase<Dims, false> {

  bool operator==(const ItemBase &Rhs) const {
    return (Rhs.MIndex == MIndex) && (Rhs.MExtent == MExtent);
  }

  bool operator!=(const ItemBase &Rhs) const { return !((*this) == Rhs); }

  operator ItemBase<Dims, true>() const {
    return ItemBase<Dims, true>(MExtent, MIndex, id<Dims>{});
  }

  size_t get_linear_id() const {
    if (1 == Dims) {
      return MIndex[0];
    }
    if (2 == Dims) {
      return MIndex[0] * MExtent[1] + MIndex[1];
    }
    return (MIndex[0] * MExtent[1] * MExtent[2]) + (MIndex[1] * MExtent[2]) +
           MIndex[2];
  }

  range<Dims> MExtent;
  id<Dims> MIndex;
};

} // namespace detail

template <int Dimensions = 1, bool with_offset = true> class item {
public:
  static constexpr int dimensions = Dimensions;

private:
#ifndef __SYCL_DISABLE_ITEM_TO_INT_CONV__
  /* Helper class for conversion operator. Void type is not suitable. User
   * cannot even try to get address of the operator __private_class(). User
   * may try to get an address of operator void() and will get the
   * compile-time error */
  class __private_class;

  template <bool B, typename T>
  using EnableIfT = std::conditional_t<B, T, __private_class>;
#endif // __SYCL_DISABLE_ITEM_TO_INT_CONV__
public:
  item() = delete;

  id<Dimensions> get_id() const { return MImpl.MIndex; }

  size_t __SYCL_ALWAYS_INLINE get_id(int Dimension) const {
    size_t Id = MImpl.MIndex[Dimension];
    __SYCL_ASSUME_INT(Id);
    return Id;
  }

  size_t __SYCL_ALWAYS_INLINE operator[](int Dimension) const {
    size_t Id = MImpl.MIndex[Dimension];
    __SYCL_ASSUME_INT(Id);
    return Id;
  }

  range<Dimensions> get_range() const { return MImpl.MExtent; }

  size_t __SYCL_ALWAYS_INLINE get_range(int Dimension) const {
    size_t Id = MImpl.MExtent[Dimension];
    __SYCL_ASSUME_INT(Id);
    return Id;
  }
#ifndef __SYCL_DISABLE_ITEM_TO_INT_CONV__
  operator EnableIfT<Dimensions == 1, std::size_t>() const { return get_id(0); }
#endif // __SYCL_DISABLE_ITEM_TO_INT_CONV__
  template <bool has_offset = with_offset>
  __SYCL2020_DEPRECATED("offsets are deprecated in SYCL2020")
  std::enable_if_t<has_offset, id<Dimensions>> get_offset() const {
    return MImpl.MOffset;
  }

  template <bool has_offset = with_offset>
  __SYCL2020_DEPRECATED("offsets are deprecated in SYCL2020")
  std::enable_if_t<has_offset, size_t> __SYCL_ALWAYS_INLINE
      get_offset(int Dimension) const {
    size_t Id = MImpl.MOffset[Dimension];
    __SYCL_ASSUME_INT(Id);
    return Id;
  }

  template <bool has_offset = with_offset>
  operator std::enable_if_t<!has_offset, item<Dimensions, true>>() const {
    return item<Dimensions, true>{MImpl.MExtent, MImpl.MIndex, /*Offset*/ {}};
  }

  size_t __SYCL_ALWAYS_INLINE get_linear_id() const {
    size_t Id = MImpl.get_linear_id();
    __SYCL_ASSUME_INT(Id);
    return Id;
  }

  item(const item &rhs) = default;

  item(item<Dimensions, with_offset> &&rhs) = default;

  item &operator=(const item &rhs) = default;

  item &operator=(item &&rhs) = default;

  bool operator==(const item &rhs) const { return rhs.MImpl == MImpl; }

  bool operator!=(const item &rhs) const { return rhs.MImpl != MImpl; }

protected:
  template <bool has_offset = with_offset>
  item(std::enable_if_t<has_offset, const range<Dimensions>> &extent,
       const id<Dimensions> &index, const id<Dimensions> &offset)
      : MImpl{extent, index, offset} {}

  template <bool has_offset = with_offset>
  item(std::enable_if_t<!has_offset, const range<Dimensions>> &extent,
       const id<Dimensions> &index)
      : MImpl{extent, index} {}

  friend class detail::Builder;

  template <int, bool> friend class item;

private:
  detail::ItemBase<Dimensions, with_offset> MImpl;
};

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_INDEX_SPACE_CLASSES_HPP
