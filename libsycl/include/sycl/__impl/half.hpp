//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL 2020 half scalar data type
/// (4.14.1.).
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_HALF_HPP
#define _LIBSYCL___IMPL_HALF_HPP

#include <sycl/__impl/detail/config.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <limits>
#include <type_traits>

#ifndef __FLT16_MANT_DIG__
#  error "Libsycl requires a target with _Float16 support for sycl::half."
#endif

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

// Type in which a mixed half/T operation is computed: T if it is a wider
// floating-point type, NarrowT otherwise.
template <typename T, typename NarrowT>
using HalfMixedT =
    std::conditional_t<std::is_floating_point_v<T> && (sizeof(T) > 2), T,
                       NarrowT>;

#ifndef __SYCL_DEVICE_ONLY__
// On the host, conversions between _Float16 and float are done in software:
// x86 without F16C lowers them to compiler-rt calls (__extendhfsf2,
// __truncsfhf2), which aren't linked by default for MSVC targets.

constexpr float halfBitsToFloat(std::uint16_t Bits) {
  std::uint32_t Sign = static_cast<std::uint32_t>(Bits & 0x8000) << 16;
  std::uint32_t Exp = (Bits >> 10) & 0x1F;
  std::uint32_t Mant = Bits & 0x3FF;
  if (Exp == 0) {
    float Abs = static_cast<float>(Mant) * 0x1p-24f;
    return Sign ? -Abs : Abs;
  }
  if (Exp == 0x1F)
    return __builtin_bit_cast(float, Sign | 0x7F800000 | (Mant << 13));
  return __builtin_bit_cast(float, Sign | ((Exp + 112) << 23) | (Mant << 13));
}

// Rounds to nearest, ties to even. NaNs become quiet NaNs.
constexpr std::uint16_t floatToHalfBits(float Value) {
  std::uint32_t Bits = __builtin_bit_cast(std::uint32_t, Value);
  std::uint32_t Sign = (Bits >> 16) & 0x8000;
  std::uint32_t Abs = Bits & 0x7FFFFFFF;
  std::uint32_t Mag = 0;
  if (Abs > 0x7F800000) {
    Mag = 0x7E00 | ((Abs >> 13) & 0x3FF);
  } else if (Abs >= 0x38800000) {
    // Normal half range or above: rebias the exponent and round off the 13
    // extra mantissa bits. A carry out of the mantissa increments the exponent.
    Abs -= (127 - 15) << 23;
    Abs += 0xFFF + ((Abs >> 13) & 1);
    Mag = Abs >> 13 < 0x7C00 ? Abs >> 13 : 0x7C00;
  } else if (Abs > 0x33000000) {
    // Above half of denorm_min(): the result is subnormal or min().
    std::uint32_t Shift = 126 - (Abs >> 23);
    std::uint32_t Mant = (Abs & 0x7FFFFF) | 0x800000;
    std::uint32_t Rem = Mant & ((1u << Shift) - 1);
    std::uint32_t Halfway = 1u << (Shift - 1);
    Mag = Mant >> Shift;
    if (Rem > Halfway || (Rem == Halfway && (Mag & 1)))
      ++Mag;
  }
  return static_cast<std::uint16_t>(Sign | Mag);
}

// Converts to float rounding to odd, so that rounding the result to half gives
// the same value as rounding Value to half directly.
template <typename T> constexpr float roundToOddFloat(T Value) {
  // Out of the float range the conversion is undefined; such values round to
  // infinity in half anyway.
  if (Value > 65536 || Value < -65536)
    return Value > 0 ? 65536.0f : -65536.0f;
  float Rounded = static_cast<float>(Value);
  T Back = static_cast<T>(Rounded);
  if (Back == Value || Value != Value)
    return Rounded;
  std::uint32_t Bits = __builtin_bit_cast(std::uint32_t, Rounded);
  if (Value > 0 ? Back > Value : Back < Value)
    --Bits;
  return __builtin_bit_cast(float, Bits | 1);
}
#endif

// half is defined in its own namespace so that argument-dependent lookup for
// half finds only its own operators, as for fundamental floating-point types.
namespace half_impl {

// TODO: restrict device usage to devices with aspect::fp16 once clang supports
// an attribute to attach aspects to types.

/// A 16-bit floating-point type conforming to the IEEE 754-2008 half precision
/// storage format.
///
/// This type is only supported on devices that have aspect::fp16.
///
/// \sa SYCL 2020 4.14.1. Scalar data types.
class half {
public:
  half() = default;
  constexpr half(const half &) = default;
  constexpr half(half &&) = default;
  constexpr half &operator=(const half &) = default;
  constexpr half &operator=(half &&) = default;

  /// Constructs a half from \p Value rounded to the nearest representable
  /// half value.
  constexpr half(float Value) : MData(toNative(Value)) {}

  /// \return the value of this half converted to float without precision loss.
  constexpr operator float() const { return toFloat(MData); }

  constexpr half &operator++() { return *this += 1; }
  constexpr half operator++(int) {
    half Old(*this);
    *this += 1;
    return Old;
  }
  constexpr half &operator--() { return *this -= 1; }
  constexpr half operator--(int) {
    half Old(*this);
    *this -= 1;
    return Old;
  }

  friend constexpr half operator+(half rhs) { return rhs; }
  friend constexpr half operator-(half rhs) {
#ifdef __SYCL_DEVICE_ONLY__
    return fromNative(-rhs.MData);
#else
    return fromNative(__builtin_bit_cast(
        _Float16, static_cast<std::uint16_t>(
                      __builtin_bit_cast(std::uint16_t, rhs.MData) ^ 0x8000)));
#endif
  }

  // Applies a binary operator to the values of two halves: natively in device
  // code, in float in host code (see halfBitsToFloat).
#ifdef __SYCL_DEVICE_ONLY__
#  define _LIBSYCL_HALF_OP(lhs, op, rhs) ((lhs).MData op(rhs).MData)
#else
#  define _LIBSYCL_HALF_OP(lhs, op, rhs)                                       \
    (toFloat((lhs).MData) op toFloat((rhs).MData))
#endif

  // The mixed operators are overloads for exact types rather than a template,
  // so that operands that convert implicitly to these types (enumerations,
  // sycl::id<1>, ...) are accepted, as by the built-in operators.
#define _LIBSYCL_HALF_MIXED_TYPES(X, op)                                       \
  X(op, int)                                                                   \
  X(op, unsigned int)                                                          \
  X(op, long)                                                                  \
  X(op, unsigned long)                                                         \
  X(op, long long)                                                             \
  X(op, unsigned long long)                                                    \
  X(op, float)                                                                 \
  X(op, double)                                                                \
  X(op, long double)

#define _LIBSYCL_GEN_MIXED_OPT(op, T)                                          \
  friend constexpr detail::HalfMixedT<T, half> operator op(half lhs, T rhs) {  \
    using OpT = detail::HalfMixedT<T, half>;                                   \
    return static_cast<OpT>(lhs) op static_cast<OpT>(rhs);                     \
  }                                                                            \
  friend constexpr detail::HalfMixedT<T, half> operator op(T lhs, half rhs) {  \
    using OpT = detail::HalfMixedT<T, half>;                                   \
    return static_cast<OpT>(lhs) op static_cast<OpT>(rhs);                     \
  }

#define _LIBSYCL_GEN_OPT(op)                                                   \
  friend constexpr half operator op(half lhs, half rhs) {                      \
    return fromNative(toNative(_LIBSYCL_HALF_OP(lhs, op, rhs)));               \
  }                                                                            \
  _LIBSYCL_HALF_MIXED_TYPES(_LIBSYCL_GEN_MIXED_OPT, op)

  _LIBSYCL_GEN_OPT(+)
  _LIBSYCL_GEN_OPT(-)
  _LIBSYCL_GEN_OPT(*)
  _LIBSYCL_GEN_OPT(/)

#undef _LIBSYCL_GEN_OPT
#undef _LIBSYCL_GEN_MIXED_OPT

  // Compound assignment converts the other operand to half first, so that
  // double operands such as literals don't make device code depend on
  // aspect::fp64.
#define _LIBSYCL_GEN_MIXED_OPT(op_assign, T)                                   \
  constexpr half &operator op_assign(T rhs) {                                  \
    return *this op_assign fromNative(toNative(rhs));                          \
  }

#define _LIBSYCL_GEN_OPT(op, op_assign)                                        \
  constexpr half &operator op_assign(half rhs) {                               \
    return *this = *this op rhs;                                               \
  }                                                                            \
  _LIBSYCL_HALF_MIXED_TYPES(_LIBSYCL_GEN_MIXED_OPT, op_assign)

  _LIBSYCL_GEN_OPT(+, +=)
  _LIBSYCL_GEN_OPT(-, -=)
  _LIBSYCL_GEN_OPT(*, *=)
  _LIBSYCL_GEN_OPT(/, /=)

#undef _LIBSYCL_GEN_OPT
#undef _LIBSYCL_GEN_MIXED_OPT

  // Integers compare as float, so that values outside the half range don't
  // round to infinity.
#define _LIBSYCL_GEN_MIXED_OPT(op, T)                                          \
  friend constexpr bool operator op(half lhs, T rhs) {                         \
    using OpT = detail::HalfMixedT<T, float>;                                  \
    return static_cast<OpT>(lhs) op static_cast<OpT>(rhs);                     \
  }                                                                            \
  friend constexpr bool operator op(T lhs, half rhs) {                         \
    using OpT = detail::HalfMixedT<T, float>;                                  \
    return static_cast<OpT>(lhs) op static_cast<OpT>(rhs);                     \
  }

#define _LIBSYCL_GEN_OPT(op)                                                   \
  friend constexpr bool operator op(half lhs, half rhs) {                      \
    return _LIBSYCL_HALF_OP(lhs, op, rhs);                                     \
  }                                                                            \
  _LIBSYCL_HALF_MIXED_TYPES(_LIBSYCL_GEN_MIXED_OPT, op)

  _LIBSYCL_GEN_OPT(==)
  _LIBSYCL_GEN_OPT(!=)
  _LIBSYCL_GEN_OPT(<)
  _LIBSYCL_GEN_OPT(>)
  _LIBSYCL_GEN_OPT(<=)
  _LIBSYCL_GEN_OPT(>=)

#undef _LIBSYCL_GEN_OPT
#undef _LIBSYCL_GEN_MIXED_OPT
#undef _LIBSYCL_HALF_MIXED_TYPES
#undef _LIBSYCL_HALF_OP

  template <typename CharT, typename Traits>
  friend std::basic_ostream<CharT, Traits> &
  operator<<(std::basic_ostream<CharT, Traits> &lhs, half rhs) {
    return lhs << static_cast<float>(rhs);
  }

  template <typename CharT, typename Traits>
  friend std::basic_istream<CharT, Traits> &
  operator>>(std::basic_istream<CharT, Traits> &lhs, half &rhs) {
    float Value = 0.0f;
    lhs >> Value;
    rhs = half(Value);
    return lhs;
  }

private:
  // Wraps a _Float16 value without a round trip through float.
  static constexpr half fromNative(_Float16 Value) {
    half Result{};
    Result.MData = Value;
    return Result;
  }

  static constexpr float toFloat(_Float16 Value) {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<float>(Value);
#else
    return detail::halfBitsToFloat(__builtin_bit_cast(std::uint16_t, Value));
#endif
  }

  // Rounds Value to the nearest half value.
  template <typename T> static constexpr _Float16 toNative(T Value) {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<_Float16>(Value);
#else
    if constexpr (std::is_floating_point_v<T> && sizeof(T) > sizeof(float))
      return toNative(detail::roundToOddFloat(Value));
    else
      // Exact for integers below 2^24; larger ones round to infinity in half.
      return __builtin_bit_cast(
          _Float16, detail::floatToHalfBits(static_cast<float>(Value)));
#endif
  }

  friend std::numeric_limits<half>;

  _Float16 MData;
};

} // namespace half_impl
} // namespace detail

/// \copydoc detail::half_impl::half
using half = detail::half_impl::half;

static_assert(sizeof(half) == 2 && alignof(half) == 2);
static_assert(std::is_trivially_copyable_v<half> &&
              std::is_standard_layout_v<half>);

_LIBSYCL_END_NAMESPACE_SYCL

namespace std {
template <> struct numeric_limits<sycl::half> {
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr float_denorm_style has_denorm = denorm_present;
  static constexpr bool has_denorm_loss = false;
  static constexpr float_round_style round_style = round_to_nearest;
  static constexpr bool is_iec559 = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = __FLT16_MANT_DIG__;
  static constexpr int digits10 = __FLT16_DIG__;
  static constexpr int max_digits10 = __FLT16_DECIMAL_DIG__;
  static constexpr int radix = 2;
  static constexpr int min_exponent = __FLT16_MIN_EXP__;
  static constexpr int min_exponent10 = __FLT16_MIN_10_EXP__;
  static constexpr int max_exponent = __FLT16_MAX_EXP__;
  static constexpr int max_exponent10 = __FLT16_MAX_10_EXP__;
  static constexpr bool traps = false;
  static constexpr bool tinyness_before = false;

  // The values are built from _Float16 directly: a signaling NaN converted
  // from float would become quiet.
  static constexpr sycl::half(min)() noexcept {
    return sycl::half::fromNative(__FLT16_MIN__);
  }
  static constexpr sycl::half lowest() noexcept {
    return sycl::half::fromNative(-__FLT16_MAX__);
  }
  static constexpr sycl::half(max)() noexcept {
    return sycl::half::fromNative(__FLT16_MAX__);
  }
  static constexpr sycl::half epsilon() noexcept {
    return sycl::half::fromNative(__FLT16_EPSILON__);
  }
  static constexpr sycl::half round_error() noexcept {
    return sycl::half::fromNative(0.5F16);
  }
  static constexpr sycl::half infinity() noexcept {
    return sycl::half::fromNative(__builtin_inff16());
  }
  static constexpr sycl::half quiet_NaN() noexcept {
    return sycl::half::fromNative(__builtin_nanf16(""));
  }
  static constexpr sycl::half signaling_NaN() noexcept {
    return sycl::half::fromNative(__builtin_nansf16(""));
  }
  static constexpr sycl::half denorm_min() noexcept {
    return sycl::half::fromNative(__FLT16_DENORM_MIN__);
  }
};

template <> struct hash<sycl::half> {
  size_t operator()(sycl::half Key) const noexcept {
    // -0 and +0 compare equal, so they must hash equally.
    return hash<uint16_t>{}(Key == 0 ? 0 : __builtin_bit_cast(uint16_t, Key));
  }
};
} // namespace std

#endif // _LIBSYCL___IMPL_HALF_HPP
