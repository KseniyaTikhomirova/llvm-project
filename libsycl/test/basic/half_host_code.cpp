//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsycl-host-only -S -emit-llvm %s -o - \
// RUN:   | FileCheck %s \
// RUN:     --implicit-check-not="{{(fadd|fsub|fmul|fdiv|fneg|fpext|fcmp [a-z]+) half}}" \
// RUN:     --implicit-check-not="{{(fptrunc|[su]itofp) .* to half}}"

// Checks that host code doesn't use floating-point operations on half, which
// x86 without F16C lowers to compiler-rt calls (__extendhfsf2, __truncsfhf2).

#include <sycl/sycl.hpp>

using sycl::half;

// CHECK: define {{.*}}test
float test(half A, half B, float F, double D, long double LD, int I,
           unsigned long long U) {
  half H = F;
  H += A;
  H -= D;
  H *= LD;
  H /= I;
  H += U;
  ++H;
  H--;
  H = -H;
  H = A + B - A * B / H;
  H = H + I;
  bool C = A < B || A == I || A > F || A <= D;
  return H + F + static_cast<float>(D + H) + static_cast<float>(LD * H) + C;
}
