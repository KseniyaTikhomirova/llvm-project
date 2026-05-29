// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <iostream>

using namespace sycl;

template <int Dimensions>
bool CheckRange(const range<Dimensions> &R,
                const std::array<std::size_t, Dimensions> &Expected,
                const char *Name) {
  for (int I = 0; I < Dimensions; ++I) {
    if (R[I] != Expected[I]) {
      std::cout << Name << " mismatch at dim " << I << ": got " << R[I]
                << " expected " << Expected[I] << std::endl;
      return false;
    }
  }
  return true;
}

int main() {
  bool Fail = false;

  sycl::range<1> OneDimRange(64);
  sycl::range<2> TwoDimRange(64, 1);
  sycl::range<3> ThreeDimRange(64, 1, 2);
  assert(OneDimRange.size() == 64);
  assert(OneDimRange.get(0) == 64);
  assert(OneDimRange[0] == 64);
  std::cout << "OneDimRange passed " << std::endl;
  assert(TwoDimRange.size() == 64);
  assert(TwoDimRange.get(0) == 64);
  assert(TwoDimRange[0] == 64);
  assert(TwoDimRange.get(1) == 1);
  assert(TwoDimRange[1] == 1);
  std::cout << "TwoDimRange passed " << std::endl;
  assert(ThreeDimRange.size() == 128);
  assert(ThreeDimRange.get(0) == 64);
  assert(ThreeDimRange[0] == 64);
  assert(ThreeDimRange.get(1) == 1);
  assert(ThreeDimRange[1] == 1);
  assert(ThreeDimRange.get(2) == 2);
  assert(ThreeDimRange[2] == 2);
  std::cout << "ThreeDimRange passed " << std::endl;

  sycl::range<1> OneDimRangeNeg(-64);
  sycl::range<1> OneDimRangeCopy(64);
  sycl::range<2> TwoDimRangeNeg(-64, -1);
  sycl::range<2> TwoDimRangeCopy(64, 1);
  sycl::range<3> ThreeDimRangeCopy(64, 1, 2);
  sycl::range<3> ThreeDimRangeNeg(-64, -1, -2);

  assert((+OneDimRange) == OneDimRange);
  assert(-OneDimRange == OneDimRangeNeg);
  assert((+TwoDimRange) == TwoDimRange);
  assert(-TwoDimRange == TwoDimRangeNeg);
  assert((+ThreeDimRange) == ThreeDimRange);
  assert(-ThreeDimRange == ThreeDimRangeNeg);

  assert((++OneDimRange) == (OneDimRangeCopy + 1));
  assert((--OneDimRange) == OneDimRangeCopy);
  assert((++TwoDimRange) == (TwoDimRangeCopy + 1));
  assert((--TwoDimRange) == TwoDimRangeCopy);
  assert((++ThreeDimRange) == (ThreeDimRangeCopy + 1));
  assert((--ThreeDimRange) == ThreeDimRangeCopy);

  assert((OneDimRange++) == OneDimRangeCopy);
  assert((OneDimRange--) == (OneDimRangeCopy + 1));
  assert((TwoDimRange++) == TwoDimRangeCopy);
  assert((TwoDimRange--) == (TwoDimRangeCopy + 1));
  assert((ThreeDimRange++) == ThreeDimRangeCopy);
  assert((ThreeDimRange--) == (ThreeDimRangeCopy + 1));

  sycl::range<3> Default3;
  sycl::range<2> Default2;
  sycl::range<1> Default1;

  assert(Default3[0] == 0 && Default3[1] == 0 && Default3[2] == 0);
  assert(Default2[0] == 0 && Default2[1] == 0);
  assert(Default1[0] == 0);

  const range<3> A(8, 9, 10);
  const range<3> B(2, 3, 5);

  Fail |= !CheckRange(A + B, {10, 12, 15}, "range+range");
  Fail |= !CheckRange(A - B, {6, 6, 5}, "range-range");
  Fail |= !CheckRange(A * B, {16, 27, 50}, "range*range");
  Fail |= !CheckRange(A / B, {4, 3, 2}, "range/range");
  Fail |= !CheckRange(A % B, {0, 0, 0}, "range%range");

  Fail |= !CheckRange(A + 1, {9, 10, 11}, "range+scalar");
  Fail |= !CheckRange(1 + A, {9, 10, 11}, "scalar+range");
  Fail |= !CheckRange(A - 1, {7, 8, 9}, "range-scalar");
  Fail |= !CheckRange(16 / B, {8, 5, 3}, "scalar/range");

  Fail |= !CheckRange(A << 1, {16, 18, 20}, "range<<scalar");
  Fail |= !CheckRange(A >> 1, {4, 4, 5}, "range>>scalar");
  Fail |= !CheckRange(A & 6, {0, 0, 2}, "range&scalar");
  Fail |= !CheckRange(A | 1, {9, 9, 11}, "range|scalar");
  Fail |= !CheckRange(A ^ 3, {11, 10, 9}, "range^scalar");

  Fail |= !CheckRange(A > B, {1, 1, 1}, "range>range");
  Fail |= !CheckRange(A < B, {0, 0, 0}, "range<range");

  Fail |= !CheckRange(A >= B, {1, 1, 1}, "range>=range");
  Fail |= !CheckRange(A <= B, {0, 0, 0}, "range<=range");

  const range<3> C(1, 0, 7);
  const range<3> D(1, 3, 0);
  Fail |= !CheckRange(C && D, {1, 0, 0}, "range&&range");
  Fail |= !CheckRange(C || D, {1, 1, 1}, "range||range");

  range<3> E = A;
  E += B;
  Fail |= !CheckRange(E, {10, 12, 15}, "range+=range");
  E -= B;
  Fail |= !CheckRange(E, {8, 9, 10}, "range-=range");
  E *= B;
  Fail |= !CheckRange(E, {16, 27, 50}, "range*=range");
  E /= B;
  Fail |= !CheckRange(E, {8, 9, 10}, "range/=range");

  E %= std::size_t{7};
  Fail |= !CheckRange(E, {1, 2, 3}, "range%=scalar");
  E <<= std::size_t{1};
  Fail |= !CheckRange(E, {2, 4, 6}, "range<<=scalar");
  E >>= std::size_t{1};
  Fail |= !CheckRange(E, {1, 2, 3}, "range>>=scalar");
  E |= std::size_t{8};
  Fail |= !CheckRange(E, {9, 10, 11}, "range|=scalar");
  E &= std::size_t{14};
  Fail |= !CheckRange(E, {8, 10, 10}, "range&=scalar");
  E ^= std::size_t{3};
  Fail |= !CheckRange(E, {11, 9, 9}, "range^=scalar");

  return Fail;
}
