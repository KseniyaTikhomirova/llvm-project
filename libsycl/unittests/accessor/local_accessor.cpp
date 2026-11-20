//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <common/scoped_binary_registration.hpp>
#include <mock/helpers.hpp>

#include <sycl/__impl/accessor.hpp>
#include <sycl/__impl/handler.hpp>
#include <sycl/__impl/local_accessor.hpp>
#include <sycl/__impl/properties/queue_properties.hpp>
#include <sycl/__impl/queue.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <functional>

using namespace sycl;
using namespace ::testing;

class sycl::detail::MockQueue : public sycl::queue {
public:
  using sycl::queue::sycl_kernel_launch;
};

namespace {

struct alignas(16) Aligned16 {
  float Data[4];
};

} // namespace

TEST(LocalAccessor, DefaultConstructedIsEmpty) {
  local_accessor<int, 1> Acc;
  EXPECT_TRUE(Acc.empty());
  EXPECT_EQ(Acc.size(), 0u);
  EXPECT_EQ(Acc.byte_size(), 0u);
  EXPECT_EQ(Acc.max_size(), 0u);
  EXPECT_EQ(Acc.get_range(), range<1>(0));
}

TEST(LocalAccessor, SizeQueries) {
  mock::MockWrapper Mock;
  queue Q;
  local_accessor<int, 2> Acc2;
  local_accessor<double, 0> Acc0;
  local_accessor<char, 1> ZeroSized;
  Q.submit([&](handler &CGH) {
    Acc2 = local_accessor<int, 2>(range<2>{2, 16}, CGH);
    Acc0 = local_accessor<double, 0>(CGH);
    ZeroSized = local_accessor<char, 1>(range<1>{0}, CGH);
  });

  EXPECT_FALSE(Acc2.empty());
  EXPECT_EQ(Acc2.size(), 32u);
  EXPECT_EQ(Acc2.byte_size(), 32 * sizeof(int));
  EXPECT_EQ(Acc2.get_range(), (range<2>{2, 16}));
  EXPECT_GT(Acc2.max_size(), 0u);

  EXPECT_FALSE(Acc0.empty());
  EXPECT_EQ(Acc0.size(), 1u);
  EXPECT_EQ(Acc0.byte_size(), sizeof(double));

  EXPECT_TRUE(ZeroSized.empty());
  EXPECT_EQ(ZeroSized.size(), 0u);
  EXPECT_EQ(ZeroSized.max_size(), 0u);
}

TEST(LocalAccessor, CopiesAreEqualAndHashEqual) {
  mock::MockWrapper Mock;
  queue Q;
  local_accessor<int, 1> Acc1, Acc3;
  Q.submit([&](handler &CGH) {
    Acc1 = local_accessor<int, 1>(1, CGH);
    Acc3 = local_accessor<int, 1>(1, CGH);
  });
  local_accessor<int, 1> Acc2 = Acc1;

  using HashT = std::hash<local_accessor<int, 1>>;
  EXPECT_EQ(Acc1, Acc2);
  EXPECT_EQ(HashT{}(Acc1), HashT{}(Acc2));
  EXPECT_NE(Acc1, Acc3);
  EXPECT_NE(HashT{}(Acc1), HashT{}(Acc3));
}

TEST(LocalAccessor, DefaultConstructedAreEqual) {
  local_accessor<int, 1> Acc1, Acc2;
  EXPECT_EQ(Acc1, Acc2);
}

TEST(LocalAccessor, ConvertsToConstDataT) {
  mock::MockWrapper Mock;
  queue Q;
  local_accessor<int, 1> Default;
  EXPECT_EQ((local_accessor<const int, 1>(Default)),
            (local_accessor<const int, 1>(Default)));

  local_accessor<int, 1> Acc;
  Q.submit([&](handler &CGH) { Acc = local_accessor<int, 1>(4, CGH); });
  local_accessor<const int, 1> ConstAcc1(Acc);
  local_accessor<const int, 1> ConstAcc2 = Acc;
  EXPECT_EQ(ConstAcc1, ConstAcc2);
  EXPECT_EQ(ConstAcc1.size(), 4u);
  EXPECT_EQ((std::hash<local_accessor<const int, 1>>{}(ConstAcc1)),
            (std::hash<local_accessor<int, 1>>{}(Acc)));
}

TEST(LocalAccessor, Swap) {
  mock::MockWrapper Mock;
  queue Q;
  local_accessor<int, 1> Acc1, Acc2;
  Q.submit([&](handler &CGH) {
    Acc1 = local_accessor<int, 1>(8, CGH);
    Acc2 = local_accessor<int, 1>(16, CGH);
  });
  const local_accessor<int, 1> Orig1 = Acc1;
  const local_accessor<int, 1> Orig2 = Acc2;

  Acc1.swap(Acc2);
  EXPECT_EQ(Acc1.size(), 16u);
  EXPECT_EQ(Acc2.size(), 8u);
  EXPECT_EQ(Acc1, Orig2);
  EXPECT_EQ(Acc2, Orig1);
}

TEST(LocalAccessor, HasNoProperties) {
  mock::MockWrapper Mock;
  queue Q;
  local_accessor<int, 1> Acc;
  Q.submit([&](handler &CGH) { Acc = local_accessor<int, 1>(1, CGH); });
  EXPECT_FALSE(Acc.has_property<property::queue::in_order>());
  EXPECT_THROW(Acc.get_property<property::queue::in_order>(), exception);
}

TEST(LocalAccessor, RejectsPropertyOfAnotherClass) {
  mock::MockWrapper Mock;
  queue Q;
  Q.submit([&](handler &CGH) {
    try {
      local_accessor<int, 1> Acc(1, CGH, {property::queue::in_order{}});
      FAIL() << "a queue property must be rejected by a local accessor";
    } catch (const exception &E) {
      EXPECT_EQ(E.code(), make_error_code(errc::invalid));
    }
  });
}

TEST(DeprecatedLocalAccessor, SizeQueriesAndEquality) {
  using AccT = accessor<int, 2, access_mode::read_write, target::local>;
  mock::MockWrapper Mock;
  queue Q;
  AccT Acc1, Acc3;
  Q.submit([&](handler &CGH) {
    Acc1 = AccT(range<2>{2, 3}, CGH);
    Acc3 = AccT(range<2>{2, 3}, CGH);
  });
  AccT Acc2 = Acc1;

  EXPECT_EQ(Acc1.get_count(), 6u);
  EXPECT_EQ(Acc1.get_size(), 6 * sizeof(int));
  EXPECT_EQ(Acc1.size(), 6u);
  EXPECT_EQ(Acc1.get_range(), (range<2>{2, 3}));
  EXPECT_EQ(Acc1, Acc2);
  EXPECT_NE(Acc1, Acc3);
  EXPECT_EQ(std::hash<AccT>{}(Acc1), std::hash<AccT>{}(Acc2));

  AccT Default;
  EXPECT_TRUE(Default.empty());
  EXPECT_EQ(Default.get_count(), 0u);
}

// The tests below check the size of the work-group local memory block that is
// requested at launch for the local accessors of a kernel. The device lays out
// the accessors in the same way, so the block size also reflects the alignment
// of each accessor's region.

struct LocalMemKernel;
struct LocalMemKernelInfo {
  using kernel_name = LocalMemKernel;
  static constexpr char kernel_entry_point_name[] = "LocalMemKernel";
};

struct KernelData {
  int Value;
};

template <detail::KernelLaunchKind Kind>
using KernelObjectT = detail::KernelObject<KernelData, Kind>;

// Launches LocalMemKernel as an nd_range kernel with Accs as the local
// accessors of the kernel object and returns the local memory size passed to
// olLaunchKernel.
template <typename... AccTs>
static std::size_t launchAndCaptureLocalMemSize(mock::MockWrapper &Mock,
                                                detail::MockQueue &Q,
                                                const AccTs &...Accs) {
  unittests::ScopedKernelRegistration Registration{"LocalMemKernel"};
  std::size_t Captured = 0;
  EXPECT_CALL(Mock.get(), olLaunchKernel(_, _, _, _, _, 1, _, _))
      .WillOnce([&Captured](ol_queue_handle_t, ol_device_handle_t,
                            ol_symbol_handle_t,
                            const ol_kernel_launch_size_args_t *Args,
                            const ol_kernel_launch_prop_t *, size_t, void **,
                            const size_t *) -> ol_result_t {
        Captured = Args->DynSharedMemory;
        return OL_SUCCESS;
      });
  KernelObjectT<detail::KernelLaunchKind::NDRange> Kernel{};
  Q.sycl_kernel_launch<LocalMemKernelInfo>(Kernel)(Accs...);
  return Captured;
}

TEST(LocalAccessorLaunch, NoLocalAccessors) {
  mock::MockWrapper Mock;
  detail::MockQueue Q;
  EXPECT_EQ(launchAndCaptureLocalMemSize(Mock, Q), 0u);
}

TEST(LocalAccessorLaunch, SingleAccessor) {
  mock::MockWrapper Mock;
  detail::MockQueue Q;
  local_accessor<int, 3> Acc;
  Q.submit([&](handler &CGH) {
    Acc = local_accessor<int, 3>(range<3>{2, 3, 4}, CGH);
  });
  EXPECT_EQ(launchAndCaptureLocalMemSize(Mock, Q, Acc), 24 * sizeof(int));
}

// Regions are aligned to their element type, so an int followed by a 16-byte
// aligned type leaves padding in between.
TEST(LocalAccessorLaunch, RegionsAreAligned) {
  mock::MockWrapper Mock;
  detail::MockQueue Q;
  local_accessor<int, 1> IntAcc;
  local_accessor<Aligned16, 1> AlignedAcc;
  Q.submit([&](handler &CGH) {
    IntAcc = local_accessor<int, 1>(1, CGH);
    AlignedAcc = local_accessor<Aligned16, 1>(1, CGH);
  });
  EXPECT_EQ(launchAndCaptureLocalMemSize(Mock, Q, IntAcc, AlignedAcc),
            2 * sizeof(Aligned16));
  EXPECT_EQ(launchAndCaptureLocalMemSize(Mock, Q, AlignedAcc, IntAcc),
            sizeof(Aligned16) + sizeof(int));
}

// Default constructed and zero-sized accessors take no local memory, but their
// regions are still aligned so that the device never forms a misaligned
// pointer.
TEST(LocalAccessorLaunch, EmptyAccessorsOnlyAddAlignment) {
  mock::MockWrapper Mock;
  detail::MockQueue Q;
  local_accessor<int, 1> Default;
  local_accessor<char, 1> ZeroSized, Byte;
  Q.submit([&](handler &CGH) {
    ZeroSized = local_accessor<char, 1>(range<1>{0}, CGH);
    Byte = local_accessor<char, 1>(1, CGH);
  });
  EXPECT_EQ(launchAndCaptureLocalMemSize(Mock, Q, Default, ZeroSized), 0u);
  EXPECT_EQ(launchAndCaptureLocalMemSize(Mock, Q, ZeroSized, Byte), 1u);
  EXPECT_EQ(launchAndCaptureLocalMemSize(Mock, Q, Byte, Default, ZeroSized),
            alignof(int));
}

TEST(LocalAccessorLaunch, DeprecatedAccessorTakesMemory) {
  mock::MockWrapper Mock;
  detail::MockQueue Q;
  using DeprecatedAccT =
      accessor<int, 1, access_mode::read_write, target::local>;
  DeprecatedAccT Deprecated;
  local_accessor<char, 0> Scalar;
  Q.submit([&](handler &CGH) {
    Scalar = local_accessor<char, 0>(CGH);
    Deprecated = DeprecatedAccT(3, CGH);
  });
  EXPECT_EQ(launchAndCaptureLocalMemSize(Mock, Q, Scalar, Deprecated),
            alignof(int) + 3 * sizeof(int));
}

// SYCL 2020 4.7.6.11: a local accessor must not be used in a kernel invoked
// with single_task or with a parallel_for taking a range.
template <detail::KernelLaunchKind Kind>
static void expectKernelArgumentError() {
  mock::MockWrapper Mock;
  detail::MockQueue Q;
  unittests::ScopedKernelRegistration Registration{"LocalMemKernel"};
  local_accessor<int, 1> Default;
  local_accessor<int, 1> Acc;
  Q.submit([&](handler &CGH) { Acc = local_accessor<int, 1>(1, CGH); });

  EXPECT_CALL(Mock.get(), olLaunchKernel(_, _, _, _, _, _, _, _)).Times(0);
  KernelObjectT<Kind> Kernel{};
  for (const local_accessor<int, 1> &Used : {Acc, Default}) {
    try {
      Q.sycl_kernel_launch<LocalMemKernelInfo>(Kernel)(Used);
      ADD_FAILURE() << "the launch must throw";
    } catch (const exception &E) {
      EXPECT_EQ(E.code(), make_error_code(errc::kernel_argument));
    }
  }
}

TEST(LocalAccessorLaunch, SingleTaskRejectsLocalAccessor) {
  expectKernelArgumentError<detail::KernelLaunchKind::SingleTask>();
}

TEST(LocalAccessorLaunch, RangeRejectsLocalAccessor) {
  expectKernelArgumentError<detail::KernelLaunchKind::Range>();
}

TEST(LocalAccessorLaunch, KernelWithoutLocalAccessorIsAccepted) {
  mock::MockWrapper Mock;
  detail::MockQueue Q;
  unittests::ScopedKernelRegistration Registration{"LocalMemKernel"};
  EXPECT_CALL(Mock.get(), olLaunchKernel(_, _, _, _, _, 1, _, _)).Times(1);
  KernelObjectT<detail::KernelLaunchKind::SingleTask> Kernel{};
  EXPECT_NO_THROW(Q.sycl_kernel_launch<LocalMemKernelInfo>(Kernel)());
}
