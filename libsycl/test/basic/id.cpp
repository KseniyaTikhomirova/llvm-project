// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <cassert>
#include <iostream>
#include <sycl/sycl.hpp>

using sycl::detail::Builder;

using namespace std;
int main() {
  /* id()
   * Construct a SYCL id with the value 0 for each dimension. */
  sycl::id<1> OneDimZeroId;
  assert(OneDimZeroId.get(0) == 0);
  sycl::id<2> TwoDimZeroId;
  assert(TwoDimZeroId.get(0) == 0 && TwoDimZeroId.get(1) == 0);
  sycl::id<3> ThreeDimZeroId;
  assert(ThreeDimZeroId.get(0) == 0 && ThreeDimZeroId.get(1) == 0 &&
         ThreeDimZeroId.get(2) == 0);

  /* id(size_t dim0)
   * Construct a 1D id with value dim0. Only valid when the template parameter
   * dimensions is equal to 1 */
  sycl::id<1> OneDimId(64);
  assert(OneDimId.get(0) == 64);

  /* id(size_t dim0, size_t dim1)
   * Construct a 2D id with values dim0, dim1. Only valid when the template
   * parameter dimensions is equal to 2. */
  sycl::id<2> TwoDimId(128, 256);
  assert(TwoDimId.get(0) == 128 && TwoDimId.get(1) == 256);

  /* id(size_t dim0, size_t dim1, size_t dim2)
   * Construct a 3D id with values dim0, dim1, dim2. Only valid when the
   * template parameter dimensions is equal to 3. */
  sycl::id<3> ThreeDimId(64, 1, 2);
  assert(ThreeDimId.get(0) == 64 && ThreeDimId.get(1) == 1 &&
         ThreeDimId.get(2) == 2);

  /* id(const range<dimensions> &range)
   * Construct an id from the dimensions of r. */
  sycl::range<1> OneDimRange(2);
  sycl::id<1> OneDimIdRange(OneDimRange);
  assert(OneDimIdRange.get(0) == 2);
  sycl::range<2> TwoDimRange(4, 8);
  sycl::id<2> TwoDimIdRange(TwoDimRange);
  assert(TwoDimIdRange.get(0) == 4 && TwoDimIdRange.get(1) == 8);
  sycl::range<3> ThreeDimRange(16, 32, 64);
  sycl::id<3> ThreeDimIdRange(ThreeDimRange);
  assert(ThreeDimIdRange.get(0) == 16 && ThreeDimIdRange.get(1) == 32 &&
         ThreeDimIdRange.get(2) == 64);

  /* id(const item<dimensions> &item)
   * Construct an id from item.get_id().*/
  sycl::item<1, true> OneDimItemWithOffset =
      Builder::createItem<1, true>({4}, {2}, {1});
  sycl::id<1> OneDimIdItem(OneDimItemWithOffset);
  assert(OneDimIdItem.get(0) == 2);
  sycl::item<2, true> TwoDimItemWithOffset =
      Builder::createItem<2, true>({8, 16}, {4, 8}, {1, 1});
  sycl::id<2> TwoDimIdItem(TwoDimItemWithOffset);
  assert(TwoDimIdItem.get(0) == 4 && TwoDimIdItem.get(1) == 8);
  sycl::item<3, true> ThreeDimItemWithOffset =
      Builder::createItem<3, true>({32, 64, 128}, {16, 32, 64}, {1, 1, 1});
  sycl::id<3> ThreeDimIdItem(ThreeDimItemWithOffset);
  assert(ThreeDimIdItem.get(0) == 16 && ThreeDimIdItem.get(1) == 32 &&
         ThreeDimIdItem.get(2) == 64);
  /* size_t get(int dimension)const
   * Return the value of the id for dimension dimension. */

  /* size_t &operator[](int dimension)const
   * Return a reference to the requested dimension of the id object. */
  sycl::id<1> OneDimIdBrackets(64);
  assert(OneDimIdBrackets[0] == 64);
  sycl::id<2> TwoDimIdBrackets(128, 256);
  assert(TwoDimIdBrackets[0] == 128 && TwoDimIdBrackets[1] == 256);
  sycl::id<3> ThreeDimIdBrackets(64, 1, 2);
  assert(ThreeDimIdBrackets[0] == 64 && ThreeDimIdBrackets[1] == 1 &&
         ThreeDimIdBrackets[2] == 2);

  /* size_t &operator[](int dimension)const
   * Return a reference to the requested dimension of the id object. */

  /* bool operatorOP(const id<dimensions> &rhs) const
   * Where OP is: ==, !=.
   * Common by-value semantics.
   * T must be equality comparable on the host application and within SYCL
   * kernel functions. Equality between two instances of T (i.e. a == b) must be
   * true if the value of all members are equal and non-equality between two
   * instances of T (i.e. a != b) must be true if the value of any members are
   * not equal, unless either instance has become invalidated by a move
   * operation. Where T is id<dimensions>. */
  {
#define FirstOneValue 10
#define SecondOneValue 19
#define FirstTwoValue 15
#define SecondTwoValue 12
#define FirstThreeValue 3
#define SecondThreeValue 22

    sycl::id<1> OneDimOpOne(FirstOneValue);
    sycl::id<1> OneDimOpTwo(SecondOneValue);
    sycl::id<1> OneDimOpAnotherOne(FirstOneValue);

    sycl::id<2> TwoDimOpOne(FirstOneValue, FirstTwoValue);
    sycl::id<2> TwoDimOpTwo(SecondOneValue, SecondTwoValue);
    sycl::id<2> TwoDimOpAnotherOne(FirstOneValue, FirstTwoValue);

    sycl::id<3> ThreeDimOpOne(FirstOneValue, FirstTwoValue, FirstThreeValue);
    sycl::id<3> ThreeDimOpTwo(SecondOneValue, SecondTwoValue, SecondThreeValue);
    sycl::id<3> ThreeDimOpAnotherOne(FirstOneValue, FirstTwoValue,
                                     FirstThreeValue);

    // OP : ==
    // id<1> == id<1>
    assert((OneDimOpOne == OneDimOpTwo) == (FirstOneValue == SecondOneValue));
    assert((OneDimOpOne == OneDimOpAnotherOne) ==
           (FirstOneValue == FirstOneValue));
    // id<2> == id<2>
    assert((TwoDimOpOne == TwoDimOpTwo) == ((FirstOneValue == SecondOneValue) &&
                                            (FirstTwoValue == SecondTwoValue)));
    assert(
        (TwoDimOpOne == TwoDimOpAnotherOne) ==
        ((FirstOneValue == FirstOneValue) && (FirstTwoValue == FirstTwoValue)));
    // id<3> == id<3>
    assert((ThreeDimOpOne == ThreeDimOpTwo) ==
           ((FirstOneValue == SecondOneValue) &&
            (FirstTwoValue == SecondTwoValue) &&
            (FirstThreeValue == SecondThreeValue)));
    assert((ThreeDimOpOne == ThreeDimOpAnotherOne) ==
           ((FirstOneValue == FirstOneValue) &&
            (FirstTwoValue == FirstTwoValue) &&
            (FirstThreeValue == FirstThreeValue)));
    // id<1> == size_t
    assert((OneDimOpOne == SecondOneValue) ==
           (FirstOneValue == SecondOneValue));
    assert((OneDimOpOne == FirstOneValue) == (FirstOneValue == FirstOneValue));

    // size_t == id<1>
    assert((FirstOneValue == OneDimOpTwo) == (FirstOneValue == SecondOneValue));
    assert((FirstOneValue == OneDimOpAnotherOne) ==
           (FirstOneValue == FirstOneValue));

    // OP : !=
    // id<1> != id<1>
    assert((OneDimOpOne != OneDimOpTwo) == (FirstOneValue != SecondOneValue));
    assert((OneDimOpOne != OneDimOpAnotherOne) ==
           (FirstOneValue != FirstOneValue));
    // id<2> != id<2>
    assert(
        ((TwoDimOpOne != TwoDimOpTwo) == (FirstOneValue != SecondOneValue)) ||
        (FirstTwoValue != SecondTwoValue));
    assert(((TwoDimOpOne != TwoDimOpAnotherOne) ==
            (FirstOneValue != FirstOneValue)) ||
           (FirstTwoValue != FirstTwoValue));
    // id<3> != id<3>
    assert((ThreeDimOpOne != ThreeDimOpTwo) ==
           ((FirstOneValue != SecondOneValue) ||
            (FirstTwoValue != SecondTwoValue) ||
            (FirstThreeValue != SecondThreeValue)));
    assert((ThreeDimOpOne != ThreeDimOpAnotherOne) ==
           ((FirstOneValue != FirstOneValue) ||
            (FirstTwoValue != FirstTwoValue) ||
            (FirstThreeValue != FirstThreeValue)));
    // id<1> != size_t
    assert((OneDimOpOne != SecondOneValue) ==
           (FirstOneValue != SecondOneValue));
    assert((OneDimOpOne != FirstOneValue) == (FirstOneValue != FirstOneValue));

    // size_t != id<1>
    assert((FirstOneValue != OneDimOpTwo) == (FirstOneValue != SecondOneValue));
    assert((FirstOneValue != OneDimOpAnotherOne) ==
           (FirstOneValue != FirstOneValue));

#undef FirstOneValue
#undef SecondOneValue
#undef FirstTwoValue
#undef SecondTwoValue
#undef FirstThreeValue
#undef SecondThreeValue
  }

  /* id<dimensions> operatorOP(const id<dimensions> &rhs) const
   * Where OP is: +, -, *, /, %, <<, >>, &, |, ^, &&, ||, <, >, <=, >=.
   * Constructs and returns a new instance of the SYCL id class template with
   * the same dimensionality as this SYCL id, where each element of the new SYCL
   * id instance is the result of an element-wise OP operator between each
   * element of this SYCL id and each element of the rhs id. If the operator
   * returns a bool the result is the cast to size_t */
  {
    size_t Value1 = 10;
    size_t Value2 = 15;
    size_t Value3 = 3;

#define OneLeftValue Value1
#define OneRightValue 2
#define TwoLeftValue Value2
#define TwoRightValue 7
#define ThreeLeftValue Value3
#define ThreeRightValue 9

    sycl::id<1> OneDimOpLeft(OneLeftValue);
    sycl::id<1> OneDimOpRight(OneRightValue);
    sycl::range<1> OneDimOpRange(OneRightValue);

    sycl::id<2> TwoDimOpLeft(OneLeftValue, TwoLeftValue);
    sycl::id<2> TwoDimOpRight(OneRightValue, TwoRightValue);
    sycl::range<2> TwoDimOpRange(OneRightValue, TwoRightValue);

    sycl::id<3> ThreeDimOpLeft(OneLeftValue, TwoLeftValue, ThreeLeftValue);
    sycl::id<3> ThreeDimOpRight(OneRightValue, TwoRightValue, ThreeRightValue);
    sycl::range<3> ThreeDimOpRange(OneRightValue, TwoRightValue,
                                   ThreeRightValue);
#define OPERATOR_TEST(op)                                                      \
  assert((OneDimOpLeft op OneDimOpRight)[0] ==                                 \
         (OneLeftValue op OneRightValue));                                     \
  assert((OneDimOpRight op OneDimOpLeft)[0] ==                                 \
         (OneRightValue op OneLeftValue));                                     \
  assert((OneDimOpLeft op OneRightValue)[0] ==                                 \
         (OneLeftValue op OneRightValue));                                     \
  assert((OneLeftValue op OneDimOpRight)[0] ==                                 \
         (OneLeftValue op OneRightValue));                                     \
  assert(((TwoDimOpLeft op TwoDimOpRight)[0] ==                                \
          (OneLeftValue op OneRightValue)) &&                                  \
         ((TwoDimOpRight op TwoDimOpLeft)[1] ==                                \
          (TwoRightValue op TwoLeftValue)));                                   \
  assert(((TwoDimOpLeft op OneRightValue)[0] ==                                \
          (OneLeftValue op OneRightValue)) &&                                  \
         ((TwoLeftValue op TwoDimOpRight)[1] ==                                \
          (TwoLeftValue op TwoRightValue)));                                   \
  assert(((ThreeDimOpLeft op ThreeDimOpRight)[0] ==                            \
          (OneLeftValue op OneRightValue)) &&                                  \
         ((ThreeDimOpLeft op ThreeDimOpRight)[1] ==                            \
          (TwoLeftValue op TwoRightValue)) &&                                  \
         ((ThreeDimOpLeft op ThreeDimOpRight)[2] ==                            \
          (ThreeLeftValue op ThreeRightValue)));                               \
  assert(((ThreeDimOpLeft op OneRightValue)[0] ==                              \
          (OneLeftValue op OneRightValue)) &&                                  \
         ((TwoLeftValue op ThreeDimOpRight)[1] ==                              \
          (TwoLeftValue op TwoRightValue)) &&                                  \
         ((ThreeDimOpLeft op ThreeRightValue)[2] ==                            \
          (ThreeLeftValue op ThreeRightValue)));

    OPERATOR_TEST(+)
    OPERATOR_TEST(-)
    OPERATOR_TEST(*)
    OPERATOR_TEST(/)
    OPERATOR_TEST(%)
    OPERATOR_TEST(<<)
    OPERATOR_TEST(>>)
    OPERATOR_TEST(&)
    OPERATOR_TEST(|)
    OPERATOR_TEST(^)
    OPERATOR_TEST(&&)
    OPERATOR_TEST(||)
    OPERATOR_TEST(<)
    OPERATOR_TEST(>)
    OPERATOR_TEST(<=)
    OPERATOR_TEST(>=)

#undef OPERATOR_TEST
#undef OPERATOR_TEST_BASIC

#undef OneLeftValue
#undef OneRightValue
#undef TwoLeftValue
#undef TwoRightValue
#undef ThreeLeftValue
#undef ThreeRightValue
  }

  /* id<dimensions> operatorOP(const id<dimensions> &rhs) const
   * Where OP is: +, -, *, /, %, <<, >>, &, |, ^.
   * Assigns each element of this SYCL id instance with the result of an
   * element-wise OP operator between each element of this SYCL id and each
   * element of the rhs id and returns a reference to this SYCL id. If the
   * operator returns a bool the result is the cast to size_t */
  {
    size_t Value1 = 10;
    size_t Value2 = 15;
    size_t Value3 = 3;

#define OneLeftValue Value1
#define OneRightValue 2
#define TwoLeftValue Value2
#define TwoRightValue 7
#define ThreeLeftValue Value3
#define ThreeRightValue 9

    sycl::id<1> OneDimOpLeft(OneLeftValue);
    sycl::id<1> OneDimOpRight(OneRightValue);
    sycl::range<1> OneDimOpRange(OneRightValue);

    sycl::id<2> TwoDimOpLeft(OneLeftValue, TwoLeftValue);
    sycl::id<2> TwoDimOpRight(OneRightValue, TwoRightValue);
    sycl::range<2> TwoDimOpRange(OneRightValue, TwoRightValue);

    sycl::id<3> ThreeDimOpLeft(OneLeftValue, TwoLeftValue, ThreeLeftValue);
    sycl::id<3> ThreeDimOpRight(OneRightValue, TwoRightValue, ThreeRightValue);
    sycl::range<3> ThreeDimOpRange(OneRightValue, TwoRightValue,
                                   ThreeRightValue);

#define OPERATOR_TEST(op)                                                      \
  OneDimOpLeft[0] = OneLeftValue;                                              \
  OneDimOpRight[0] = OneRightValue;                                            \
  assert((OneDimOpLeft op## = OneDimOpRight)[0] ==                             \
         (OneLeftValue op OneRightValue));                                     \
  OneDimOpLeft[0] = OneLeftValue;                                              \
  assert((OneDimOpLeft op## = OneRightValue)[0] ==                             \
         (OneLeftValue op OneRightValue));                                     \
  TwoDimOpLeft[0] = OneLeftValue;                                              \
  TwoDimOpLeft[1] = TwoLeftValue;                                              \
  TwoDimOpRight[0] = OneRightValue;                                            \
  TwoDimOpRight[1] = TwoRightValue;                                            \
  assert(((TwoDimOpLeft op## = TwoDimOpRight)[0] ==                            \
          (OneLeftValue op OneRightValue)) &&                                  \
         (TwoDimOpLeft[1] == (TwoLeftValue op TwoRightValue)));                \
  TwoDimOpLeft[0] = OneLeftValue;                                              \
  TwoDimOpLeft[1] = TwoLeftValue;                                              \
  assert(((TwoDimOpLeft op## = OneRightValue)[0] ==                            \
          (OneLeftValue op OneRightValue)) &&                                  \
         (TwoDimOpLeft[1] == (TwoLeftValue op OneRightValue)));                \
  ThreeDimOpLeft[0] = OneLeftValue;                                            \
  ThreeDimOpLeft[1] = TwoLeftValue;                                            \
  ThreeDimOpLeft[2] = ThreeLeftValue;                                          \
  ThreeDimOpRight[0] = OneRightValue;                                          \
  ThreeDimOpRight[1] = TwoRightValue;                                          \
  ThreeDimOpRight[2] = ThreeRightValue;                                        \
  assert(((ThreeDimOpLeft op## = ThreeDimOpRight)[0] ==                        \
          (OneLeftValue op OneRightValue)) &&                                  \
         (ThreeDimOpLeft[1] == (TwoLeftValue op TwoRightValue)) &&             \
         (ThreeDimOpLeft[2] == (ThreeLeftValue op ThreeRightValue)));          \
  ThreeDimOpLeft[0] = OneLeftValue;                                            \
  ThreeDimOpLeft[1] = TwoLeftValue;                                            \
  ThreeDimOpLeft[2] = ThreeLeftValue;                                          \
  assert(((ThreeDimOpLeft op## = OneRightValue)[0] ==                          \
          (OneLeftValue op OneRightValue)) &&                                  \
         (ThreeDimOpLeft[1] == (TwoLeftValue op OneRightValue)) &&             \
         (ThreeDimOpLeft[2] == (ThreeLeftValue op OneRightValue)));            \
  OneDimOpLeft[0] = OneLeftValue;                                              \
  OneDimOpRange[0] = OneRightValue;                                            \
  assert((OneDimOpLeft op## = OneDimOpRange)[0] ==                             \
         (OneLeftValue op OneRightValue));                                     \
  TwoDimOpLeft[0] = OneLeftValue;                                              \
  TwoDimOpLeft[1] = TwoLeftValue;                                              \
  TwoDimOpRange[0] = OneRightValue;                                            \
  TwoDimOpRange[1] = TwoRightValue;                                            \
  assert(((TwoDimOpLeft op## = TwoDimOpRange)[0] ==                            \
          (OneLeftValue op OneRightValue)) &&                                  \
         (TwoDimOpLeft[1] == (TwoLeftValue op TwoRightValue)));                \
  ThreeDimOpLeft[0] = OneLeftValue;                                            \
  ThreeDimOpLeft[1] = TwoLeftValue;                                            \
  ThreeDimOpLeft[2] = ThreeLeftValue;                                          \
  ThreeDimOpRange[0] = OneRightValue;                                          \
  ThreeDimOpRange[1] = TwoRightValue;                                          \
  ThreeDimOpRange[2] = ThreeRightValue;                                        \
  assert(((ThreeDimOpLeft op## = ThreeDimOpRange)[0] ==                        \
          (OneLeftValue op OneRightValue)) &&                                  \
         (ThreeDimOpLeft[1] == (TwoLeftValue op TwoRightValue)) &&             \
         (ThreeDimOpLeft[2] == (ThreeLeftValue op ThreeRightValue)));

    OPERATOR_TEST(+)
    OPERATOR_TEST(-)
    OPERATOR_TEST(*)
    OPERATOR_TEST(/)
    OPERATOR_TEST(%)
    OPERATOR_TEST(<<)
    OPERATOR_TEST(>>)
    OPERATOR_TEST(&)
    OPERATOR_TEST(|)
    OPERATOR_TEST(^)

#undef OPERATOR_TEST
#undef OPERATOR_TEST_BASIC

#undef OneLeftValue
#undef OneRightValue
#undef TwoLeftValue
#undef TwoRightValue
#undef ThreeLeftValue
#undef ThreeRightValue
  }

/* operator size_t() const
 * Test implicit cast from id<1> to size_t and int value
 * Should fails on cast from id<2> and id<3> */
#define NumValue 16
  {
    sycl::id<1> OneDimIdCastToNum(NumValue);
    size_t Number1 = OneDimIdCastToNum;
    int Number2 = OneDimIdCastToNum;
    size_t Number3 = (size_t)OneDimIdCastToNum;
    int Number4 = (int)OneDimIdCastToNum;
    size_t Number5 = (int)OneDimIdCastToNum;
    assert((Number1 == NumValue) && (Number2 == NumValue) &&
           (Number3 == NumValue) && (Number4 == NumValue) &&
           (Number5 == NumValue));
  }

#undef NumValue

  {
    sycl::id<1> OneDimId(64);
    sycl::id<1> OneDimIdNeg(-64);
    sycl::id<1> OneDimIdCopy(64);
    sycl::id<2> TwoDimId(64, 1);
    sycl::id<2> TwoDimIdNeg(-64, -1);
    sycl::id<2> TwoDimIdCopy(64, 1);
    sycl::id<3> ThreeDimId(64, 1, 2);
    sycl::id<3> ThreeDimIdNeg(-64, -1, -2);
    sycl::id<3> ThreeDimIdCopy(64, 1, 2);

    assert((+OneDimId) == OneDimId);
    assert(-OneDimId == OneDimIdNeg);
    assert((+TwoDimId) == TwoDimId);
    assert(-TwoDimId == TwoDimIdNeg);
    assert((+ThreeDimId) == ThreeDimId);
    assert(-ThreeDimId == ThreeDimIdNeg);

    assert((++OneDimId) == (OneDimIdCopy + 1));
    assert((--OneDimId) == (OneDimIdCopy));
    assert((++TwoDimId) == (TwoDimIdCopy + 1));
    assert((--TwoDimId) == (TwoDimIdCopy));
    assert((++ThreeDimId) == (ThreeDimIdCopy + 1));
    assert((--ThreeDimId) == (ThreeDimIdCopy));

    assert((OneDimId++) == (OneDimIdCopy));
    assert((OneDimId--) == (OneDimIdCopy + 1));
    assert((TwoDimId++) == (TwoDimIdCopy));
    assert((TwoDimId--) == (TwoDimIdCopy + 1));
    assert((ThreeDimId++) == (ThreeDimIdCopy));
    assert((ThreeDimId--) == (ThreeDimIdCopy + 1));
  }
}
