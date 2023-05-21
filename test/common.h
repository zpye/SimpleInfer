#ifndef SIMPLE_INFER_TEST_TEST_COMMON_H_
#define SIMPLE_INFER_TEST_TEST_COMMON_H_

#include <catch2/catch.hpp>

#include <cmath>

#define CHECK_EQ(a, b)                CHECK((a) == (b))
#define CHECK_LT(a, b)                CHECK((a) < (b))
#define CHECK_FLOAT_EPS_EQ(a, b, eps) CHECK_LT(std::abs((a) - (b)), eps)
#define CHECK_FLOAT_EQ(a, b)          CHECK_FLOAT_EPS_EQ(a, b, 1e-6)

#endif  // SIMPLE_INFER_TEST_TEST_COMMON_H_
