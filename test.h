#pragma once

#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include "utility.h"

namespace cz {

template <class T>
void __EXPECT_EQ(bool is_ok, T actual, T expect) {
  if (!is_ok) {
    std::cout << "Expect: " << std::fixed << std::setprecision(10) << expect
              << std::endl;
    std::cout << "Actual: " << std::fixed << std::setprecision(10) << actual
              << std::endl;
    assert(false);
  }
}

template <class T>
void __EXPECT_NE(bool is_ok, T actual, T expect) {
  if (!is_ok) {
    std::cout << "Expect not: " << std::fixed << std::setprecision(10) << expect
              << std::endl;
    std::cout << "Actual: " << std::fixed << std::setprecision(10) << actual
              << std::endl;
    assert(false);
  }
}

void EXPECT_EQ(const std::string& e, const char* a) {
  std::string sa(a);
  __EXPECT_EQ(IsEq(e, sa), e, sa);
}

template <class T>
void EXPECT_EQ(T e, T a) {
  __EXPECT_EQ(IsEq(e, a), e, a);
}

template <class T>
void EXPECT_NE(T e, T a) {
  __EXPECT_NE(!IsEq(e, a), e, a);
}

template <class T>
void EXPECT_LE(T a, T b) {
  if (!(a <= b)) {
    std::cout << "Expect: " << a << " <= " << b << std::endl;
    assert(false);
  }
}

#define RUN_TEST(test_name)           \
  printf("Testing %s\n", #test_name); \
  test_name();                        \
  printf("Passed %s\n", #test_name);

}  // namespace cz
