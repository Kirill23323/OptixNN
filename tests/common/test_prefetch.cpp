#include <gtest/gtest.h>
#include <vector>
#include "OptixNNlib/prefetch.h"

TEST(Prefetch, PrefetchL1) {
    int data = 42;
    EXPECT_NO_THROW(PrefetchL1(&data));
    
    std::vector<int> vec = {1, 2, 3};
    EXPECT_NO_THROW(PrefetchL1(vec.data()));
}

TEST(Prefetch, PrefetchL2) {
    double data = 3.14;
    EXPECT_NO_THROW(PrefetchL2(&data));
    
    std::vector<double> vec = {1.0, 2.0, 3.0};
    EXPECT_NO_THROW(PrefetchL2(vec.data()));
}

TEST(Prefetch, PrefetchL3) {
    char data = 'a';
    EXPECT_NO_THROW(PrefetchL3(&data));
    
    std::string str = "hello";
    EXPECT_NO_THROW(PrefetchL3(str.data()));
}

TEST(Prefetch, PrefetchNullptr) {
    // Prefetch on nullptr should not crash, as it's just a hint
    EXPECT_NO_THROW(PrefetchL1(nullptr));
    EXPECT_NO_THROW(PrefetchL2(nullptr));
    EXPECT_NO_THROW(PrefetchL3(nullptr));
}

TEST(Prefetch, PrefetchInvalidAddress) {
    // Even invalid addresses should not crash since prefetch is speculative
    void* invalid = reinterpret_cast<void*>(0xDEADBEEF);
    EXPECT_NO_THROW(PrefetchL1(invalid));
    EXPECT_NO_THROW(PrefetchL2(invalid));
    EXPECT_NO_THROW(PrefetchL3(invalid));
}