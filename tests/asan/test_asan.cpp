#include <gtest/gtest.h>

// Самый простой тест
TEST(AsanTest, BasicAssertion) {
    EXPECT_EQ(5 + 5, 10);
}

// Проверка bool
TEST(AsanTest, TrueIsTrue) {
    EXPECT_TRUE(true);
}

// Проверка float 
TEST(AsanTest, FloatCompare) {
    float a = 0.1f * 10;
    EXPECT_NEAR(a, 1.0f, 1e-5);
}