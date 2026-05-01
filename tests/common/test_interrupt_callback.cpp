#include <gtest/gtest.h>
#include <stdexcept>
#include <thread>
#include "OptixNNlib/interrupt_callback.h"

class TestInterruptCallback : public InterruptCallback {
public:
    void operator()() override {
        call_count++;
        if (should_throw) {
            throw std::runtime_error("Interrupted");
        }
    }
    
    int call_count = 0;
    bool should_throw = false;
};

TEST(InterruptCallback, SetAndGet) {
    TestInterruptCallback cb;
    InterruptCallback::Set(&cb);
    EXPECT_EQ(InterruptCallback::Get(), &cb);
}

TEST(InterruptCallback, CheckNoCallback) {
    InterruptCallback::Set(nullptr);
    // Should not throw
    EXPECT_NO_THROW(InterruptCallback::Check());
}

TEST(InterruptCallback, CheckWithCallback) {
    TestInterruptCallback cb;
    InterruptCallback::Set(&cb);
    
    EXPECT_NO_THROW(InterruptCallback::Check());
    EXPECT_EQ(cb.call_count, 1);
}

TEST(InterruptCallback, CheckThrows) {
    TestInterruptCallback cb;
    cb.should_throw = true;
    InterruptCallback::Set(&cb);
    
    EXPECT_THROW(InterruptCallback::Check(), std::runtime_error);
}

TEST(InterruptCallback, GetPeriodHint) {
    EXPECT_EQ(InterruptCallback::GetPeriodHint(0), 1);
    EXPECT_EQ(InterruptCallback::GetPeriodHint(100), 100);  // 10000 / 100 = 100
    EXPECT_EQ(InterruptCallback::GetPeriodHint(1000), 10);  // 10000 / 1000 = 10
    EXPECT_EQ(InterruptCallback::GetPeriodHint(1000000), 100000);  // capped
}

TEST(InterruptScope, SetsAndRestores) {
    TestInterruptCallback cb1, cb2;
    InterruptCallback::Set(&cb1);
    
    {
        InterruptScope scope(&cb2);
        EXPECT_EQ(InterruptCallback::Get(), &cb2);
        InterruptCallback::Check();
        EXPECT_EQ(cb2.call_count, 1);
    }
    
    // Restored
    EXPECT_EQ(InterruptCallback::Get(), &cb1);
}

TEST(InterruptCallback, ThreadLocal) {
    TestInterruptCallback cb1, cb2;
    
    std::thread t1([&]() {
        InterruptCallback::Set(&cb1);
        EXPECT_EQ(InterruptCallback::Get(), &cb1);
        InterruptCallback::Check();
        EXPECT_EQ(cb1.call_count, 1);
    });
    
    std::thread t2([&]() {
        InterruptCallback::Set(&cb2);
        EXPECT_EQ(InterruptCallback::Get(), &cb2);
        InterruptCallback::Check();
        EXPECT_EQ(cb2.call_count, 1);
    });
    
    t1.join();
    t2.join();
}