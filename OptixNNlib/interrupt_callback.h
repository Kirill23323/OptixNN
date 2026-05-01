#pragma once
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <algorithm>

class InterruptCallback {
public:
    virtual ~InterruptCallback() = default;

    virtual void operator()() = 0;

    static void Set(InterruptCallback* cb) {
        instance = cb;
    }

    static InterruptCallback* Get() {
        return instance;
    }

    static inline void Check() {
        if (instance) {
            (*instance)();  
        }
    }

    static size_t GetPeriodHint(size_t work_per_iter) {
        constexpr size_t TARGET = 10000;

        if (work_per_iter == 0) {
            return 1;
        } 

        size_t period = TARGET / work_per_iter;

        // ограничения, чтобы не было крайностей
        if (period < 1) {
            return 1;
        } 
        if (period > 100000) {
            return 100000;
        } 

        return period;
    }

private:
    
    static inline thread_local InterruptCallback* instance = nullptr;
};


class InterruptScope {
    InterruptCallback* prev;

public:
    explicit InterruptScope(InterruptCallback* cb) {
        prev = InterruptCallback::Get();
        InterruptCallback::Set(cb);
    }

    ~InterruptScope() {
        InterruptCallback::Set(prev);
    }
};