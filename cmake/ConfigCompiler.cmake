# ===============================
#  COMMON COMPILER FLAGS
# ===============================

# Оптимизация
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Строгие варнинги
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

# Для x86 — автоподбор фичей
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()
