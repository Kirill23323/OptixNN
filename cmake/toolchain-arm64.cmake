# ==============================
#  TOOLCHAIN FOR AARCH64 LINUX
# ==============================

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Компиляторы для ARM
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# Генерация NEON-инструкций
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")

# Sanitize/PIE fix (QEMU compatibility)
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-pie")
