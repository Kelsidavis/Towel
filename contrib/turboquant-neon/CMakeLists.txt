cmake_minimum_required(VERSION 3.16)
project(turboquant-neon C)

set(CMAKE_C_STANDARD 11)

# Detect ARM — this only builds on AArch64
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    message(FATAL_ERROR "turboquant-neon requires AArch64 (ARM64)")
endif()

add_library(turboquant_neon STATIC turboquant_neon.c)
target_include_directories(turboquant_neon PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(turboquant_neon PRIVATE -O3 -march=native -Wall -Wextra)
target_link_libraries(turboquant_neon PRIVATE m)

add_executable(tq_bench_neon tq_bench_neon.c)
target_link_libraries(tq_bench_neon PRIVATE turboquant_neon m)
target_compile_options(tq_bench_neon PRIVATE -O3 -march=native)
