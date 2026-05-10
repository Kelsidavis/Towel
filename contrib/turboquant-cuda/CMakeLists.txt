if (GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_executable(tq-bench tq-bench.cpp)
    target_link_libraries(tq-bench PRIVATE ggml llama CUDA::cudart_static ${CMAKE_THREAD_LIBS_INIT})
    target_include_directories(tq-bench PRIVATE
        ${CMAKE_SOURCE_DIR}/ggml/include
        ${CMAKE_SOURCE_DIR}/include
    )
    install(TARGETS tq-bench RUNTIME)
endif()
