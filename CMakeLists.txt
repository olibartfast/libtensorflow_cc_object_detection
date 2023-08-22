cmake_minimum_required(VERSION 3.20...3.27)
project(libtensorflow_cc_object_detection)
find_package(TensorFlow REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 17)
option(BUILD_TESTS "Build test target" OFF)
option(BUILD_BENCHMARKS "Build benchmark target" OFF)

if(BUILD_TESTS)
    find_package(GTest REQUIRED)
    enable_testing()
endif()

if(BUILD_BENCHMARKS)
    find_package(benchmark REQUIRED)
endif()

add_executable(${PROJECT_NAME}
    src/libtensorflow_cc_object_detection.cpp
    src/TFObjectDetection.cpp
)

target_include_directories(${PROJECT_NAME}
    PRIVATE ${TensorFlow_INCLUDE_DIRS}
    PRIVATE ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE ${TensorFlow_LIBRARIES}
    PRIVATE ${OpenCV_LIBS}
)

if(BUILD_TESTS)
    add_executable(${PROJECT_NAME}_tests
        tests/TFObjectDetection_test.cpp
        src/TFObjectDetection.cpp
    )

    target_include_directories(${PROJECT_NAME}_tests
        PRIVATE ${TensorFlow_INCLUDE_DIRS}
        PRIVATE ${OpenCV_INCLUDE_DIRS}
        PRIVATE ${GTEST_INCLUDE_DIRS}
        PRIVATE src
    )

    target_link_libraries(${PROJECT_NAME}_tests
        PRIVATE ${TensorFlow_LIBRARIES}
        PRIVATE ${OpenCV_LIBS}
        PRIVATE ${GTEST_BOTH_LIBRARIES}
    )

    add_test(NAME ${PROJECT_NAME}_tests COMMAND ${PROJECT_NAME}_tests --gtest_output=xml:${PROJECT_BINARY_DIR}/test_results.xml)
    set_tests_properties(${PROJECT_NAME}_tests PROPERTIES TIMEOUT 300)
endif()

if(BUILD_BENCHMARKS)
    add_executable(${PROJECT_NAME}_benchmark
        benchmarks/TFObjectDetection_benchmark.cpp
        src/TFObjectDetection.cpp
    )

    target_include_directories(${PROJECT_NAME}_benchmark
        PRIVATE ${TensorFlow_INCLUDE_DIRS}
        PRIVATE ${OpenCV_INCLUDE_DIRS}
        PRIVATE ${benchmark_INCLUDE_DIRS}
        PRIVATE src
    )

    target_link_libraries(${PROJECT_NAME}_benchmark
        PRIVATE ${TensorFlow_LIBRARIES}
        PRIVATE ${OpenCV_LIBS}
        PRIVATE ${benchmark_LIBRARIES}
        benchmark
    )

    set_target_properties(${PROJECT_NAME}_benchmark
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
    )
endif()
