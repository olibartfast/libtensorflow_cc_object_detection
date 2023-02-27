cmake_minimum_required(VERSION 3.0)

project(libtensorflow_cc_object_detection)
find_package(TensorFlow REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/libtensorflow_cc_object_detection.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${TensorFlow_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${TensorFlow_LIBRARIES}  ${OpenCV_LIBS})