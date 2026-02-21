cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(BestHistogramEqualization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)

add_executable(${PROJECT_NAME} src/main.cpp
    src/equalizations.cpp
    src/metrics.cpp
    src/utilities.cpp
)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
