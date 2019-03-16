cmake_minimum_required(VERSION 3.0.0)
project(vo_kit VERSION 0.1.0)

# Build Type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()
message("Build Type: " ${CMAKE_BUILD_TYPE})

# C++11
# set(CMAKE_CXX_STANDARD 11)
add_definitions(-std=c++11)

# Debug and log messages
if(CMAKE_BUILD_TYPE MATCHES "Debug")
	add_definitions(-D_DEBUG_MODE_)
endif(CMAKE_BUILD_TYPE MATCHES "Debug")

# Output message
set(DEBUG_OUTPUT_MESSAGES true)
if(DEBUG_OUTPUT_MESSAGES)
	add_definitions(-D_OUTPUT_MESSAGES_)
endif(DEBUG_OUTPUT_MESSAGES)

# 打开提示信息；Debug模式下，打开调试信息；打开优化；
if( CMAKE_BUILD_TYPE MATCHES "Debug" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall -g -march=native -fopenmp")
else( CMAKE_BUILD_TYPE MATCHES "Release" )
# release模式下，最高优化；
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall -O3 -march=native  -fopenmp")
endif( CMAKE_BUILD_TYPE MATCHES "Debug" )

# OpenCV Library
find_package(OpenCV 3.2.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Eigen Library
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Module aligment
add_subdirectory(module/alignment)
set(Alignment_INCLUDE_DIR ${PROJECT_SOURCE_DIR}module/alignment/include)

# Module math_utils
add_subdirectory(module/math_utils)
set(Math_utils_INCLUDE_DIR ${PROJECT_SOURCE_DIR}module/math_utils/include)

