# CMakeLists.txt for alignment
set(Module "math_utils")

# Include
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source
# set(MATH_UTILS_CPP src/optimizer_base.cpp)

# # Library
# set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)
# add_library(optimizer_base SHARED ${MATH_UTILS_CPP})

# Execute
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
add_executable(test_optimizer test/test_optimizer.cpp )
