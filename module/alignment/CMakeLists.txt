# CMakeLists.txt for alignment
set(Module "alignment")

# Include
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source
set(ALIGNMENT_CPP src/base_alignment.cpp src/feature_alignment.cpp src/sparse_image_alignment.cpp)

# Execute
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
add_executable(test_ftr_align ${ALIGNMENT_CPP} test/test_ftr_align.cpp)
add_executable(test_sparse_img_align ${ALIGNMENT_CPP} test/test_sparse_img_align.cpp)
target_link_libraries(test_ftr_align ${OpenCV_LIBS})
target_link_libraries(test_sparse_img_align ${OpenCV_LIBS})