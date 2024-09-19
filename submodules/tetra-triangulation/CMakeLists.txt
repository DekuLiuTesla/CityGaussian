cmake_minimum_required(VERSION 3.16.3)
project(fast_traversal LANGUAGES CXX CUDA)
enable_language(CUDA)

include(FetchContent)
# Add paths to our CMake code to the module path, so they can be found automatically by
# CMake.
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_BUILD_TYPE "Release")

find_library(libcnpy cnpy ${CONDA_PREFIX}/lib)
include_directories(${CONDA_PREFIX}/include)

# TODO: fix this
find_package(Torch REQUIRED)
find_package(CGAL REQUIRED)
# find_package(pybind11 REQUIRED)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.9.2
)
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()


# Put all the runtime stuff in the same directory.  By default, CMake puts each targets'
# output into their own directory.  We want all the targets to be put in the same
# directory, and we can do this by setting these variables.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(BUILD_DIRECTORY "${CMAKE_BINARY_DIR}/build")
set(CUDA_GENERATED_OUTPUT_DIR "${BUILD_DIRECTORY}")


#------------------------------------------------------------------------------
# set environment
#------------------------------------------------------------------------------

# find_package(CUDA REQUIRED)
# include_directories(${CUDA_INCLUDE_DIRS})

# configure_file("${CMAKE_SOURCE_DIR}/path_util.py.in" "${CMAKE_SOURCE_DIR}/../examples/path_util.py")


#------------------------------------------------------------------------------
# main build
#------------------------------------------------------------------------------

# configure_file(${generated_files} file.ptx.h)
add_link_options(-flto=auto)  # disables some warning
pybind11_add_module(tetranerf_cpp_extension 
    src/triangulation.cpp
    src/py_binding.cpp)

target_link_libraries(tetranerf_cpp_extension PRIVATE ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})
target_link_libraries(tetranerf_cpp_extension PRIVATE CGAL::CGAL)
target_compile_features(tetranerf_cpp_extension PRIVATE cxx_std_17)
set_target_properties(tetranerf_cpp_extension
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tetranerf/utils/extension"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
# target_compile_definitions(tetranerf_cpp_extension PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
