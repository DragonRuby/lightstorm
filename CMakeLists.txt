cmake_minimum_required(VERSION 3.27)
project(LightStorm)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(cmake/mruby.cmake)

find_package(LLVM CONFIG REQUIRED)
message("Using LLVM_CONFIG: ${LLVM_CONFIG}")
find_package(MLIR CONFIG REQUIRED)
message("Using MLIR_CONFIG: ${MLIR_CONFIG}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_DIR}")
include(AddLLVM)
include(TableGen)
include(AddMLIR)

add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(runtime)
