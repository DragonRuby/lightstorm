cmake_minimum_required(VERSION 3.27)
project(lightstorm)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)

include(cmake/mruby.cmake)
include(cmake/lightstorm.cmake)

find_package(LLVM CONFIG REQUIRED)
message("Using LLVM_CONFIG: ${LLVM_CONFIG}")
find_package(MLIR CONFIG REQUIRED)
message("Using MLIR_CONFIG: ${MLIR_CONFIG}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_DIR}")
include(AddLLVM)
include(TableGen)
include(AddMLIR)

set(LIGHTSTORM_CFLAGS -g)

if(LIGHTSTORM_LTO)
  set(LIGHTSTORM_CFLAGS -g -flto -O3)
endif()

add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(tests)
