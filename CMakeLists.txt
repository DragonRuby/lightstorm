cmake_minimum_required(VERSION 3.27)
project(lightstorm)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PROJECT_VERSION 0.1.0)

set(PROJECT_DESCRIPTION "Minimalistic ahead-of-time Ruby compiler")
set(PROJECT_HOMEPAGE_URL "https://github.com/dragonruby/lightstorm")

include(GNUInstallDirs)

find_package(LLVM CONFIG REQUIRED)
message("Using LLVM_CONFIG: ${LLVM_CONFIG}")
find_package(MLIR CONFIG REQUIRED)
message("Using MLIR_CONFIG: ${MLIR_CONFIG}")

include(cmake/mruby.cmake)
include(cmake/version.cmake)
include(cmake/lightstorm.cmake)

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
