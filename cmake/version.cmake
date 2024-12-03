execute_process(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT
  OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
  COMMAND date "+%d %b %Y"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE BUILD_DATE
  OUTPUT_STRIP_TRAILING_WHITESPACE)

set(MRUBY_VERSION
    ${MRUBY_RELEASE_MAJOR}.${MRUBY_RELEASE_MINOR}.${MRUBY_RELEASE_TEENY})

configure_file(${CMAKE_SOURCE_DIR}/lib/version.cpp
               ${CMAKE_BINARY_DIR}/lib/version.cpp @ONLY)

add_library(lightstorm_version ${CMAKE_BINARY_DIR}/lib/version.cpp)
target_include_directories(lightstorm_version
                           PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(lightstorm_version SYSTEM
                           PRIVATE ${LLVM_INCLUDE_DIRS})
