add_library(lightstorm_runtime STATIC
  lightstorm_runtime.c
)
target_include_directories(lightstorm_runtime
  PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/third_party/mruby/include
)
