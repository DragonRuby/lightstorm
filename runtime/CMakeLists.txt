add_library(lightstorm_runtime STATIC
  lightstorm_runtime.c
  lightstorm_runtime.h
)
target_include_directories(lightstorm_runtime
  PRIVATE
  ${CMAKE_SOURCE_DIR}/third_party/mruby/include
)
