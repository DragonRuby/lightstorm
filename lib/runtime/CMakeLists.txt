add_library(lightstorm_runtime STATIC
  lightstorm_runtime.c
)
target_include_directories(lightstorm_runtime
  PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/third_party/mruby/include
  ${CMAKE_SOURCE_DIR}/third_party/mruby/build/host/include
)
target_compile_options(lightstorm_runtime PRIVATE -flto)
add_dependencies(lightstorm_runtime mruby_static)
