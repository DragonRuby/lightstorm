add_library(lightstorm_runtime STATIC
  runtime.c
)
target_include_directories(lightstorm_runtime
  PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/third_party/mruby/include
  ${CMAKE_SOURCE_DIR}/third_party/mruby/build/host/include
)
target_compile_options(lightstorm_runtime PRIVATE -g)
add_dependencies(lightstorm_runtime mruby_static)

add_library(lightstorm_runtime_main STATIC
  runtime_main.c
)
target_include_directories(lightstorm_runtime_main
  PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/third_party/mruby/include
  ${CMAKE_SOURCE_DIR}/third_party/mruby/build/host/include
)
target_compile_options(lightstorm_runtime_main PRIVATE -g)
add_dependencies(lightstorm_runtime_main mruby_static)
