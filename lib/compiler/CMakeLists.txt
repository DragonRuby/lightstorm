add_library(lightstorm_compiler
  compiler.cpp
)
target_include_directories(lightstorm_compiler
  PRIVATE
  ${CMAKE_SOURCE_DIR}/third_party/mruby/include
  ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(lightstorm_compiler PRIVATE mruby_static)
target_compile_options(lightstorm_compiler PRIVATE -fno-rtti -fno-exceptions)
