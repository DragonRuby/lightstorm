add_library(lightstorm_compiler STATIC
  compiler.cpp
  converter.cpp
)
target_include_directories(lightstorm_compiler
  PRIVATE
  ${CMAKE_SOURCE_DIR}/third_party/mruby/include
  ${CMAKE_SOURCE_DIR}/include
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
target_link_libraries(lightstorm_compiler
  PRIVATE
  mruby_static
  rite_dialect
  MLIRFuncDialect
  MLIRIndexDialect
  MLIRTransforms
)
target_compile_options(lightstorm_compiler PRIVATE -fno-rtti -fno-exceptions)
