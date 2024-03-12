add_library(lightstorm_conversion STATIC
  c_conversion.cpp
)
target_include_directories(lightstorm_conversion
  PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(lightstorm_conversion
  PRIVATE SYSTEM
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
target_link_libraries(lightstorm_conversion
  PRIVATE
  rite_dialect
  MLIRTransforms
  MLIREmitCDialect
  MLIRFuncTransforms
  MLIRFuncToEmitC
  MLIRTargetCpp
)
target_compile_options(lightstorm_conversion PRIVATE -fno-rtti -fno-exceptions)
