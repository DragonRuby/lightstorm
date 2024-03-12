add_library(lightstorm_optimizations STATIC
  optimizations.cpp
)
target_include_directories(lightstorm_optimizations
  PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(lightstorm_optimizations
  PRIVATE SYSTEM
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
target_link_libraries(lightstorm_optimizations
  PRIVATE
  rite_dialect
)
target_compile_options(lightstorm_optimizations PRIVATE -fno-rtti -fno-exceptions)
