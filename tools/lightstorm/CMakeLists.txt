add_executable(lightstorm lightstorm.cpp)
target_link_libraries(lightstorm PRIVATE lightstorm_compiler rite_dialect LLVMOption)
target_compile_options(lightstorm PRIVATE -fno-rtti -fno-exceptions)
target_include_directories(lightstorm
  PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
