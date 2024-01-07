add_library(rite_dialect STATIC rite.cpp)
target_link_libraries(rite_dialect LLVMSupport MLIRDialect)
target_include_directories(rite_dialect
  PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}/include
)
target_compile_options(rite_dialect PUBLIC -fno-rtti -fno-exceptions -fvisibility=hidden)
# Needed by mlir-tablegen
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})

set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/lightstorm/dialect/rite.td)
mlir_tablegen(RiteDialect.h.inc -gen-dialect-decls)
mlir_tablegen(RiteDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(RiteOps.h.inc -gen-op-decls)
mlir_tablegen(RiteOps.cpp.inc -gen-op-defs)
mlir_tablegen(RiteTypeDefs.h.inc -gen-typedef-decls)
mlir_tablegen(RiteTypeDefs.cpp.inc -gen-typedef-defs)
mlir_tablegen(RiteEnums.h.inc -gen-enum-decls)
mlir_tablegen(RiteEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(RiteAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(RiteAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(RiteGen)
add_dependencies(rite_dialect RiteGen)
