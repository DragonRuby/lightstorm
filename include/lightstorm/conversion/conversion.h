#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

namespace lightstorm {

void convertRiteToEmitC(mlir::MLIRContext &context, mlir::ModuleOp module);
void convertMLIRToC(mlir::MLIRContext &context, mlir::ModuleOp module, llvm::raw_ostream &out);

} // namespace lightstorm
