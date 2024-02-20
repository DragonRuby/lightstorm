#pragma once

#include "lightstorm/config/config.h"
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

namespace lightstorm {

void convertRiteToEmitC(const LightstormConfig &config, mlir::MLIRContext &context,
                        mlir::ModuleOp module);
void convertMLIRToC(const LightstormConfig &config, mlir::MLIRContext &context,
                    mlir::ModuleOp module, llvm::raw_ostream &out);

} // namespace lightstorm
