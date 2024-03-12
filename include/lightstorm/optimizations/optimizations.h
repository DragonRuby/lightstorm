#pragma once

#include "lightstorm/config/config.h"
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

namespace lightstorm {

void applyOptimizations(const LightstormConfig &config, mlir::MLIRContext &context,
                        mlir::ModuleOp module);
}
