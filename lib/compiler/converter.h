#pragma once

#include "lightstorm/config/config.h"
#include <mlir/IR/BuiltinOps.h>
#include <mruby.h>

namespace lightstorm {

mlir::ModuleOp convertProcToMLIR(const LightstormConfig &config, mlir::MLIRContext &context,
                                 mrb_state *mrb, struct RProc *proc);

} // namespace lightstorm
