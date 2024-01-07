#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mruby.h>

namespace lightstorm {

mlir::ModuleOp convertProcToMLIR(mlir::MLIRContext &context, mrb_state *mrb, struct RProc *proc);

} // namespace lightstorm
