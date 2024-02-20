#pragma once

#include "lightstorm/config/config.h"
#include <filesystem>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

namespace lightstorm {

std::optional<mlir::ModuleOp> compileSourceFile(const LightstormConfig &config,
                                                mlir::MLIRContext &context,
                                                const std::filesystem::path &file_path);

} // namespace lightstorm
