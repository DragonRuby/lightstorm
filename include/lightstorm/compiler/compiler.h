#pragma once

#include <filesystem>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

namespace lightstorm {

class Compiler {
public:
  std::optional<mlir::ModuleOp> compileSourceFile(mlir::MLIRContext &context,
                                                  const std::filesystem::path &file_path);

private:
};

} // namespace lightstorm
