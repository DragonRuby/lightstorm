#include <filesystem>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/EmitC/IR/EmitC.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Index/IR/IndexDialect.h>

#include "lightstorm/compiler/compiler.h"
#include "lightstorm/config/config.h"
#include "lightstorm/conversion/conversion.h"
#include "lightstorm/dialect/rite.h"

llvm::cl::OptionCategory LightstormCategory("lightstorm");

llvm::cl::opt<std::string> Input(llvm::cl::Positional, llvm::cl::Required,
                                 llvm::cl::desc("Input file"), llvm::cl::cat(LightstormCategory));

llvm::cl::opt<std::string> Output("o", llvm::cl::Optional, llvm::cl::desc("Output file"),
                                  llvm::cl::cat(LightstormCategory));

llvm::cl::opt<std::string> EntryPoint("entry", llvm::cl::Optional,
                                      llvm::cl::desc("Entry point function"),
                                      llvm::cl::init("lightstorm_top"),
                                      llvm::cl::cat(LightstormCategory));

llvm::cl::opt<bool> Verbose("v", llvm::cl::Optional, llvm::cl::desc("Verbose mode"),
                            llvm::cl::cat(LightstormCategory));

int main(int argc, char **argv) {
  llvm::llvm_shutdown_obj shutdownGuard;

  llvm::cl::HideUnrelatedOptions(LightstormCategory);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  std::error_code ec;
  if (!std::filesystem::exists(Input.getValue(), ec)) {
    llvm::errs() << "Input file doesn't exist: ";
    if (ec) {
      llvm::errs() << ec.message();
    }
    llvm::errs() << "\n";
    return 1;
  }

  mlir::DialectRegistry registry;
  registry.insert<rite::RiteDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::index::IndexDialect>();
  registry.insert<mlir::emitc::EmitCDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  lightstorm::LightstormConfig config;
  config.verbose = Verbose.getValue();
  config.entry = EntryPoint.getValue();

  auto module = lightstorm::compileSourceFile(config, context, Input.getValue());
  assert(module && "Could not convert Ruby to MLIR");

  if (config.verbose) {
    module->print(llvm::errs());
  }

  lightstorm::convertRiteToEmitC(config, context, *module);

  if (config.verbose) {
    module->print(llvm::errs());
  }

  std::filesystem::path outputFile(Output.getValue());
  if (outputFile.empty()) {
    lightstorm::convertMLIRToC(config, context, *module, llvm::errs());
  } else {
    llvm::raw_fd_stream out(outputFile.string(), ec);
    if (ec) {
      llvm::errs() << "Cannot create output file: " << ec.message() << "\n";
      return 1;
    }

    lightstorm::convertMLIRToC(config, context, *module, out);
  }
  return 0;
}
