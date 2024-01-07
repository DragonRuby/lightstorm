#include <filesystem>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>

#include "lightstorm/compiler/compiler.h"

llvm::cl::OptionCategory LightstormCategory("lightstorm");

llvm::cl::opt<std::string> Input(llvm::cl::Positional, llvm::cl::Required,
                                 llvm::cl::desc("Input file"), llvm::cl::cat(LightstormCategory));

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

  lightstorm::Compiler compiler;
  compiler.compileSourceFile(Input.getValue());

  return 0;
}
