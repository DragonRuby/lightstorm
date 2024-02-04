#include "lightstorm/compiler/compiler.h"
#include "converter.h"
#include <cassert>
#include <iostream>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <mruby.h>
#include <mruby/compile.h>

using namespace lightstorm;

std::optional<mlir::ModuleOp> Compiler::compileSourceFile(mlir::MLIRContext &context,
                                                          const std::filesystem::path &file_path) {
  assert(exists(file_path) && "Cannot compile file");
  mrb_state *mrb = mrb_open();
  assert(mrb && "Out of memory?");
  mrbc_context *mrbc = mrbc_context_new(mrb);
  assert(mrbc && "Out of memory?");
  mrbc_filename(mrb, mrbc, file_path.c_str());
  mrbc->capture_errors = TRUE;
  mrbc->no_optimize = TRUE;

  FILE *f = fopen(file_path.c_str(), "r");
  assert(f && "No file?");

  struct mrb_parser_state *p = mrb_parse_file(mrb, f, mrbc);

  if (!p) {
    /* only occur when memory ran out */
    std::cerr << "Failed to create parser state (out of memory)\n";
    fclose(f);
    mrbc_context_free(mrb, mrbc);
    mrb_close(mrb);
    exit(1);
    return {};
  }

  if (0 < p->nerr) {
    /* parse error */
    std::cerr << file_path.c_str() << ":" << p->error_buffer[0].lineno << ":"
              << p->error_buffer[0].column << ": " << (p->error_buffer[0].message ?: "") << "\n";
    fclose(f);
    mrb_parser_free(p);
    mrbc_context_free(mrb, mrbc);
    mrb_close(mrb);
    exit(1);
    return {};
  }

  struct RProc *proc = mrb_generate_code(mrb, p);
  assert(proc && "Could not generate code");
  auto module = convertProcToMLIR(context, mrb, proc);
  fclose(f);
  mrb_parser_free(p);
  mrbc_context_free(mrb, mrbc);
  mrb_close(mrb);

  if (mlir::failed(mlir::verify(module))) {
    return {};
  }

  mlir::PassManager construction(&context);
  construction.addPass(mlir::createMem2Reg());
  construction.addPass(mlir::createCSEPass());
  if (construction.run(module).failed()) {
    module.print(llvm::errs());
    llvm::errs() << "\nFailed to run passes\n";
    exit(1);
  }

  return module;
}
