#include "lightstorm/compiler/compiler.h"
#include <cassert>
#include <iostream>
#include <mruby.h>
#include <mruby/compile.h>

using namespace lightstorm;

extern "C" {
void mrb_codedump_all(mrb_state *, struct RProc *);
}

void Compiler::compileSourceFile(const std::filesystem::path &file_path) {
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
    return;
  }

  if (0 < p->nerr) {
    /* parse error */
    std::cerr << file_path.c_str() << ":" << p->error_buffer[0].lineno << ": "
              << p->error_buffer[0].message << "\n";
    fclose(f);
    mrb_parser_free(p);
    mrbc_context_free(mrb, mrbc);
    mrb_close(mrb);
    return;
  }

  struct RProc *proc = mrb_generate_code(mrb, p);
  mrb_codedump_all(mrb, proc);

  fclose(f);
  mrb_parser_free(p);
  mrbc_context_free(mrb, mrbc);
  mrb_close(mrb);
}
