#include "simple_kpc.h"
#include <mruby.h>
#include <mruby/proc.h>

mrb_value lightstorm_top(mrb_state *mrb, mrb_value self);

int main() {
  sk_init();

  sk_events *e = sk_events_create();
  sk_events_push(e, "cycles", "FIXED_CYCLES");
  sk_events_push(e, "instructions", "FIXED_INSTRUCTIONS");
  sk_events_push(e, "branches", "INST_BRANCH");
  sk_events_push(e, "branch misses", "BRANCH_MISPRED_NONSPEC");
  sk_events_push(e, "load/stores", "INST_LDST");
  sk_events_push(e, "INST_SIMD_LD", "INST_SIMD_LD");
  sk_events_push(e, "INST_SIMD_ST", "INST_SIMD_ST");
  sk_events_push(e, "INST_BRANCH_INDIR", "INST_BRANCH_INDIR");

  mrb_state *mrb = mrb_open();
  struct RProc *proc = mrb_proc_new_cfunc(mrb, lightstorm_top);
  MRB_PROC_SET_TARGET_CLASS(proc, mrb->object_class);
  mrb->c->ci->proc = proc;
  mrb_value self = mrb_top_self(mrb);
  mrb->c->ci->stack[0] = self;
  sk_in_progress_measurement *m = sk_start_measurement(e);
  lightstorm_top(mrb, self);
  sk_finish_measurement(m);
  mrb_close(mrb);
  sk_events_destroy(e);
  return 0;
}
