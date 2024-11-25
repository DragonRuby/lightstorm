#include "simple_kpc.h"
#include <mruby.h>
#include <mruby/irep.h>
#include <mruby/proc.h>

extern const uint8_t lightstorm_bench[];

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
  sk_in_progress_measurement *m = sk_start_measurement(e);
  mrb_load_irep(mrb, lightstorm_bench);
  sk_finish_measurement(m);
  mrb_close(mrb);
  sk_events_destroy(e);
  return 0;
}
