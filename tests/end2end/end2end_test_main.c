#include <mruby.h>
#include <mruby/irep.h>
#include <mruby/proc.h>

mrb_value lightstorm_top(mrb_state *mrb, mrb_value self);
extern const uint8_t lightstorm_host[];

int main() {
  mrb_state *mrb = mrb_open();
  struct RProc *proc = mrb_proc_new_cfunc(mrb, lightstorm_top);
  MRB_PROC_SET_TARGET_CLASS(proc, mrb->object_class);
  mrb->c->ci->proc = proc;
  mrb_value self = mrb_top_self(mrb);
  mrb->c->ci->stack[0] = self;
  lightstorm_top(mrb, self);
  mrb_load_irep(mrb, lightstorm_host);
  mrb_close(mrb);
  return 0;
}
