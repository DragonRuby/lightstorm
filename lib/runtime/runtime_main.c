#include <mruby.h>
#include <mruby/proc.h>

mrb_value lightstorm_top(mrb_state *mrb, mrb_value self);

int main() {
  mrb_state *mrb = mrb_open();
  struct RProc *proc = mrb_proc_new_cfunc(mrb, lightstorm_top);
  MRB_PROC_SET_TARGET_CLASS(proc, mrb->object_class);
  mrb->c->ci->proc = proc;
  mrb_value self = mrb_top_self(mrb);
  mrb->c->ci->stack[0] = self;
  lightstorm_top(mrb, self);
  mrb_close(mrb);
  return 0;
}
