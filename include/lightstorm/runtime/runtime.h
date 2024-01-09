#pragma once

#include <mruby.h>

#define LIGHTSTORM_INLINE __attribute__((always_inline)) static inline

LIGHTSTORM_INLINE mrb_value ls_load_self(mrb_state *mrb) {
  return mrb->c->ci->stack[0];
}

LIGHTSTORM_INLINE mrb_value ls_load_i(mrb_state *mrb, int64_t i) {
  return mrb_fixnum_value(i);
}

LIGHTSTORM_INLINE mrb_value ls_funcall_1(mrb_state *mrb, mrb_value recv, mrb_sym name, mrb_int argc,
                                         mrb_value v0) {
  return mrb_funcall_id(mrb, recv, name, argc, v0);
}

mrb_value lightstorm_top(mrb_state *mrb, mrb_value self);
