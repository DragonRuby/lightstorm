#pragma once

#include <mruby.h>

#define LIGHTSTORM_INLINE __attribute__((always_inline)) inline

LIGHTSTORM_INLINE mrb_value ls_load_self(mrb_state *mrb) {
  return mrb->c->ci->stack[0];
}

LIGHTSTORM_INLINE mrb_value ls_load_i(mrb_state *mrb, int64_t i) {
  return mrb_fixnum_value(i);
}

LIGHTSTORM_INLINE mrb_value ls_load_nil(mrb_state *mrb) {
  return mrb_nil_value();
}

LIGHTSTORM_INLINE mrb_value ls_funcall_1(mrb_state *mrb, mrb_value recv, mrb_sym name, mrb_int argc,
                                         mrb_value v0) {
  return mrb_funcall_id(mrb, recv, name, argc, v0);
}

// TODO: Inline all this here or turn on LTO?
mrb_value ls_compare_gt(mrb_state *mrb, mrb_value lhs, mrb_value rhs);
mrb_value ls_compare_ge(mrb_state *mrb, mrb_value lhs, mrb_value rhs);
mrb_value ls_compare_lt(mrb_state *mrb, mrb_value lhs, mrb_value rhs);
mrb_value ls_compare_le(mrb_state *mrb, mrb_value lhs, mrb_value rhs);
mrb_value ls_compare_eq(mrb_state *mrb, mrb_value lhs, mrb_value rhs);

mrb_value lightstorm_top(mrb_state *mrb, mrb_value self);
