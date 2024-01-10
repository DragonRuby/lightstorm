#pragma once

#include <mruby.h>

#define LIGHTSTORM_INLINE __attribute__((always_inline)) inline
// EmitC uses `bool` for conditional branch predicates
typedef int bool;

LIGHTSTORM_INLINE mrb_value ls_load_self(mrb_state *mrb) {
  return mrb->c->ci->stack[0];
}

LIGHTSTORM_INLINE mrb_value ls_load_i(mrb_state *mrb, int64_t i) {
  return mrb_fixnum_value(i);
}

LIGHTSTORM_INLINE mrb_value ls_load_nil(mrb_state *mrb) {
  return mrb_nil_value();
}

#define ls_funcall_0(mrb, recv, name, argc) mrb_funcall_id(mrb, recv, name, argc, NULL)
#define ls_funcall_1(mrb, recv, name, argc, ...) mrb_funcall_id(mrb, recv, name, argc, __VA_ARGS__)
#define ls_funcall_2(mrb, recv, name, argc, ...) mrb_funcall_id(mrb, recv, name, argc, __VA_ARGS__)
#define ls_funcall_3(mrb, recv, name, argc, ...) mrb_funcall_id(mrb, recv, name, argc, __VA_ARGS__)
#define ls_funcall_4(mrb, recv, name, argc, ...) mrb_funcall_id(mrb, recv, name, argc, __VA_ARGS__)
#define ls_funcall_5(mrb, recv, name, argc, ...) mrb_funcall_id(mrb, recv, name, argc, __VA_ARGS__)
#define ls_funcall_6(mrb, recv, name, argc, ...) mrb_funcall_id(mrb, recv, name, argc, __VA_ARGS__)
#define ls_funcall_7(mrb, recv, name, argc, ...) mrb_funcall_id(mrb, recv, name, argc, __VA_ARGS__)
#define ls_funcall_8(mrb, recv, name, argc, ...) mrb_funcall_id(mrb, recv, name, argc, __VA_ARGS__)
#define ls_funcall_9(mrb, recv, name, argc, ...) mrb_funcall_id(mrb, recv, name, argc, __VA_ARGS__)

// TODO: Inline all this here or turn on LTO?
mrb_value ls_compare_gt(mrb_state *mrb, mrb_value lhs, mrb_value rhs);
mrb_value ls_compare_ge(mrb_state *mrb, mrb_value lhs, mrb_value rhs);
mrb_value ls_compare_lt(mrb_state *mrb, mrb_value lhs, mrb_value rhs);
mrb_value ls_compare_le(mrb_state *mrb, mrb_value lhs, mrb_value rhs);
mrb_value ls_compare_eq(mrb_state *mrb, mrb_value lhs, mrb_value rhs);

LIGHTSTORM_INLINE int ls_value_true(mrb_value value) {
  return mrb_true_p(value);
}
LIGHTSTORM_INLINE int ls_value_false(mrb_value value) {
  return mrb_false_p(value);
}
LIGHTSTORM_INLINE int ls_value_nil(mrb_value value) {
  return mrb_nil_p(value);
}

mrb_value lightstorm_top(mrb_state *mrb, mrb_value self);
