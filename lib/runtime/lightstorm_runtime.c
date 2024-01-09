#include "lightstorm/runtime/runtime.h"
#include <mruby/presym.h>
#include <mruby/proc.h>

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

#define TYPES2(a, b) ((((uint16_t)(a)) << 8) | (((uint16_t)(b)) & 0xff))

#define FS_COMPARE(mrb, lhs, rhs, op, sym)                                                         \
  int result;                                                                                      \
  switch (TYPES2(mrb_type(lhs), mrb_type(rhs))) {                                                  \
  case TYPES2(MRB_TT_FIXNUM, MRB_TT_FIXNUM):                                                       \
    result = mrb_fixnum(lhs) op mrb_fixnum(rhs);                                                   \
    break;                                                                                         \
  case TYPES2(MRB_TT_FIXNUM, MRB_TT_FLOAT):                                                        \
    result = mrb_fixnum(lhs) op mrb_float(rhs);                                                    \
    break;                                                                                         \
  case TYPES2(MRB_TT_FLOAT, MRB_TT_FIXNUM):                                                        \
    result = mrb_float(lhs) op mrb_fixnum(rhs);                                                    \
    break;                                                                                         \
  case TYPES2(MRB_TT_FLOAT, MRB_TT_FLOAT):                                                         \
    result = mrb_float(lhs) op mrb_float(rhs);                                                     \
    break;                                                                                         \
  default: {                                                                                       \
    mrb_sym mid = MRB_OPSYM(sym);                                                                  \
    mrb_value ret = mrb_funcall_id(mrb, lhs, mid, 1, &rhs);                                        \
    return ret;                                                                                    \
  }                                                                                                \
  }                                                                                                \
  if (result) {                                                                                    \
    return mrb_true_value();                                                                       \
  }                                                                                                \
  return mrb_false_value();

LIGHTSTORM_INLINE mrb_value ls_compare_le(mrb_state *mrb, mrb_value lhs, mrb_value rhs) {
  FS_COMPARE(mrb, lhs, rhs, <=, le);
}
LIGHTSTORM_INLINE mrb_value ls_compare_lt(mrb_state *mrb, mrb_value lhs, mrb_value rhs) {
  FS_COMPARE(mrb, lhs, rhs, <, lt);
}
LIGHTSTORM_INLINE mrb_value ls_compare_ge(mrb_state *mrb, mrb_value lhs, mrb_value rhs) {
  FS_COMPARE(mrb, lhs, rhs, >=, ge);
}
LIGHTSTORM_INLINE mrb_value ls_compare_gt(mrb_state *mrb, mrb_value lhs, mrb_value rhs) {
  FS_COMPARE(mrb, lhs, rhs, >, gt);
}
LIGHTSTORM_INLINE mrb_value ls_compare_eq(mrb_state *mrb, mrb_value lhs, mrb_value rhs) {
  if (mrb_obj_eq(mrb, lhs, rhs)) {
    return mrb_true_value();
  }
  FS_COMPARE(mrb, lhs, rhs, ==, eq);
}

#undef FS_COMPARE
