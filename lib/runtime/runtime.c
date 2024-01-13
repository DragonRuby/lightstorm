#include "lightstorm/runtime/runtime.h"
#include <assert.h>
#include <mruby/array.h>
#include <mruby/class.h>
#include <mruby/error.h>
#include <mruby/hash.h>
#include <mruby/numeric.h>
#include <mruby/presym.h>
#include <mruby/proc.h>
#include <mruby/string.h>

void mrb_exc_set(mrb_state *mrb, mrb_value exc);

#pragma mark - Comparisons

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

#pragma mark - Arithmetic

#define OP_MATH(op_name, lhs, rhs)                                                                 \
  /* need to check if op is overridden */                                                          \
  switch (TYPES2(mrb_type(lhs), mrb_type(rhs))) {                                                  \
    OP_MATH_CASE_INTEGER(op_name);                                                                 \
    OP_MATH_CASE_FLOAT(op_name, integer, float);                                                   \
    OP_MATH_CASE_FLOAT(op_name, float, integer);                                                   \
    OP_MATH_CASE_FLOAT(op_name, float, float);                                                     \
    OP_MATH_CASE_STRING_##op_name();                                                               \
  default: {                                                                                       \
    mrb_sym mid = MRB_OPSYM(op_name);                                                              \
    mrb_value ret = mrb_funcall_id(mrb, lhs, mid, 1, &rhs);                                        \
    return ret;                                                                                    \
  }                                                                                                \
  }

#define OP_MATH_CASE_INTEGER(op_name)                                                              \
  case TYPES2(MRB_TT_INTEGER, MRB_TT_INTEGER): {                                                   \
    mrb_int x = mrb_integer(lhs), y = mrb_integer(rhs), z;                                         \
    if (mrb_int_##op_name##_overflow(x, y, &z)) {                                                  \
      OP_MATH_OVERFLOW_INT();                                                                      \
    } else {                                                                                       \
      return mrb_fixnum_value(z);                                                                  \
    }                                                                                              \
  }
#define OP_MATH_CASE_FLOAT(op_name, t1, t2)                                                        \
  case TYPES2(OP_MATH_TT_##t1, OP_MATH_TT_##t2): {                                                 \
    mrb_float z = mrb_##t1(lhs) OP_MATH_OP_##op_name mrb_##t2(rhs);                                \
    return mrb_float_value(mrb, z);                                                                \
  }

#define OP_MATH_OVERFLOW_INT()                                                                     \
  {                                                                                                \
    mrb_value exc = mrb_exc_new_lit(mrb, E_RANGE_ERROR, "integer overflow");                       \
    mrb_exc_set(mrb, exc);                                                                         \
    mrb_exc_raise(mrb, mrb_obj_value(mrb->exc));                                                   \
    return mrb_nil_value();                                                                        \
  }

#define OP_MATH_CASE_STRING_add()                                                                  \
  case TYPES2(MRB_TT_STRING, MRB_TT_STRING): {                                                     \
    return mrb_str_plus(mrb, lhs, rhs);                                                            \
  }
#define OP_MATH_CASE_STRING_sub() (void)0
#define OP_MATH_CASE_STRING_mul() (void)0
#define OP_MATH_OP_add +
#define OP_MATH_OP_sub -
#define OP_MATH_OP_mul *
#define OP_MATH_TT_integer MRB_TT_INTEGER
#define OP_MATH_TT_float MRB_TT_FLOAT

LIGHTSTORM_INLINE mrb_value ls_arith_add(mrb_state *mrb, mrb_value lhs, mrb_value rhs) {
  OP_MATH(add, lhs, rhs);
}

LIGHTSTORM_INLINE mrb_value ls_arith_sub(mrb_state *mrb, mrb_value lhs, mrb_value rhs) {
  OP_MATH(sub, lhs, rhs);
}

LIGHTSTORM_INLINE mrb_value ls_arith_mul(mrb_state *mrb, mrb_value lhs, mrb_value rhs) {
  OP_MATH(mul, lhs, rhs);
}

LIGHTSTORM_INLINE mrb_value ls_arith_div(mrb_state *mrb, mrb_value lhs, mrb_value rhs) {
  mrb_int mrb_num_div_int(mrb_state * mrb, mrb_int x, mrb_int y);
  mrb_float mrb_num_div_flo(mrb_state * mrb, mrb_float x, mrb_float y);
  mrb_float x, y, f;
  switch (TYPES2(mrb_type(lhs), mrb_type(rhs))) {
  case TYPES2(MRB_TT_INTEGER, MRB_TT_INTEGER): {
    mrb_int xi = mrb_integer(lhs);
    mrb_int yi = mrb_integer(rhs);
    mrb_int div = mrb_num_div_int(mrb, xi, yi);
    return mrb_int_value(mrb, div);
  } break;
  case TYPES2(MRB_TT_INTEGER, MRB_TT_FLOAT):
    x = (mrb_float)mrb_integer(lhs);
    y = mrb_float(rhs);
    break;
  case TYPES2(MRB_TT_FLOAT, MRB_TT_INTEGER):
    x = mrb_float(lhs);
    y = (mrb_float)mrb_integer(rhs);
    break;
  case TYPES2(MRB_TT_FLOAT, MRB_TT_FLOAT):
    x = mrb_float(lhs);
    y = mrb_float(rhs);
    break;
  default: {
    mrb_sym mid = MRB_OPSYM(div);
    mrb_value ret = mrb_funcall_id(mrb, lhs, mid, 1, &rhs);
    return ret;
  }
  }

  f = mrb_num_div_flo(mrb, x, y);
  return mrb_float_value(mrb, f);
}

LIGHTSTORM_INLINE mrb_value ls_load_target_class_value(mrb_state *mrb) {
  struct RClass *targetClass = mrb_vm_ci_target_class(mrb->c->ci);
  if (!targetClass) {
    mrb_value exc = mrb_exc_new_lit(mrb, E_TYPE_ERROR, "no target class or module");
    mrb_exc_raise(mrb, exc);
  }
  return mrb_obj_value(targetClass);
}

LIGHTSTORM_INLINE mrb_value ls_create_method(mrb_state *mrb, mrb_func_t func) {
  return mrb_obj_value(mrb_proc_new_cfunc(mrb, func));
}

LIGHTSTORM_INLINE mrb_value ls_define_method(mrb_state *mrb, mrb_value target, mrb_value method,
                                             mrb_sym mid) {
  struct RClass *targetClass = mrb_class_ptr(target);
  assert(targetClass->tt == MRB_TT_CLASS || targetClass->tt == MRB_TT_MODULE ||
         targetClass->tt == MRB_TT_SCLASS);
  struct RProc *proc = mrb_proc_ptr(method);
  mrb_method_t m;
  MRB_METHOD_FROM_PROC(m, proc);
  mrb_define_method_raw(mrb, targetClass, mid, m);
  return mrb_nil_value();
}

LIGHTSTORM_INLINE mrb_value ls_array(mrb_state *mrb, mrb_int size, ...) {
  mrb_value argv[size];
  va_list args;
  va_start(args, size);
  for (mrb_int i = 0; i < size; i++) {
    argv[i] = va_arg(args, mrb_value);
  }
  va_end(args);
  return mrb_ary_new_from_values(mrb, size, argv);
}

LIGHTSTORM_INLINE mrb_value ls_hash(mrb_state *mrb, mrb_int size, ...) {
  mrb_value argv[size];
  va_list args;
  va_start(args, size);
  for (mrb_int i = 0; i < size; i++) {
    argv[i] = va_arg(args, mrb_value);
  }
  va_end(args);

  mrb_value hash = mrb_hash_new_capa(mrb, size);
  for (int i = 0; i < size; i += 2) {
    mrb_hash_set(mrb, hash, argv[i], argv[i + 1]);
  }

  return hash;
}

LIGHTSTORM_INLINE mrb_value ls_define_module(mrb_state *mrb, mrb_value target, mrb_sym sym) {
  if (mrb_nil_p(target)) {
    // TODO: check when proc is null, this shouldn't happen?
    // TODO: Fix target class
    assert(mrb->c->ci->proc);
    struct RClass *baseclass = MRB_PROC_TARGET_CLASS(mrb->c->ci->proc);
    if (!baseclass)
      baseclass = mrb->object_class;
    target = mrb_obj_value(baseclass);
  }
  struct RClass *cls = mrb_vm_define_module(mrb, target, sym);
  //  mrb_gc_arena_restore(mrb, ai);
  return mrb_obj_value(cls);
}

LIGHTSTORM_INLINE mrb_value ls_exec(mrb_state *mrb, mrb_value receiver, mrb_func_t func) {
  struct RClass *targetClass = mrb_class_ptr(receiver);
  const struct RProc *upperProc = mrb->c->ci->proc;
  struct RProc *proc = mrb_proc_new_cfunc(mrb, func);
  MRB_PROC_SET_TARGET_CLASS(proc, targetClass);
  proc->flags |= MRB_PROC_SCOPE;
  proc->upper = upperProc;
  mrb_vm_ci_proc_set(mrb->c->ci, proc);
  mrb_value ret = func(mrb, receiver);
  return ret;
}
