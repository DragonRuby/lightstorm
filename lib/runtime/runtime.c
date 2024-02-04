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
#include <mruby/throw.h>
#include <stdlib.h>

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
    mrb_value ret = ls_send_argv(mrb, lhs, mid, 1, &rhs);                                          \
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
    mrb_value ret = ls_send_argv(mrb, lhs, mid, 1, &rhs);                                          \
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
    mrb_value ret = ls_send_argv(mrb, lhs, mid, 1, &rhs);
    return ret;
  }
  }

  f = mrb_num_div_flo(mrb, x, y);
  return mrb_float_value(mrb, f);
}

LIGHTSTORM_INLINE mrb_value ls_load_target_class_value(mrb_state *mrb) {
  struct RClass *targetClass = MRB_PROC_TARGET_CLASS(mrb->c->ci->proc);
  if (!targetClass) {
    mrb_value exc = mrb_exc_new_lit(mrb, E_TYPE_ERROR, "no target class or module");
    mrb_exc_raise(mrb, exc);
  }
  return mrb_obj_value(targetClass);
}

LIGHTSTORM_INLINE mrb_value ls_create_method(mrb_state *mrb, mrb_func_t func) {
  struct RProc *proc = mrb_proc_new_cfunc(mrb, func);
  proc->flags |= MRB_PROC_SCOPE;
  proc->flags |= MRB_PROC_STRICT;
  struct RClass *targetClass = mrb->c->ci->proc ? MRB_PROC_TARGET_CLASS(mrb->c->ci->proc) : NULL;
  if (!targetClass)
    targetClass = mrb->object_class;
  MRB_PROC_SET_TARGET_CLASS(proc, targetClass);
  proc->upper = mrb->c->ci->proc;
  return mrb_obj_value(proc);
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

LIGHTSTORM_INLINE mrb_value ls_hash_add(mrb_state *mrb, mrb_value hash, mrb_int size, ...) {
  mrb_ensure_hash_type(mrb, hash);
  mrb_value argv[size];
  va_list args;
  va_start(args, size);
  for (mrb_int i = 0; i < size; i++) {
    argv[i] = va_arg(args, mrb_value);
  }
  va_end(args);

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
  proc->flags |= MRB_PROC_STRICT;
  proc->upper = upperProc;
  mrb_vm_ci_proc_set(mrb->c->ci, proc);
  // Since we do not pop/push callinfo (stack frame), we need to set and restore the right `self`
  mrb_value old_self = ls_load_self_value(mrb);
  ls_store_self_value(mrb, receiver);
  mrb_value ret = func(mrb, receiver);
  ls_store_self_value(mrb, old_self);
  mrb_vm_ci_proc_set(mrb->c->ci, upperProc);
  return ret;
}

LIGHTSTORM_INLINE mrb_value ls_send_argv(mrb_state *mrb, mrb_value recv, mrb_sym name, mrb_int argc,
                                         mrb_value *argv) {
  struct mrb_jmpbuf *prev_jmp = mrb->jmp;
  struct mrb_jmpbuf c_jmp;
  mrb_value ret = mrb_nil_value();
  mrb_value old_self = ls_load_self_value(mrb);
  if (mrb_nil_p(old_self)) {
    // In certain cases (`new` followed by `initialize`) mruby pushes the stack frame but doesn't
    // preserve `self`
    // In this case we propagate self from the parent stack frame
    // TODO: add caching
    mrb_sym new_sym = mrb_intern(mrb, "new", 3);
    if (new_sym == name) {
      //         parent stack frame
      old_self = (mrb->c->ci - 1)->stack[0];
      ls_store_self_value(mrb, old_self);
    }
  }
  MRB_TRY(&c_jmp) {
    mrb->jmp = &c_jmp;
    ret = mrb_funcall_argv(mrb, recv, name, argc, argv);
    mrb->jmp = prev_jmp;
  }
  MRB_CATCH(&c_jmp) {
    mrb_print_error(mrb);
    abort();
  }
  MRB_END_EXC(&c_jmp);
  ls_store_self_value(mrb, old_self);
  return ret;
}

LIGHTSTORM_INLINE mrb_value ls_send(mrb_state *mrb, mrb_value recv, mrb_sym name, mrb_int argc,
                                    ...) {
  mrb_value argv[argc];
  va_list args;
  va_start(args, argc);
  for (mrb_int i = 0; i < argc; i++) {
    argv[i] = va_arg(args, mrb_value);
  }
  va_end(args);
  return ls_send_argv(mrb, recv, name, argc, argv);
}

LIGHTSTORM_INLINE mrb_value ls_sendv(mrb_state *mrb, mrb_value recv, mrb_sym name,
                                     mrb_value array) {
  assert(mrb_type(array) == MRB_TT_ARRAY);
  mrb_int argc = RARRAY_LEN(array);
  mrb_value *argv = RARRAY_PTR(array);
  return ls_send_argv(mrb, recv, name, argc, argv);
}

LIGHTSTORM_INLINE mrb_value ls_vm_define_class(mrb_state *mrb, mrb_value base, mrb_value super,
                                               mrb_sym id) {
  if (mrb_nil_p(base)) {
    // TODO: Fix target class
    struct RClass *baseclass = mrb->c->ci->proc ? MRB_PROC_TARGET_CLASS(mrb->c->ci->proc) : NULL;
    if (!baseclass)
      baseclass = mrb->object_class;
    base = mrb_obj_value(baseclass);
  }
  struct RClass *c = mrb_vm_define_class(mrb, base, super, id);
  return mrb_obj_value(c);
}

LIGHTSTORM_INLINE mrb_value ls_apost(mrb_state *mrb, mrb_value array, mrb_int pre, mrb_int post) {
  int len, idx;
  if (!mrb_array_p(array)) {
    array = mrb_ary_new_from_values(mrb, 1, &array);
  }
  mrb_value retVal = mrb_ary_new_capa(mrb, post + 1);
  struct RArray *ary = mrb_ary_ptr(array);
  len = (int)ARY_LEN(ary);
  if (len > pre + post) {
    mrb_value head = mrb_ary_new_from_values(mrb, len - pre - post, ARY_PTR(ary) + pre);
    mrb_ary_push(mrb, retVal, head);
    while (post--) {
      mrb_ary_push(mrb, retVal, ARY_PTR(ary)[len - post - 1]);
    }
  } else {
    mrb_value head = mrb_ary_new_capa(mrb, 0);
    mrb_ary_push(mrb, retVal, head);
    for (idx = 0; idx + pre < len; idx++) {
      mrb_ary_push(mrb, retVal, ARY_PTR(ary)[pre + idx]);
    }
    while (idx < post) {
      mrb_ary_push(mrb, retVal, mrb_nil_value());
      idx++;
    }
  }
  return retVal;
}

LIGHTSTORM_INLINE mrb_value ls_hash_merge(mrb_state *mrb, mrb_value h1, mrb_value h2) {
  mrb_ensure_hash_type(mrb, h1);
  mrb_hash_merge(mrb, h1, h2);
  return h1;
}

// coming from mruby/src/value_array.h
LIGHTSTORM_INLINE void value_move(mrb_value *s1, const mrb_value *s2, size_t n) {
  if (n == 0)
    return;
  if (s1 > s2 && s1 < s2 + n) {
    s1 += n;
    s2 += n;
    while (n-- > 0) {
      *--s1 = *--s2;
    }
  } else if (s1 != s2) {
    while (n-- > 0) {
      *s1++ = *s2++;
    }
  } else {
    /* nothing to do. */
  }
}

LIGHTSTORM_INLINE void stack_clear(mrb_value *from, size_t count) {
#ifdef MRB_NAN_BOXING
  while (count-- > 0) {
    SET_NIL_VALUE(*from);
    from++;
  }
#else
  memset(from, 0, sizeof(mrb_value) * count);
#endif
}

static void argnum_error(mrb_state *mrb, mrb_int num) {
  mrb_value exc;
  mrb_value str;
  mrb_int argc = mrb->c->ci->argc;

  if (argc < 0) {
    mrb_value args = mrb->c->ci->stack[1];
    if (mrb_array_p(args)) {
      argc = RARRAY_LEN(args);
    }
  }
  if (mrb->c->ci->mid) {
    str =
        mrb_format(mrb, "'%n': wrong number of arguments (%i for %i)", mrb->c->ci->mid, argc, num);
  } else {
    str = mrb_format(mrb, "wrong number of arguments (%i for %i)", argc, num);
  }
  exc = mrb_exc_new_str(mrb, E_ARGUMENT_ERROR, str);
  mrb_exc_set(mrb, exc);
}

LIGHTSTORM_INLINE mrb_value ls_enter(mrb_state *mrb, mrb_int requiredArgs) {
  // Copied almost verbatim from firestorm_runtime.c and mruby/vm.c
  mrb_int m1 = requiredArgs;
  mrb_int o = 0;
  mrb_int r = 0;
  mrb_int m2 = 0;
  mrb_int kd = 0;
  /* unused
  int b  = MRB_ASPEC_BLOCK(a);
  */
  mrb_int argc = mrb->c->ci->argc;
  mrb_value *argv = mrb->c->ci->stack + 1;
  mrb_value *const argv0 = argv;
  mrb_int const len = m1 + o + r + m2;
  mrb_int const blk_pos = len + kd + 1;
  mrb_value *blk = &argv[argc < 0 ? 1 : argc];
  mrb_int kargs = kd;

  /* arguments is passed with Array */
  if (argc < 0) {
    struct RArray *ary = mrb_ary_ptr(mrb->c->ci->stack[1]);
    argv = ARY_PTR(ary);
    argc = (int)ARY_LEN(ary);
    mrb_gc_protect(mrb, mrb->c->ci->stack[1]);
  }

  /* strict argument check */
  if (mrb->c->ci->proc && MRB_PROC_STRICT_P(mrb->c->ci->proc)) {
    if (argc < m1 + m2 || (argc > len + kd)) {
      // argnum_error sets mrb->exc
      argnum_error(mrb, m1 + m2);
      mrb_exc_raise(mrb, mrb_obj_value(mrb->exc));
    }
  }
  /* extract first argument array to arguments */
  else if (len > 1 && argc == 1 && mrb_array_p(argv[0])) {
    mrb_gc_protect(mrb, argv[0]);
    argc = (int)RARRAY_LEN(argv[0]);
    argv = RARRAY_PTR(argv[0]);
  }

  /* no rest arguments */
  if (argc - kargs < len) {
    mrb_int mlen = m2;
    if (argc < m1 + m2) {
      mlen = m1 < argc ? argc - m1 : 0;
    }
    mrb->c->ci->stack[blk_pos] = *blk; /* move block */

    /* copy mandatory and optional arguments */
    if (argv0 != argv && argv) {
      value_move(&mrb->c->ci->stack[1], argv, argc - mlen); /* m1 + o */
    }
    if (argc < m1) {
      stack_clear(&mrb->c->ci->stack[argc + 1], m1 - argc);
    }
  } else {
    if (argv0 != argv) {
      mrb->c->ci->stack[blk_pos] = *blk; /* move block */
      value_move(&mrb->c->ci->stack[1], argv, m1 + o);
    }
    if (argv0 == argv) {
      mrb->c->ci->stack[blk_pos] = *blk; /* move block */
    }
  }

  mrb->c->ci->argc = (int16_t)(len + kd);
  if (mrb->c->ci->argc == 0) {
    // Once in a while GC kicks in and clears everything between `mrb->c->ci->stack + argc` and
    // `mrb->c->stend` thus clearing `self` if argc is zero
    mrb->c->ci->argc = 1;
  }
  return mrb_nil_value();
}
