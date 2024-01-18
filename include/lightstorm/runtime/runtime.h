#pragma once

#include <mruby.h>
#include <mruby/variable.h>

#define LIGHTSTORM_INLINE __attribute__((always_inline)) inline
// EmitC uses `bool` for conditional branch predicates
typedef int bool;

LIGHTSTORM_INLINE mrb_value ls_load_i(mrb_state *mrb, int64_t i) {
  return mrb_fixnum_value(i);
}

LIGHTSTORM_INLINE mrb_value ls_load_nil_value(mrb_state *mrb) {
  return mrb_nil_value();
}
LIGHTSTORM_INLINE mrb_value ls_load_self_value(mrb_state *mrb) {
  return mrb->c->ci->stack[0];
}
LIGHTSTORM_INLINE mrb_value ls_load_true_value(mrb_state *mrb) {
  return mrb_true_value();
}
LIGHTSTORM_INLINE mrb_value ls_load_false_value(mrb_state *mrb) {
  return mrb_false_value();
}

#define ls_send_0(mrb, recv, name, argc) ls_send(mrb, recv, name, argc, NULL)
#define ls_send_1(mrb, recv, name, argc, ...) ls_send(mrb, recv, name, argc, __VA_ARGS__)
#define ls_send_2(mrb, recv, name, argc, ...) ls_send(mrb, recv, name, argc, __VA_ARGS__)
#define ls_send_3(mrb, recv, name, argc, ...) ls_send(mrb, recv, name, argc, __VA_ARGS__)
#define ls_send_4(mrb, recv, name, argc, ...) ls_send(mrb, recv, name, argc, __VA_ARGS__)
#define ls_send_5(mrb, recv, name, argc, ...) ls_send(mrb, recv, name, argc, __VA_ARGS__)
#define ls_send_6(mrb, recv, name, argc, ...) ls_send(mrb, recv, name, argc, __VA_ARGS__)
#define ls_send_7(mrb, recv, name, argc, ...) ls_send(mrb, recv, name, argc, __VA_ARGS__)
#define ls_send_8(mrb, recv, name, argc, ...) ls_send(mrb, recv, name, argc, __VA_ARGS__)
#define ls_send_9(mrb, recv, name, argc, ...) ls_send(mrb, recv, name, argc, __VA_ARGS__)

mrb_value ls_send(mrb_state *mrb, mrb_value recv, mrb_sym name, mrb_int argc, ...);

#define ls_hash_0(mrb, size) ls_hash(mrb, size, NULL)
#define ls_hash_1(mrb, size, ...) ls_hash(mrb, size, __VA_ARGS__)
#define ls_hash_2(mrb, size, ...) ls_hash(mrb, size, __VA_ARGS__)
#define ls_hash_3(mrb, size, ...) ls_hash(mrb, size, __VA_ARGS__)
#define ls_hash_4(mrb, size, ...) ls_hash(mrb, size, __VA_ARGS__)
#define ls_hash_5(mrb, size, ...) ls_hash(mrb, size, __VA_ARGS__)
#define ls_hash_6(mrb, size, ...) ls_hash(mrb, size, __VA_ARGS__)
#define ls_hash_7(mrb, size, ...) ls_hash(mrb, size, __VA_ARGS__)
#define ls_hash_8(mrb, size, ...) ls_hash(mrb, size, __VA_ARGS__)
#define ls_hash_9(mrb, size, ...) ls_hash(mrb, size, __VA_ARGS__)

mrb_value ls_compare_gt(mrb_state *mrb, mrb_value lhs, mrb_value rhs);
mrb_value ls_compare_ge(mrb_state *mrb, mrb_value lhs, mrb_value rhs);
mrb_value ls_compare_lt(mrb_state *mrb, mrb_value lhs, mrb_value rhs);
mrb_value ls_compare_le(mrb_state *mrb, mrb_value lhs, mrb_value rhs);
mrb_value ls_compare_eq(mrb_state *mrb, mrb_value lhs, mrb_value rhs);

LIGHTSTORM_INLINE int ls_predicate_is_true(mrb_value value) {
  return mrb_true_p(value);
}
LIGHTSTORM_INLINE int ls_predicate_is_false(mrb_value value) {
  return mrb_false_p(value);
}
LIGHTSTORM_INLINE int ls_predicate_is_nil(mrb_value value) {
  return mrb_nil_p(value);
}

mrb_value ls_arith_add(mrb_state *mrb, mrb_value lhs, mrb_value rhs);
mrb_value ls_arith_sub(mrb_state *mrb, mrb_value lhs, mrb_value rhs);
mrb_value ls_arith_mul(mrb_state *mrb, mrb_value lhs, mrb_value rhs);
mrb_value ls_arith_div(mrb_state *mrb, mrb_value lhs, mrb_value rhs);

LIGHTSTORM_INLINE mrb_value ls_load_local_variable(mrb_state *mrb, int64_t idx) {
  return mrb->c->ci->stack[idx];
}
LIGHTSTORM_INLINE mrb_value ls_load_sym(mrb_state *mrb, mrb_sym sym) {
  return mrb_symbol_value(sym);
}
LIGHTSTORM_INLINE mrb_value ls_load_singleton_class(mrb_state *mrb, mrb_value target) {
  return mrb_singleton_class(mrb, target);
}

mrb_value ls_load_target_class_value(mrb_state *mrb);
mrb_value ls_create_method(mrb_state *mrb, mrb_func_t func);
mrb_value ls_define_method(mrb_state *mrb, mrb_value target, mrb_value method, mrb_sym mid);

mrb_value ls_array(mrb_state *mrb, mrb_int size, ...);
mrb_value ls_hash(mrb_state *mrb, mrb_int size, ...);

mrb_value ls_define_module(mrb_state *mrb, mrb_value target, mrb_sym sym);

mrb_value ls_exec(mrb_state *mrb, mrb_value receiver, mrb_func_t func);

LIGHTSTORM_INLINE mrb_value ls_get_const(mrb_state *mrb, mrb_sym sym) {
  return mrb_vm_const_get(mrb, sym);
}

// Temporary prototypes
static mrb_value _irep_0(mrb_state *, mrb_value);
static mrb_value _irep_1(mrb_state *, mrb_value);
static mrb_value _irep_2(mrb_state *, mrb_value);
static mrb_value _irep_3(mrb_state *, mrb_value);
static mrb_value _irep_4(mrb_state *, mrb_value);
static mrb_value _irep_5(mrb_state *, mrb_value);
static mrb_value _irep_6(mrb_state *, mrb_value);
static mrb_value _irep_7(mrb_state *, mrb_value);
static mrb_value _irep_8(mrb_state *, mrb_value);
static mrb_value _irep_9(mrb_state *, mrb_value);
static mrb_value _irep_10(mrb_state *, mrb_value);
