#include "converter.h"
#include "lightstorm/dialect/rite.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mruby/opcode.h>
#include <mruby/proc.h>
#include <unordered_set>

using namespace lightstorm;

extern "C" {
const char *mrb_debug_get_filename(mrb_state *mrb, const mrb_irep *irep, uint32_t pc);
int32_t mrb_debug_get_line(mrb_state *mrb, const mrb_irep *irep, uint32_t pc);
void mrb_codedump_all(mrb_state *, struct RProc *);
}

const char *fs_opcode_name(mrb_code code) {
  switch (code) {
#define OPCODE(x, _)                                                                               \
  case OP_##x:                                                                                     \
    return "OP_" #x;
#include "mruby/ops.h"
#undef OPCODE
  default:
    return "unknown opcode?";
  }
  return "unknown opcode?";
}

struct Regs {
  uint32_t a;
  uint32_t b;
  uint32_t c;
};

static void populatePrototypes(mlir::MLIRContext &context,
                               std::unordered_map<const mrb_irep *, mlir::func::FuncOp> &functions,
                               mrb_state *mrb, const mrb_irep *irep) {
  const char *filename = mrb_debug_get_filename(mrb, irep, 0);
  auto line = mrb_debug_get_line(mrb, irep, 0);
  auto functionLocation = mlir::FileLineColLoc::get(&context, filename, line, 0);

  mlir::Type mrb_state_t(rite::mrb_stateType::get(&context));
  mlir::Type mrb_value_t(rite::mrb_valueType::get(&context));

  mlir::OpBuilder builder(&context);
  auto functionName = "_irep_" + std::to_string(functions.size());

  auto functionType = builder.getFunctionType({ mrb_state_t, mrb_value_t }, { mrb_value_t });
  auto function = mlir::func::FuncOp::create(functionLocation, functionName, functionType);
  function.setVisibility(mlir::SymbolTable::Visibility::Private);
  functions[irep] = function;
  for (size_t i = 0; i < irep->rlen; i++) {
    populatePrototypes(context, functions, mrb, irep->reps[i]);
  }
}

static void createBody(mlir::MLIRContext &context, mrb_state *mrb, mlir::func::FuncOp func,
                       const mrb_irep *irep) {
  const char *filename = mrb_debug_get_filename(mrb, irep, 0);
  auto line = mrb_debug_get_line(mrb, irep, 0);
  mlir::Type mrb_value_t(rite::mrb_valueType::get(&context));

  auto functionLocation = mlir::FileLineColLoc::get(&context, filename, line, 0);
  auto entryBlock = func.addEntryBlock();

  auto state = func.getArgument(0);

  mlir::OpBuilder builder(&context);
  builder.setInsertionPointToEnd(entryBlock);

  auto symbol = [&](mrb_sym sym) {
    return rite::mrb_symAttr::get(&context, mrb_sym_name(mrb, sym));
  };

  std::unordered_map<int64_t, rite::VirtualRegisterOp> vregs;
  auto vreg = [&](int64_t reg) {
    if (!vregs.contains(reg)) {
      mlir::OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(entryBlock);
      vregs[reg] = builder.create<rite::VirtualRegisterOp>(
          functionLocation, rite::mrb_value_ptrType::get(&context), builder.getIndexAttr(reg));
    }
    return vregs[reg];
  };

  auto store = [&](int64_t reg, mlir::Value value) {
    auto r = vreg(reg);
    builder.create<rite::StoreOp>(functionLocation, r, value);
  };
  auto load = [&](int64_t reg) {
    auto r = vreg(reg);
    return builder.create<rite::LoadOp>(functionLocation, mrb_value_t, r);
  };

  // Potential jump targets
  std::unordered_map<int64_t, mlir::Operation *> addressMapping;
  // Real jump targets
  std::unordered_map<mlir::Operation *, std::vector<int64_t>> jumpTargets;

  // Just a temp instruction for construction of CFG
  auto tempRegister = vreg(0);

  const mrb_code *sp = irep->iseq;
  for (uint16_t pc_offset = 0; pc_offset < irep->ilen; pc_offset++) {
    const mrb_code *pc_base = (irep->iseq + pc_offset);
    const mrb_code *pc = pc_base;
    line = mrb_debug_get_line(mrb, irep, pc - irep->iseq);
    auto location = mlir::FileLineColLoc::get(&context, filename, line, pc_offset);

    addressMapping[pc_offset] = &entryBlock->back();

    auto address = [&]() { return builder.getIndexAttr(pc_offset); };

    Regs regs{};
    auto opcode = (mrb_insn)*pc;
    pc++;
    switch (opcode) {
    case OP_NOP:
    case OP_STOP:
      // NOOP
      break;

    case OP_LOADSELF: {
      // OPCODE(LOADSELF,   B)        /* R(a) = self */
      regs.a = READ_B();
      vreg(regs.a);
      auto def = builder.create<rite::LoadSelfOp>(location, mrb_value_t, address(), state);
      store(regs.a, def);
    } break;

    case OP_LOADI__1:
    case OP_LOADI_0:
    case OP_LOADI_1:
    case OP_LOADI_2:
    case OP_LOADI_3:
    case OP_LOADI_4:
    case OP_LOADI_5:
    case OP_LOADI_6:
    case OP_LOADI_7: {
      // OPCODE(LOADI_x,   B)        /* R(a) = mrb_int(x) */
      regs.a = READ_B();
      int64_t i = opcode - OP_LOADI__1 - 1;
      auto value = builder.create<mlir::arith::ConstantOp>(location, builder.getI64IntegerAttr(i));
      auto def = builder.create<rite::LoadIOp>(location, mrb_value_t, address(), state, value);
      store(regs.a, def);
    } break;

    case OP_LOADI: {
      // OPCODE(LOADI,      BB)       /* R(a) = mrb_int(b) */
      regs.a = READ_B();
      regs.b = READ_B();
      auto value = builder.create<mlir::arith::ConstantOp>(
          location, builder.getI64IntegerAttr(int64_t(regs.b)));
      auto def = builder.create<rite::LoadIOp>(location, mrb_value_t, address(), state, value);
      store(regs.a, def);
    } break;

    case OP_LOADNIL: {
      // OPCODE(LOADNIL,   B)        /* R(a) = nil */
      regs.a = READ_B();
      auto def = builder.create<rite::LoadNilOp>(location, mrb_value_t, address(), state);
      store(regs.a, def);
    } break;

    case OP_SEND: {
      // OPCODE(SEND,       BBB)      /* R(a) = call(R(a),Syms(b),R(a+1),...,R(a+c)) */
      regs.a = READ_B();
      regs.b = READ_B();
      regs.c = READ_B();
      std::vector<int64_t> usesAttr;
      std::vector<mlir::Value> argv; //(regs.c, phony);
      for (auto i = regs.a; i < regs.a + regs.c; i++) {
        argv.push_back(load(i + 1));
      }
      usesAttr.push_back(regs.a);
      for (auto i = regs.a; i < regs.a + regs.c; i++) {
        usesAttr.push_back(i + 1);
      }
      if (regs.c == 127) {
        // remove all, but preserve the receiver
        argv.clear();
        usesAttr.clear();
        // receiver
        usesAttr.push_back(regs.a);
        // actual argv packed in an array
        usesAttr.push_back(regs.a + 1);
      }
      auto def = builder.create<rite::SendOp>(location,
                                              mrb_value_t,
                                              address(),
                                              state,
                                              load(regs.a),
                                              symbol(irep->syms[regs.b]),
                                              builder.getI64IntegerAttr(regs.c),
                                              argv);
      store(regs.a, def);
    } break;

    case OP_RETURN: {
      // OPCODE(RETURN,     B)        /* return R(a) (normal) */
      regs.a = READ_B();
      auto val = load(regs.a);
      builder.create<rite::ReturnOp>(location, mrb_value_t, address(), state, val);
    } break;

#pragma mark - Comparisons

    case OP_GE: {
      // OPCODE(GE,         B)        /* R(a) = R(a)>=R(a+1) */
      regs.a = READ_B();
      auto def = builder.create<rite::GeOp>(
          location, mrb_value_t, address(), state, load(regs.a), load(regs.a + 1));
      store(regs.a, def);
    } break;

    case OP_LT: {
      // OPCODE(LT,         B)        /* R(a) = R(a)<R(a+1) */
      regs.a = READ_B();
      auto def = builder.create<rite::LtOp>(
          location, mrb_value_t, address(), state, load(regs.a), load(regs.a + 1));
      store(regs.a, def);
    } break;

    case OP_LE: {
      // OPCODE(LE,         B)        /* R(a) = R(a)<=R(a+1) */
      regs.a = READ_B();
      auto def = builder.create<rite::LeOp>(
          location, mrb_value_t, address(), state, load(regs.a), load(regs.a + 1));
      store(regs.a, def);
    } break;
    case OP_EQ: {
      // OPCODE(EQ,         B)        /* R(a) = R(a)==R(a+1) */
      regs.a = READ_B();
      auto def = builder.create<rite::EqOp>(
          location, mrb_value_t, address(), state, load(regs.a), load(regs.a + 1));
      store(regs.a, def);
    } break;

    case OP_GT: {
      // OPCODE(GT,         B)        /* R(a) = R(a)>R(a+1) */
      regs.a = READ_B();
      auto def = builder.create<rite::GtOp>(
          location, mrb_value_t, address(), state, load(regs.a), load(regs.a + 1));
      store(regs.a, def);
    } break;

#pragma mark - Jumps

    case OP_JMP: {
      // OPCODE(JMP,        S)        /* pc+=a */
      regs.a = READ_S();
      uint32_t target = (pc - irep->iseq) + (int16_t)regs.a;
      // inserting a temp unused variable just to attach the target to something
      jumpTargets[load(0)] = { target };
    } break;

    // TODO: should just forbid this construct?
    case OP_JMPUW: {
      // OPCODE(JMPUW,      S)        /* unwind_and_jump_to(a) */
      regs.a = READ_S();
      uint32_t target = (pc - irep->iseq) + (int16_t)regs.a;
      // inserting a temp unused variable just to attach the target to something
      jumpTargets[load(0)] = { target };
    } break;

    case OP_JMPNOT: {
      // OPCODE(JMPNOT,     BS)       /* if !R(a) pc+=b */
      regs.a = READ_B();
      regs.b = READ_S();
      uint32_t target = (pc - irep->iseq) + (int16_t)regs.b;
      auto next_address = (uint16_t)(pc_offset + pc - pc_base);
      auto pred = builder.create<rite::BranchPredicateOp>(
          location, builder.getI1Type(), load(regs.a), rite::BranchPredicate::bp_false);
      jumpTargets[pred] = { target, next_address };
    } break;

    case OP_JMPIF: {
      // OPCODE(JMPIF,      BS)       /* if R(a) pc+=b */
      regs.a = READ_B();
      regs.b = READ_S();
      uint32_t target = (pc - irep->iseq) + (int16_t)regs.b;
      auto next_address = (uint16_t)(pc_offset + pc - pc_base);
      auto pred = builder.create<rite::BranchPredicateOp>(
          location, builder.getI1Type(), load(regs.a), rite::BranchPredicate::bp_true);
      jumpTargets[pred] = { target, next_address };
    } break;

    case OP_JMPNIL: {
      // OPCODE(JMPNIL,     BS)       /* if R(a)==nil pc+=b */
      regs.a = READ_B();
      regs.b = READ_S();
      uint32_t target = (pc - irep->iseq) + (int16_t)regs.b;
      auto next_address = (uint16_t)(pc_offset + pc - pc_base);
      auto pred = builder.create<rite::BranchPredicateOp>(
          location, builder.getI1Type(), load(regs.a), rite::BranchPredicate::bp_nil);
      jumpTargets[pred] = { target, next_address };
    } break;

    default: {
      using namespace std::string_literals;
      auto msg = "Hit unsupported op: "s + fs_opcode_name(opcode);
      llvm_unreachable(msg.c_str());
    }
    }

    pc_offset += pc - pc_base - 1;
  }

  for (auto &[address, op] : addressMapping) {
    addressMapping[address] = op->getNextNode();
  }

  std::unordered_map<mlir::Operation *, mlir::Operation *> fallthroughJumps;

  // Split blocks
  for (auto &[op, targets] : jumpTargets) {
    std::vector<mlir::Operation *> opTargets;
    for (auto target : targets) {
      auto targetOp = addressMapping.at(target);
      assert(targetOp);
      auto previousOp = targetOp->getPrevNode();
      assert(previousOp);
      targetOp->getBlock()->splitBlock(targetOp);
      if (!jumpTargets.contains(previousOp)) {
        fallthroughJumps[previousOp] = targetOp;
      }
    }
  }

  for (auto &[op, targets] : jumpTargets) {
    builder.setInsertionPointAfter(op);
    if (llvm::isa<rite::BranchPredicateOp>(op)) {
      assert(op->getNumResults() == 1);
      assert(targets.size() == 2);
      builder.create<mlir::cf::CondBranchOp>(op->getLoc(),
                                             op->getResult(0),
                                             addressMapping.at(targets[0])->getBlock(),
                                             addressMapping.at(targets[1])->getBlock());
    } else {
      assert(targets.size() == 1);
      builder.create<mlir::cf::BranchOp>(op->getLoc(), addressMapping.at(targets[0])->getBlock());
    }
  }

  for (auto &[op, target] : fallthroughJumps) {
    builder.setInsertionPointAfter(op);
    builder.create<mlir::cf::BranchOp>(op->getLoc(), target->getBlock());
  }
  if (tempRegister->getUsers().empty()) {
    tempRegister->erase();
  }
}

mlir::ModuleOp lightstorm::convertProcToMLIR(mlir::MLIRContext &context, mrb_state *mrb,
                                             struct RProc *proc) {
  mrb_codedump_all(mrb, proc);

  const char *filename = mrb_debug_get_filename(mrb, proc->body.irep, 0);
  auto line = mrb_debug_get_line(mrb, proc->body.irep, 0);
  auto moduleLocation = mlir::FileLineColLoc::get(&context, filename, line, 0);

  std::unordered_map<const mrb_irep *, mlir::func::FuncOp> functions;
  populatePrototypes(context, functions, mrb, proc->body.irep);
  functions[proc->body.irep].setName("lightstorm_top");

  auto module = mlir::ModuleOp::create(moduleLocation, filename);

  for (auto &[irep, func] : functions) {
    createBody(context, mrb, func, irep);
    module.push_back(func);
  }

  return module;
}
