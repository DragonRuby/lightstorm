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
                       const mrb_irep *irep,
                       const std::unordered_map<const mrb_irep *, mlir::func::FuncOp> &functions) {
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

  for (int i = 0; i < irep->nlocals - 1; i++) {
    auto index = builder.create<mlir::arith::ConstantIndexOp>(functionLocation, i + 1);
    auto def =
        builder.create<rite::LoadLocalVariableOp>(functionLocation, mrb_value_t, state, index);
    store(i + 1, def);
  }

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

    Regs regs{};
    auto opcode = (mrb_insn)*pc;
    pc++;
    switch (opcode) {
    case OP_NOP:
    case OP_STOP:
      // NOOP
      break;

    case OP_MOVE: {
      // OPCODE(MOVE,       BB)       /* R(a) = R(b) */
      regs.a = READ_B();
      regs.b = READ_B();
      store(regs.a, load(regs.b));
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
      auto def = builder.create<rite::LoadIOp>(location, mrb_value_t, state, value);
      store(regs.a, def);
    } break;

    case OP_LOADI: {
      // OPCODE(LOADI,      BB)       /* R(a) = mrb_int(b) */
      regs.a = READ_B();
      regs.b = READ_B();
      auto value = builder.create<mlir::arith::ConstantOp>(
          location, builder.getI64IntegerAttr(int64_t(regs.b)));
      auto def = builder.create<rite::LoadIOp>(location, mrb_value_t, state, value);
      store(regs.a, def);
    } break;

    case OP_LOADINEG: {
      // OPCODE(LOADINEG,      BB)       /* R(a) = mrb_int(-b) */
      regs.a = READ_B();
      regs.b = READ_B();
      auto value = builder.create<mlir::arith::ConstantOp>(
          location, builder.getI64IntegerAttr(-int64_t(regs.b)));
      auto def = builder.create<rite::LoadIOp>(location, mrb_value_t, state, value);
      store(regs.a, def);
    } break;

    case OP_LOADI16: {
      // OPCODE(LOADI16,    BS)       /* R(a) = mrb_int(b) */
      regs.a = READ_B();
      regs.b = READ_S();
      int64_t val = (int16_t)regs.b;
      auto value =
          builder.create<mlir::arith::ConstantOp>(location, builder.getI64IntegerAttr(val));
      auto def = builder.create<rite::LoadIOp>(location, mrb_value_t, state, value);
      store(regs.a, def);
    } break;

    case OP_LOADI32: {
      // OPCODE(LOADI32,    BSS)      /* R(a) = mrb_int((b<<16)+c) */
      regs.a = READ_B();
      regs.b = READ_S();
      regs.c = READ_S();
      int64_t val = (int16_t)regs.b;
      val = (val << 16) + regs.c;
      auto value =
          builder.create<mlir::arith::ConstantOp>(location, builder.getI64IntegerAttr(val));
      auto def = builder.create<rite::LoadIOp>(location, mrb_value_t, state, value);
      store(regs.a, def);
    } break;

    case OP_LOADNIL:
    case OP_LOADSELF:
    case OP_LOADT:
    case OP_LOADF: {
      regs.a = READ_B();
      rite::LoadValueKind kinds[] = { rite::LoadValueKind::nil_value,
                                      rite::LoadValueKind::self_value,
                                      rite::LoadValueKind::true_value,
                                      rite::LoadValueKind::false_value };
      int64_t idx = opcode - OP_LOADNIL;
      auto def = builder.create<rite::LoadValueOp>(location, mrb_value_t, state, kinds[idx]);
      store(regs.a, def);
    } break;

    case OP_TCLASS: {
      // OPCODE(TCLASS,     B)        /* R(a) = target_class */
      regs.a = READ_B();
      auto def = builder.create<rite::LoadValueOp>(
          location, mrb_value_t, state, rite::LoadValueKind::target_class_value);
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
      auto mid = builder.create<rite::InternSymOp>(
          location, builder.getUI32IntegerAttr(0).getType(), state, symbol(irep->syms[regs.b]));
      auto argc =
          builder.create<mlir::arith::ConstantOp>(location, builder.getI64IntegerAttr(regs.c));
      auto def =
          builder.create<rite::SendOp>(location, mrb_value_t, state, load(regs.a), mid, argc, argv);
      store(regs.a, def);
    } break;

    case OP_ENTER: {
      // OPCODE(ENTER,      W)        /* arg setup according to flags (23=m5:o5:r1:m5:k5:d1:b1) */
      regs.a = READ_W();
      // Skipping for now
    } break;

    case OP_RETURN: {
      // OPCODE(RETURN,     B)        /* return R(a) (normal) */
      regs.a = READ_B();
      auto val = load(regs.a);
      builder.create<rite::ReturnOp>(location, mrb_value_t, state, val);
    } break;

    case OP_METHOD: {
      // OPCODE(METHOD,     BB)       /* R(a) = lambda(SEQ[b],L_METHOD) */
      regs.a = READ_B();
      regs.b = READ_B();
      const mrb_irep *ref = irep->reps[regs.b];
      auto funcOp = functions.at(ref);
      auto refAttr = mlir::FlatSymbolRefAttr::get(&context, funcOp.getName());
      auto def = builder.create<rite::MethodOp>(location, mrb_value_t, state, refAttr);
      store(regs.a, def);
    } break;

    case OP_DEF: {
      // OPCODE(DEF,        BB)       /* R(a).newmethod(Syms(b),R(a+1)) */
      regs.a = READ_B();
      regs.b = READ_B();
      auto mid = builder.create<rite::InternSymOp>(
          location, builder.getUI32IntegerAttr(0).getType(), state, symbol(irep->syms[regs.b]));
      builder.create<rite::DefOp>(
          location, mrb_value_t, state, load(regs.a), load(regs.a + 1), mid);
    } break;

    case OP_EQ:
    case OP_LT:
    case OP_LE:
    case OP_GT:
    case OP_GE: {
      regs.a = READ_B();
      rite::CmpKind kinds[] = { rite::CmpKind::eq,
                                rite::CmpKind::lt,
                                rite::CmpKind::le,
                                rite::CmpKind::gt,
                                rite::CmpKind::ge };
      int64_t idx = opcode - OP_EQ;
      auto def = builder.create<rite::CmpOp>(
          location, mrb_value_t, state, load(regs.a), load(regs.a + 1), kinds[idx]);
      store(regs.a, def);
    } break;

    // TODO: should just reject OP_JMPUW ?
    case OP_JMPUW:
    case OP_JMP: {
      regs.a = READ_S();
      uint32_t target = (pc - irep->iseq) + (int16_t)regs.a;
      // inserting a temp unused variable just to attach the target to something
      jumpTargets[load(0)] = { target };
    } break;

    case OP_JMPIF:
    case OP_JMPNOT:
    case OP_JMPNIL: {
      regs.a = READ_B();
      regs.b = READ_S();

      rite::BranchPredicateKind kinds[] = { rite::BranchPredicateKind::is_true,
                                            rite::BranchPredicateKind::is_false,
                                            rite::BranchPredicateKind::is_nil };

      int64_t idx = opcode - OP_JMPIF;

      uint32_t target = (pc - irep->iseq) + (int16_t)regs.b;
      auto next_address = (uint16_t)(pc_offset + pc - pc_base);
      auto pred = builder.create<rite::BranchPredicateOp>(
          location, builder.getI1Type(), load(regs.a), kinds[idx]);
      jumpTargets[pred] = { target, next_address };
    } break;

    case OP_ADD:
    case OP_ADDI:
    case OP_SUB:
    case OP_SUBI:
    case OP_MUL:
    case OP_DIV: {
      regs.a = READ_B();
      mlir::Value rhs{};
      if (opcode == OP_ADDI || opcode == OP_SUBI) {
        regs.b = READ_B();
        auto value = builder.create<mlir::arith::ConstantOp>(
            location, builder.getI64IntegerAttr(int64_t(regs.b)));
        rhs = builder.create<rite::LoadIOp>(location, mrb_value_t, state, value);
      } else {
        rhs = load(regs.a + 1);
      }

      rite::ArithKind kinds[] = {
        rite::ArithKind::add, rite::ArithKind::add, rite::ArithKind::sub,
        rite::ArithKind::sub, rite::ArithKind::mul, rite::ArithKind::div
      };

      int64_t idx = opcode - OP_ADD;
      auto def = builder.create<rite::ArithOp>(
          location, mrb_value_t, state, load(regs.a), rhs, kinds[idx]);
      store(regs.a, def);
    } break;

    default: {
      using namespace std::string_literals;
      auto msg = "Hit unsupported op: "s + fs_opcode_name(opcode);
      llvm_unreachable(msg.c_str());
    }
    }

    pc_offset += pc - pc_base - 1;
  }

  ///
  /// CFG construction
  ///

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
    createBody(context, mrb, func, irep, functions);
    module.push_back(func);
  }

  return module;
}
