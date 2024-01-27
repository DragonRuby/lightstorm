#include "converter.h"
#include "lightstorm/dialect/rite.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
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

static void frontend_error(mlir::FileLineColLoc &location, std::string_view msg) {
  llvm::errs() << location.getFilename().str() << ":" << location.getLine() << ": " << msg << "\n";
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

  // A set of virtual registers used for SSA generation
  // These should be destructed by mem2reg pass once the IR construction is complete
  std::unordered_map<int64_t, rite::VirtualRegisterOp> vregs;
  // Lazily generate a virtual register for load/store operations
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

  // Initialize local variables
  for (int i = 0; i < irep->nlocals - 1; i++) {
    auto index = builder.create<mlir::arith::ConstantIndexOp>(functionLocation, i + 1);
    auto def =
        builder.create<rite::LoadLocalVariableOp>(functionLocation, mrb_value_t, state, index);
    store(i + 1, def);
  }

  // To reconstruct CFG, each opcode goes into a separate basic block
  // A mapping 'address' -> 'block' is maintained separately to rewire the CFG once all bytecode
  // is processed
  std::unordered_map<int64_t, mlir::Block *> addressMapping;

  // There are three cases for jumps:
  //  - unconditional (OP_JMP) with a single jump target
  //  - conditional (OP_JMPIF, OP_JMPNOT, OP_JMPNIL) with two jump targets (if-then/else)
  //  - exceptions: this one is unsupported at the moment
  // We track them separately by mapping a basic block to its jump targets
  // The jump targets later used (in combination with the address mapping) to rewire CFG
  std::unordered_map<mlir::Block *, int64_t> unconditionalTargets;
  std::unordered_map<mlir::Block *, std::pair<int64_t, int64_t>> conditionalTargets;

  auto body = func.addBlock();
  builder.create<mlir::cf::BranchOp>(functionLocation, body);
  builder.setInsertionPointToStart(body);

  // Iterating over each (supported) bytecode instruction to convert it into MLIR operations
  for (uint16_t pc_offset = 0; pc_offset < irep->ilen; pc_offset++) {
    // By default, each basic block falls through unconditionally to the next basic block imitating
    // sequential bytecode execution.
    // In case of OP_RETURN no fallthrough happens, thus we shouldn't add CondBranch at the end of
    // the loop
    bool fallthrough = true;
    const mrb_code *pc_base = (irep->iseq + pc_offset);
    const mrb_code *pc = pc_base;
    line = mrb_debug_get_line(mrb, irep, pc - irep->iseq);
    auto location = mlir::FileLineColLoc::get(&context, filename, line, pc_offset);

    addressMapping[pc_offset] = body;

    auto symbol = [&](mrb_sym sym) {
      auto attr = rite::mrb_symAttr::get(&context, mrb_sym_name(mrb, sym));
      auto symType = rite::mrb_symType::get(&context);
      return builder.create<rite::InternSymOp>(location, symType, state, attr);
    };

    Regs regs{};
    auto opcode = (mrb_insn)*pc;
    pc++;
    switch (opcode) {
    case OP_NOP:
    case OP_STOP:
      // NOOP
      // still generates a basic block, but these are unreachable and will be eliminated in the end
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

    case OP_LOADL: {
      // OPCODE(LOADL,      BB)       /* R(a) = Pool(b) */
      regs.a = READ_B();
      regs.b = READ_B();
      auto poolValue = irep->pool[regs.b];
      switch (poolValue.tt) {
      case IREP_TT_INT32: {
        int64_t val = poolValue.u.i32;
        auto value =
            builder.create<mlir::arith::ConstantOp>(location, builder.getI64IntegerAttr(val));
        auto def = builder.create<rite::LoadIOp>(location, mrb_value_t, state, value);
        store(regs.a, def);
      } break;
      case IREP_TT_INT64: {
        int64_t val = poolValue.u.i64;
        auto value =
            builder.create<mlir::arith::ConstantOp>(location, builder.getI64IntegerAttr(val));
        auto def = builder.create<rite::LoadIOp>(location, mrb_value_t, state, value);
        store(regs.a, def);
      } break;
      case IREP_TT_FLOAT: {
        auto value = builder.create<mlir::arith::ConstantOp>(
            location, builder.getF64FloatAttr(poolValue.u.f));
        auto def = builder.create<rite::LoadFloatOp>(location, mrb_value_t, state, value);
        store(regs.a, def);
      } break;
      default:
        llvm_unreachable("should not happen (tt:string)");
        break;
      }
    } break;

    case OP_STRING: {
      // OPCODE(STRING,     BB)       /* R(a) = str_dup(Lit(b)) */
      regs.a = READ_B();
      regs.b = READ_B();
      uint32_t len = irep->pool[regs.b].tt >> 2;
      const char *str = irep->pool[regs.b].u.str;
      auto ui32t = builder.getUI32IntegerAttr(0).getType();
      auto lenCst =
          builder.create<mlir::arith::ConstantOp>(location, builder.getI64IntegerAttr(len));
      auto def = builder.create<rite::LoadStringOp>(
          location, mrb_value_t, state, builder.getStringAttr(llvm::StringRef(str, len)), lenCst);
      store(regs.a, def);
    } break;

    case OP_STRCAT: {
      // OPCODE(STRCAT,     B)        /* str_cat(R(a),R(a+1)) */
      regs.a = READ_B();
      builder.create<rite::StrCatOp>(location, mrb_value_t, state, load(regs.a), load(regs.a + 1));
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

    case OP_LOADSYM: {
      // OPCODE(LOADSYM,    BB)       /* R(a) = Syms(b) */
      regs.a = READ_B();
      regs.b = READ_B();
      auto sym = symbol(irep->syms[regs.b]);
      auto def = builder.create<rite::LoadSymOp>(location, mrb_value_t, state, sym);
      store(regs.a, def);
    } break;

    case OP_SCLASS: {
      // OPCODE(SCLASS,     B)        /* R(a) = R(a).singleton_class */
      regs.a = READ_B();
      auto def = builder.create<rite::SClassOp>(location, mrb_value_t, state, load(regs.a));
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
      auto mid = symbol(irep->syms[regs.b]);
      auto argc =
          builder.create<mlir::arith::ConstantOp>(location, builder.getI64IntegerAttr(regs.c));
      auto def =
          builder.create<rite::SendOp>(location, mrb_value_t, state, load(regs.a), mid, argc, argv);
      store(regs.a, def);
    } break;

    case OP_EXEC: {
      // OPCODE(EXEC,       BB)       /* R(a) = blockexec(R(a),SEQ[b]) */
      regs.a = READ_B();
      regs.b = READ_B();
      const mrb_irep *ref = irep->reps[regs.b];
      auto funcOp = functions.at(ref);
      auto refAttr = mlir::FlatSymbolRefAttr::get(&context, funcOp.getName());
      auto def = builder.create<rite::ExecOp>(location, mrb_value_t, state, load(regs.a), refAttr);
      store(regs.a, def);
    } break;

    case OP_ENTER: {
      // OPCODE(ENTER,      W)        /* arg setup according to flags (23=m5:o5:r1:m5:k5:d1:b1) */
      regs.a = READ_W();
      // Skipping for now
      mrb_int a = regs.a;
      mrb_int o = MRB_ASPEC_OPT(a);
      mrb_int r = MRB_ASPEC_REST(a);
      mrb_int m2 = MRB_ASPEC_POST(a);
      mrb_int kd = (MRB_ASPEC_KEY(a) > 0 || MRB_ASPEC_KDICT(a)) ? 1 : 0;
      mrb_int block = MRB_ASPEC_BLOCK(a);
      if (o != 0) {
        frontend_error(location, "Default arguments are not supported yet");
        exit(1);
      }
      if (r != 0 || m2 != 0) {
        frontend_error(location, "Variable arguments are not supported yet");
        exit(1);
      }
      if (kd != 0) {
        frontend_error(location, "Keyword arguments are not supported yet");
        exit(1);
      }
      if (block != 0) {
        frontend_error(location, "Block arguments are not supported yet");
        exit(1);
      }
    } break;

    case OP_RETURN: {
      // OPCODE(RETURN,     B)        /* return R(a) (normal) */
      regs.a = READ_B();
      auto val = load(regs.a);
      builder.create<rite::ReturnOp>(location, mrb_value_t, state, val);
      fallthrough = false;
    } break;

    case OP_MODULE: {
      // OPCODE(MODULE,     BB)       /* R(a) = newmodule(R(a),Syms(b)) */
      regs.a = READ_B();
      regs.b = READ_B();
      auto sym = symbol(irep->syms[regs.b]);
      auto def = builder.create<rite::ModuleOp>(location, mrb_value_t, state, load(regs.a), sym);
      store(regs.a, def);
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
      auto mid = symbol(irep->syms[regs.b]);
      builder.create<rite::DefOp>(
          location, mrb_value_t, state, load(regs.a), load(regs.a + 1), mid);
    } break;

    case OP_ALIAS: {
      // OPCODE(ALIAS,      BB)       /* alias_method(target_class,Syms(a),Syms(b)) */
      regs.a = READ_B();
      regs.b = READ_B();
      builder.create<rite::AliasOp>(
          location, mrb_value_t, state, symbol(irep->syms[regs.a]), symbol(irep->syms[regs.b]));
    } break;

    case OP_UNDEF: {
      // OPCODE(UNDEF,      B)        /* undef_method(target_class,Syms(a)) */
      regs.a = READ_B();
      builder.create<rite::UndefOp>(location, mrb_value_t, state, symbol(irep->syms[regs.a]));
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
      unconditionalTargets[body] = target;
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
      conditionalTargets[body] = { target, next_address };
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

    ///
    /// Array Ops
    ///
    case OP_ARRAY: {
      // OPCODE(ARRAY,      BB)       /* R(a) = ary_new(R(a),R(a+1)..R(a+b)) */
      regs.a = READ_B();
      regs.b = READ_B();
      std::vector<mlir::Value> argv;
      for (uint32_t i = regs.a; i < regs.a + regs.b; i++) {
        argv.push_back(load(i));
      }
      auto argc =
          builder.create<mlir::arith::ConstantOp>(location, builder.getI64IntegerAttr(regs.b));
      auto def = builder.create<rite::ArrayOp>(location, mrb_value_t, state, argc, argv);
      store(regs.a, def);
    } break;

    case OP_ARRAY2: {
      // Modeled as ArrayOp
      // OPCODE(ARRAY2,     BBB)      /* R(a) = ary_new(R(b),R(b+1)..R(b+c)) */
      regs.a = READ_B();
      regs.b = READ_B();
      regs.c = READ_B();
      std::vector<mlir::Value> argv;
      for (uint32_t i = regs.b; i < regs.b + regs.c; i++) {
        argv.push_back(load(i));
      }
      auto argc =
          builder.create<mlir::arith::ConstantOp>(location, builder.getI64IntegerAttr(regs.c));
      auto def = builder.create<rite::ArrayOp>(location, mrb_value_t, state, argc, argv);
      store(regs.a, def);
    } break;

    case OP_AREF: {
      // OPCODE(AREF,       BBB)      /* R(a) = R(b)[c] */
      regs.a = READ_B();
      regs.b = READ_B();
      regs.c = READ_B();
      auto index = builder.create<mlir::arith::ConstantOp>(location, builder.getIndexAttr(regs.c));
      auto def = builder.create<rite::ARefOp>(location, mrb_value_t, state, load(regs.b), index);
      store(regs.a, def);
    } break;

    case OP_APOST: {
      // OPCODE(APOST,      BBB)      /* *R(a),R(a+1)..R(a+c) = R(a)[b..] */
      regs.a = READ_B();
      regs.b = READ_B();
      regs.c = READ_B();
      auto pre = builder.create<mlir::arith::ConstantOp>(location, builder.getIndexAttr(regs.b));
      auto post = builder.create<mlir::arith::ConstantOp>(location, builder.getIndexAttr(regs.c));
      auto zero = builder.create<mlir::arith::ConstantOp>(location, builder.getIndexAttr(0));
      auto array =
          builder.create<rite::APostOp>(location, mrb_value_t, state, load(regs.a), pre, post);
      for (uint32_t i = 1; i < regs.c + 1; i++) {
        auto index = builder.create<mlir::arith::ConstantOp>(location, builder.getIndexAttr(i));
        auto def = builder.create<rite::ARefOp>(location, mrb_value_t, state, array, index);
        store(regs.a + i, def);
      }
      auto def = builder.create<rite::ARefOp>(location, mrb_value_t, state, array, zero);
      store(regs.a, def);
    } break;

    case OP_ARYPUSH: {
      // OPCODE(ARYPUSH,    B)        /* ary_push(R(a),R(a+1)) */
      regs.a = READ_B();
      builder.create<rite::ArrayPushOp>(
          location, mrb_value_t, state, load(regs.a), load(regs.a + 1));
    } break;

    case OP_ARYCAT: {
      // OPCODE(ARYCAT,     B)        /* ary_cat(R(a),R(a+1)) */
      regs.a = READ_B();
      auto def = builder.create<rite::ArrayCatOp>(
          location, mrb_value_t, state, load(regs.a), load(regs.a + 1));
      store(regs.a, def);
    } break;

    ///
    /// Hash Ops
    ///
    case OP_HASH: {
      // OPCODE(HASH,       BB)       /* R(a) = hash_new(R(a),R(a+1)..R(a+b*2-1)) */
      regs.a = READ_B();
      regs.b = READ_B();
      auto argc = regs.b * 2;
      std::vector<mlir::Value> argv;
      for (int i = regs.a; i < regs.a + argc; i++) {
        argv.push_back(load(i));
      }
      auto argcOp =
          builder.create<mlir::arith::ConstantOp>(location, builder.getI64IntegerAttr(argc));
      auto def = builder.create<rite::HashOp>(location, mrb_value_t, state, argcOp, argv);
      store(regs.a, def);
    } break;

    case OP_HASHCAT: {
      // OPCODE(HASHCAT,    B)        /* R(a) = hash_cat(R(a),R(a+1)) */
      regs.a = READ_B();
      auto def = builder.create<rite::HashCatOp>(
          location, mrb_value_t, state, load(regs.a), load(regs.a + 1));
      store(regs.a, def);
    } break;

    case OP_GETCONST: {
      // OPCODE(GETCONST,   BB)       /* R(a) = constget(Syms(b)) */
      regs.a = READ_B();
      regs.b = READ_B();
      auto sym = symbol(irep->syms[regs.b]);
      auto def = builder.create<rite::GetConstOp>(location, mrb_value_t, state, sym);
      store(regs.a, def);
    } break;

    case OP_SETCONST: {
      // OPCODE(SETCONST,   BB)       /* constset(Syms(b),R(a)) */
      regs.a = READ_B();
      regs.b = READ_B();
      auto sym = symbol(irep->syms[regs.b]);
      builder.create<rite::SetConstOp>(location, mrb_value_t, state, sym, load(regs.a));
    } break;

    case OP_GETGV: {
      // OPCODE(GETGV,      BB)       /* R(a) = getglobal(Syms(b)) */
      regs.a = READ_B();
      regs.b = READ_B();
      auto def =
          builder.create<rite::GetGVOp>(location, mrb_value_t, state, symbol(irep->syms[regs.b]));
      store(regs.a, def);
    } break;

    case OP_SETGV: {
      // OPCODE(SETGV,      BB)       /* setglobal(Syms(b), R(a)) */
      regs.a = READ_B();
      regs.b = READ_B();
      builder.create<rite::SetGVOp>(
          location, mrb_value_t, state, symbol(irep->syms[regs.b]), load(regs.a));
    } break;

    case OP_GETIV: {
      // OPCODE(GETIV,      BB)       /* R(a) = ivget(Syms(b)) */
      regs.a = READ_B();
      regs.b = READ_B();
      auto self = builder.create<rite::LoadValueOp>(
          location, mrb_value_t, state, rite::LoadValueKind::self_value);
      auto def = builder.create<rite::GetIVOp>(
          location, mrb_value_t, state, self, symbol(irep->syms[regs.b]));
      store(regs.a, def);
    } break;

    case OP_SETIV: {
      // OPCODE(SETIV,      BB)       /* ivset(Syms(b),R(a)) */
      regs.a = READ_B();
      regs.b = READ_B();
      auto self = builder.create<rite::LoadValueOp>(
          location, mrb_value_t, state, rite::LoadValueKind::self_value);
      builder.create<rite::SetIVOp>(
          location, mrb_value_t, state, self, symbol(irep->syms[regs.b]), load(regs.a));
    } break;

    case OP_GETCV: {
      // OPCODE(GETCV,      BB)       /* R(a) = cvget(Syms(b)) */
      regs.a = READ_B();
      regs.b = READ_B();
      auto def =
          builder.create<rite::GetCVOp>(location, mrb_value_t, state, symbol(irep->syms[regs.b]));
      store(regs.a, def);
    } break;

    case OP_SETCV: {
      // OPCODE(SETCV,      BB)       /* cvset(Syms(b),R(a)) */
      regs.a = READ_B();
      regs.b = READ_B();
      builder.create<rite::SetCVOp>(
          location, mrb_value_t, state, symbol(irep->syms[regs.b]), load(regs.a));
    } break;

    case OP_GETMCNST: {
      // OPCODE(GETMCNST,   BB)       /* R(a) = R(a)::Syms(b) */
      regs.a = READ_B();
      regs.b = READ_B();
      auto def = builder.create<rite::GetMCNSTOp>(
          location, mrb_value_t, state, load(regs.a), symbol(irep->syms[regs.b]));
      store(regs.a, def);
    } break;

    case OP_SETMCNST: {
      // OPCODE(SETMCNST,   BB)       /* R(a+1)::Syms(b) = R(a) */
      regs.a = READ_B();
      regs.b = READ_B();
      builder.create<rite::SetMCNSTOp>(
          location, mrb_value_t, state, load(regs.a + 1), symbol(irep->syms[regs.b]), load(regs.a));
    } break;

    case OP_CLASS: {
      // OPCODE(CLASS,      BB)       /* R(a) = newclass(R(a),Syms(b),R(a+1)) */
      regs.a = READ_B();
      regs.b = READ_B();
      auto def = builder.create<rite::ClassOp>(
          location, mrb_value_t, state, load(regs.a), load(regs.a + 1), symbol(irep->syms[regs.b]));
      store(regs.a, def);
    } break;

    case OP_RANGE_INC: {
      // OPCODE(RANGE_INC,  B)        /* R(a) = range_new(R(a),R(a+1),FALSE) */
      regs.a = READ_B();
      auto def = builder.create<rite::RangeIncOp>(
          location, mrb_value_t, state, load(regs.a), load(regs.a + 1));
      store(regs.a, def);
    } break;

    case OP_RANGE_EXC: {
      // OPCODE(RANGE_EXC,  B)        /* R(a) = range_new(R(a),R(a+1),TRUE) */
      regs.a = READ_B();
      auto def = builder.create<rite::RangeExcOp>(
          location, mrb_value_t, state, load(regs.a), load(regs.a + 1));
      store(regs.a, def);
    } break;

    case OP_LAMBDA: {
      regs.a = READ_B();
      regs.b = READ_B();
      frontend_error(location, "Lambdas are not supported");
    } break;

    case OP_LAMBDA16: {
      regs.a = READ_B();
      regs.b = READ_S();
      frontend_error(location, "Lambdas are not supported");
    } break;

    case OP_BLOCK: {
      regs.a = READ_B();
      regs.b = READ_B();
      frontend_error(location, "Blocks are not supported");
    } break;

    case OP_BLOCK16: {
      regs.a = READ_B();
      regs.b = READ_S();
      frontend_error(location, "Blocks are not supported");
    } break;

    case OP_RETURN_BLK: {
      regs.a = READ_B();
      frontend_error(location, "Blocks are not supported");
    } break;

    case OP_EXCEPT: {
      regs.a = READ_B();
      frontend_error(location, "Exceptions are not supported");
    } break;

    case OP_RESCUE: {
      regs.a = READ_B();
      regs.b = READ_B();
      frontend_error(location, "Exceptions are not supported");
    } break;

    case OP_RAISEIF: {
      regs.a = READ_B();
      frontend_error(location, "Exceptions are not supported");
    } break;

    default: {
      using namespace std::string_literals;
      auto msg = "Hit unsupported op: "s + fs_opcode_name(opcode);
      llvm_unreachable(msg.c_str());
    }
    }

    body = func.addBlock();
    if (fallthrough) {
      builder.create<mlir::cf::BranchOp>(functionLocation, body);
    }
    builder.setInsertionPointToStart(body);

    pc_offset += pc - pc_base - 1;
  }

  // Rewiring CFG

  // Basic block with unconditional targets must terminate with an unconditional fallthrough
  // cf::BranchOp Once we insert a new BranchOp with the right target, the old fallthrough BranchOp
  // should be removed
  for (auto &[block, target] : unconditionalTargets) {
    builder.setInsertionPointToEnd(block);
    auto &branch = block->back();
    assert(branch.hasTrait<mlir::OpTrait::IsTerminator>());
    assert(llvm::isa<mlir::cf::BranchOp>(branch));
    builder.create<mlir::cf::BranchOp>(branch.getLoc(), addressMapping.at(target));
    branch.erase();
  }
  // Basic block with conditional targets must terminate with an unconditional fallthrough
  // cf::BranchOp which is preceded by a rite::BranchPredicateOp
  // Once we insert CondBranchOp with the right targets, the old fallthrough BranchOp should be
  // removed
  for (auto &[block, targets] : conditionalTargets) {
    builder.setInsertionPointToEnd(block);
    auto &branch = block->back();
    assert(branch.hasTrait<mlir::OpTrait::IsTerminator>());
    auto predicate = branch.getPrevNode();
    assert(predicate);
    assert(llvm::isa<rite::BranchPredicateOp>(predicate));
    builder.create<mlir::cf::CondBranchOp>(predicate->getLoc(),
                                           predicate->getResult(0),
                                           addressMapping.at(targets.first),
                                           addressMapping.at(targets.second));
    branch.erase();
  }

  // Eliminating unreachable basic blocks (among other things)
  mlir::PassManager pipeline(&context);
  pipeline.addPass(mlir::createCanonicalizerPass());
  if (mlir::failed(pipeline.run(func))) {
    llvm::errs() << "Failed to canonicalize IR\n";
    exit(1);
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
  auto &top = functions[proc->body.irep];
  top.setName("lightstorm_top");
  top.setVisibility(mlir::SymbolTable::Visibility::Public);

  auto module = mlir::ModuleOp::create(moduleLocation, filename);

  for (auto &[irep, func] : functions) {
    createBody(context, mrb, func, irep, functions);
    module.push_back(func);
  }

  return module;
}
