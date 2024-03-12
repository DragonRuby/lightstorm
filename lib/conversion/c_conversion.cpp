#include "lightstorm/conversion/conversion.h"
#include "lightstorm/dialect/rite.h"
#include <mlir/Conversion/FuncToEmitC/FuncToEmitC.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/EmitC/IR/EmitC.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/Cpp/CppEmitter.h>
#include <mlir/Transforms/DialectConversion.h>
#include <sstream>
#include <unordered_set>

struct LightstormConversionContext {
  mlir::MLIRContext &context;
  mlir::TypeConverter &converter;
};

template <typename Op> struct LightstormConversionPattern : public mlir::ConversionPattern {
  LightstormConversionPattern(LightstormConversionContext &loweringContext)
      : conversionContext(loweringContext),
        mlir::ConversionPattern(loweringContext.converter, Op::getOperationName(), 1,
                                &loweringContext.context) {}

  mlir::LogicalResult matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
                                      mlir::ConversionPatternRewriter &rewriter) const final {
    auto typesafeOp = mlir::cast<Op>(op);
    std::vector<mlir::Type> operandTypes;
    for (auto t : op->getOperandTypes()) {
      operandTypes.push_back(conversionContext.converter.convertType(t));
    }
    mlir::Type resultType{};
    if (!op->getResultTypes().empty()) {
      assert(op->getResultTypes().size() == 1 && "Only supporting one return value");
      resultType = conversionContext.converter.convertType(op->getResultTypes().front());
    }
    return matchAndRewrite(typesafeOp, operands, operandTypes, resultType, rewriter);
  }

  virtual mlir::LogicalResult matchAndRewrite(Op op, llvm::ArrayRef<mlir::Value> operands,
                                              llvm::ArrayRef<mlir::Type> operandTypes,
                                              mlir::Type resultType,
                                              mlir::ConversionPatternRewriter &rewriter) const = 0;

  LightstormConversionContext &conversionContext;
};

static std::string cCompatibleSymName(const std::string &sym) {
  std::string s;
  std::stringstream ss(s);
  for (auto c : sym) {
    if (isalpha(c) || isnumber(c)) {
      ss << c;
      continue;
    }
    switch (c) {
      // clang-format off
      case '@': ss << "_at_"; break;
      case '!': ss << "_excl_"; break;
      case '<': ss << "_lt_"; break;
      case '>': ss << "_gt_"; break;
      case ':': ss << "_col_"; break;
      case ' ': ss << "_space_"; break;
      case '=': ss << "_eql_"; break;
      case '%': ss << "_percent_"; break;
      case '^': ss << "_caret_"; break;
      case '$': ss << "_dollar_"; break;
      case '&': ss << "_amp_"; break;
      case '?': ss << "_q_"; break;
      case '[': ss << "_sq_op_"; break;
      case ']': ss << "_sq_cl_"; break;
      case '-': ss << "_minus_"; break;
      case '+': ss << "_plus_"; break;
      case '*': ss << "_mul_"; break;
      default: ss << c; break;
      // clang-format on
    }
  }
  return ss.str();
}

namespace lightstorm_conversion {

static auto opaqueCallOp(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                         mlir::TypeRange resultType, mlir::StringRef callee,
                         mlir::ValueRange operands) {
  return rewriter.create<mlir::emitc::CallOpaqueOp>(
      loc, resultType, callee, mlir::ArrayAttr{}, mlir::ArrayAttr{}, operands);
}

template <typename Op> struct DirectOpConversion : public LightstormConversionPattern<Op> {
  explicit DirectOpConversion(LightstormConversionContext &conversionContext, std::string name)
      : LightstormConversionPattern<Op>(conversionContext), name(std::move(name)) {}

  mlir::LogicalResult matchAndRewrite(Op op, llvm::ArrayRef<mlir::Value> operands,
                                      llvm::ArrayRef<mlir::Type> operandTypes,
                                      mlir::Type resultType,
                                      mlir::ConversionPatternRewriter &rewriter) const final {
    auto newOp = opaqueCallOp(rewriter, op->getLoc(), resultType, name, operands);
    if (newOp.getNumResults()) {
      assert(newOp.getNumResults() == 1 && "Only supporting single return value");
      rewriter.replaceOp(op, newOp.getResult(0));
    } else {
      rewriter.eraseOp(op);
    }
    return mlir::success();
  }
  std::string name;
};

struct InternSymOpConversion : public LightstormConversionPattern<rite::InternSymOp> {
  explicit InternSymOpConversion(LightstormConversionContext &conversionContext)
      : LightstormConversionPattern(conversionContext) {}

  mlir::LogicalResult matchAndRewrite(rite::InternSymOp op, llvm::ArrayRef<mlir::Value> operands,
                                      llvm::ArrayRef<mlir::Type> operandTypes,
                                      mlir::Type resultType,
                                      mlir::ConversionPatternRewriter &rewriter) const final {
    auto symName = cCompatibleSymName(op.getMidAttr().getSymname());
    auto functionName = "_ls_sym_getter_" + symName;
    auto mrbSym = opaqueCallOp(rewriter, op->getLoc(), resultType, functionName, operands.front());
    rewriter.replaceOp(op, mrbSym.getResult(0));
    return mlir::success();
  }
};

struct LoadStringOpConversion : public LightstormConversionPattern<rite::LoadStringOp> {
  explicit LoadStringOpConversion(LightstormConversionContext &conversionContext)
      : LightstormConversionPattern(conversionContext) {}

  mlir::LogicalResult matchAndRewrite(rite::LoadStringOp op, llvm::ArrayRef<mlir::Value> operands,
                                      llvm::ArrayRef<mlir::Type> operandTypes,
                                      mlir::Type resultType,
                                      mlir::ConversionPatternRewriter &rewriter) const final {
    auto charType = mlir::emitc::OpaqueType::get(getContext(), "const char");
    auto charPtrType = mlir::emitc::PointerType::get(charType);
    auto str = '"' + op.getStr().str() + '"';
    auto strAttr = mlir::emitc::OpaqueAttr::get(getContext(), str);
    auto strVar = rewriter.create<mlir::emitc::ConstantOp>(op->getLoc(), charPtrType, strAttr);
    mlir::ValueRange newOperands{ operands.front(), strVar, operands.back() };
    auto call = opaqueCallOp(rewriter, op->getLoc(), resultType, "ls_load_string", newOperands);
    rewriter.replaceOp(op, call.getResult(0));
    return mlir::success();
  }
};

struct LocationOpConversion : public LightstormConversionPattern<rite::LocationOp> {
  explicit LocationOpConversion(LightstormConversionContext &conversionContext)
      : LightstormConversionPattern(conversionContext) {}

  mlir::LogicalResult matchAndRewrite(rite::LocationOp op, llvm::ArrayRef<mlir::Value> operands,
                                      llvm::ArrayRef<mlir::Type> operandTypes,
                                      mlir::Type resultType,
                                      mlir::ConversionPatternRewriter &rewriter) const final {
    std::string comment("// ");
    comment += op.getLocation().str();
    rewriter.create<mlir::emitc::VerbatimOp>(op->getLoc(), comment);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

static mlir::Value createMethodRef(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                                   mlir::FlatSymbolRefAttr method) {
  auto functionName = method.getAttr().str();
  auto prototypeAttr =
      rewriter.getStringAttr("mrb_value " + functionName + "(mrb_state *, mrb_value);");
  rewriter.create<mlir::emitc::VerbatimOp>(loc, prototypeAttr);
  auto mrb_func_t = mlir::emitc::OpaqueType::get(rewriter.getContext(), "mrb_func_t");
  return rewriter.create<mlir::emitc::ConstantOp>(
      loc, mrb_func_t, mlir::emitc::OpaqueAttr::get(rewriter.getContext(), functionName));
}

template <typename Op> struct MethodRefOpConversion : public LightstormConversionPattern<Op> {
  explicit MethodRefOpConversion(LightstormConversionContext &conversionContext, std::string name)
      : LightstormConversionPattern<Op>(conversionContext), name(std::move(name)) {}

  mlir::LogicalResult matchAndRewrite(Op op, llvm::ArrayRef<mlir::Value> operands,
                                      llvm::ArrayRef<mlir::Type> operandTypes,
                                      mlir::Type resultType,
                                      mlir::ConversionPatternRewriter &rewriter) const final {
    auto ref = createMethodRef(rewriter, op->getLoc(), op.getMethodAttr());
    std::vector<mlir::Value> argv{ operands };
    argv.push_back(ref);
    auto newOp = opaqueCallOp(rewriter, op->getLoc(), resultType, name, argv);
    rewriter.replaceOp(op, newOp.getResult(0));
    return mlir::success();
  }
  std::string name;
};

struct ReturnOpConversion : public LightstormConversionPattern<rite::ReturnOp> {
  explicit ReturnOpConversion(LightstormConversionContext &conversionContext)
      : LightstormConversionPattern(conversionContext) {}

  mlir::LogicalResult matchAndRewrite(rite::ReturnOp op, llvm::ArrayRef<mlir::Value> operands,
                                      llvm::ArrayRef<mlir::Type> operandTypes,
                                      mlir::Type resultType,
                                      mlir::ConversionPatternRewriter &rewriter) const final {
    if (operands.size() > 1) {
      // `mrb_state* mrb` is zeroth operand
      rewriter.create<mlir::func::ReturnOp>(op->getLoc(), operands.back());
    } else {
      // no value returned
      rewriter.create<mlir::func::ReturnOp>(op->getLoc());
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

template <typename Op> struct KindOpConversion : public LightstormConversionPattern<Op> {
  explicit KindOpConversion(LightstormConversionContext &conversionContext)
      : LightstormConversionPattern<Op>(conversionContext) {}

  static std::string kindName(rite::CmpKind kind) {
    return "ls_compare_" + rite::stringifyCmpKind(kind).str();
  }
  static std::string kindName(rite::BranchPredicateKind kind) {
    return "ls_predicate_" + rite::stringifyBranchPredicateKind(kind).str();
  }
  static std::string kindName(rite::ArithKind kind) {
    return "ls_arith_" + rite::stringifyArithKind(kind).str();
  }
  static std::string kindName(rite::LoadValueKind kind) {
    return "ls_load_" + rite::stringifyLoadValueKind(kind).str();
  }

  mlir::LogicalResult matchAndRewrite(Op op, llvm::ArrayRef<mlir::Value> operands,
                                      llvm::ArrayRef<mlir::Type> operandTypes,
                                      mlir::Type resultType,
                                      mlir::ConversionPatternRewriter &rewriter) const final {
    auto kind = kindName(op.getKind());
    if constexpr (std::is_same<Op, rite::ArithNoEscapeOp>::value) {
      kind += "_no_escape";
    }
    auto newOp = opaqueCallOp(rewriter, op->getLoc(), resultType, kind, operands);
    rewriter.replaceOp(op, newOp.getResult(0));
    return mlir::success();
  }
};

struct StackAllocationOpConversion : public LightstormConversionPattern<rite::StackAllocationOp> {
  explicit StackAllocationOpConversion(LightstormConversionContext &conversionContext)
      : LightstormConversionPattern(conversionContext) {}

  mlir::LogicalResult matchAndRewrite(rite::StackAllocationOp op, llvm::ArrayRef<mlir::Value> operands,
                                      llvm::ArrayRef<mlir::Type> operandTypes,
                                      mlir::Type resultType,
                                      mlir::ConversionPatternRewriter &rewriter) const final {
    assert(resultType.isa<mlir::emitc::PointerType>());
    auto stackValueType = resultType.cast<mlir::emitc::PointerType>().getPointee();
    auto stackValue = opaqueCallOp(rewriter, op->getLoc(), stackValueType, "LS_ALLOC_STACK_VALUE", operands);
    auto addrOf = rewriter.create<mlir::emitc::ApplyOp>(op->getLoc(), resultType, rewriter.getStringAttr("&"), stackValue.getResult(0));
    rewriter.replaceOp(op, addrOf->getResults());
    return mlir::success();
  }
};

} // namespace lightstorm_conversion

#define DirectOpConversion(Op, function)                                                           \
  patterns.add<lightstorm_conversion::DirectOpConversion<Op>>(loweringContext,                     \
                                                              std::string("" #function))

#define MethodRefOpConversion(Op, function)                                                        \
  patterns.add<lightstorm_conversion::MethodRefOpConversion<Op>>(loweringContext,                  \
                                                                 std::string("" #function))

namespace lightstorm_conversion_passes {
// Creates a getter function for each symbol mentioned in any function
// The getter is used later during the lowering/conversion phase
class ExtractSymIntern
    : public mlir::PassWrapper<ExtractSymIntern, mlir::OperationPass<mlir::ModuleOp>> {
public:
  explicit ExtractSymIntern(mlir::TypeConverter &typeConverter) : typeConverter(typeConverter) {}

private:
  void runOnOperation() override {
    auto module = getOperation();
    std::unordered_set<std::string> symbols;
    module->walk([&](rite::InternSymOp op) { symbols.insert(op.getMidAttr().getSymname()); });

    mlir::OpBuilder builder(module.getBodyRegion());
    for (auto &symbol : symbols) {
      auto symName = cCompatibleSymName(symbol);
      auto functionName = "_ls_sym_getter_" + symName;

      auto symType = typeConverter.convertType(rite::mrb_symType::get(&getContext()));
      auto mrbState = typeConverter.convertType(rite::mrb_stateType::get(&getContext()));
      auto functionType = builder.getFunctionType({ mrbState }, { symType });
      auto function =
          builder.create<mlir::emitc::FuncOp>(module->getLoc(), functionName, functionType);
      function.setSpecifiersAttr(builder.getStrArrayAttr({ "LIGHTSTORM_INLINE", "static" }));
      function.setVisibility(mlir::SymbolTable::Visibility::Private);
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(function.addEntryBlock());

      auto quotedSymName = '"' + symbol + '"';

      std::string s;
      std::stringstream macro(s);
      macro << "LS_INTERN_SYMBOL(" << quotedSymName << ", " << std::to_string(symbol.size())
            << ");";

      builder.create<mlir::emitc::VerbatimOp>(function->getLoc(), macro.str());

      auto fakeAttr = mlir::emitc::OpaqueAttr::get(&getContext(), "0");
      auto fakeSym = builder.create<mlir::emitc::ConstantOp>(function->getLoc(), symType, fakeAttr);
      builder.create<mlir::emitc::ReturnOp>(function->getLoc(), fakeSym);
    }
  }
  mlir::TypeConverter &typeConverter;
};
} // namespace lightstorm_conversion_passes

void lightstorm::convertRiteToEmitC(const LightstormConfig &config, mlir::MLIRContext &context,
                                    mlir::ModuleOp module) {
  mlir::TypeConverter typeConverter;

  mlir::ConversionTarget target(context);
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalOp<mlir::arith::ConstantOp>();
  target.addLegalDialect<mlir::func::FuncDialect>();
  target.addLegalDialect<mlir::BuiltinDialect>();
  target.addLegalDialect<mlir::emitc::EmitCDialect>();

  target.addDynamicallyLegalDialect<mlir::cf::ControlFlowDialect>(
      [&](mlir::Operation *op) { return typeConverter.isLegal(op->getOperandTypes()); });

  target.addDynamicallyLegalOp<mlir::func::FuncOp>(
      [&](mlir::func::FuncOp op) { return typeConverter.isLegal(op.getFunctionType()); });

  typeConverter.addConversion([&](mlir::Type type) -> std::optional<mlir::Type> {
    if (type.isa<rite::mrb_valueType>()) {
      return mlir::emitc::OpaqueType::get(&context, "mrb_value");
    }
    if (type.isa<rite::mrb_symType>()) {
      return mlir::emitc::OpaqueType::get(&context, "mrb_sym");
    }
    if (type.isa<rite::mrb_stateType>()) {
      auto opaque = mlir::emitc::OpaqueType::get(&context, "mrb_state");
      return mlir::emitc::PointerType::get(opaque);
    }
    if (type.isa<rite::stack_allocated_valueType>()) {
      return mlir::emitc::PointerType::get(mlir::emitc::OpaqueType::get(&context, "struct RFloat"));
    }
    if (type.isa<mlir::FunctionType>()) {
      auto functionType = type.cast<mlir::FunctionType>();
      llvm::SmallVector<mlir::Type> results;
      llvm::SmallVector<mlir::Type> inputs;
      if (mlir::failed(typeConverter.convertTypes(functionType.getResults(), results))) {
        llvm::errs() << "Failed to convert function result types\n";
        return std::nullopt;
      }
      if (mlir::failed(typeConverter.convertTypes(functionType.getInputs(), inputs))) {
        llvm::errs() << "Failed to convert function input types\n";
        return std::nullopt;
      }
      return mlir::FunctionType::get(&context, inputs, results);
    }
    return type;
  });

  mlir::PassManager pipeline(&context);
  pipeline.addPass(std::make_unique<lightstorm_conversion_passes::ExtractSymIntern>(typeConverter));
  if (mlir::failed(pipeline.run(module))) {
    module.print(llvm::errs());
    llvm::errs() << "Cannot extract mrb_syms\n";
    exit(1);
  }

  mlir::RewritePatternSet patterns(&context);

  LightstormConversionContext loweringContext{ context, typeConverter };

  mlir::populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, typeConverter);
  mlir::populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  mlir::populateFuncToEmitCPatterns(patterns);
  patterns.add<
      ///
      lightstorm_conversion::InternSymOpConversion,
      lightstorm_conversion::LoadStringOpConversion,
      lightstorm_conversion::ReturnOpConversion,
      lightstorm_conversion::LocationOpConversion,
      lightstorm_conversion::StackAllocationOpConversion,
      lightstorm_conversion::KindOpConversion<rite::BranchPredicateOp>,
      lightstorm_conversion::KindOpConversion<rite::ArithOp>,
      lightstorm_conversion::KindOpConversion<rite::ArithNoEscapeOp>,
      lightstorm_conversion::KindOpConversion<rite::CmpOp>,
      lightstorm_conversion::KindOpConversion<rite::LoadValueOp>

      ///
      >(loweringContext);

  MethodRefOpConversion(rite::ExecOp, ls_exec);
  MethodRefOpConversion(rite::MethodOp, ls_create_method);

  DirectOpConversion(rite::LoadIOp, ls_load_i);
  DirectOpConversion(rite::LoadFloatOp, ls_load_f);
  DirectOpConversion(rite::LoadSymOp, ls_load_sym);
  DirectOpConversion(rite::LoadLocalVariableOp, ls_load_local_variable);
  DirectOpConversion(rite::DefOp, ls_define_method);
  DirectOpConversion(rite::SClassOp, ls_load_singleton_class);
  DirectOpConversion(rite::ModuleOp, ls_define_module);
  DirectOpConversion(rite::ClassOp, ls_vm_define_class);
  DirectOpConversion(rite::EnterOp, ls_enter);
  DirectOpConversion(rite::StrCatOp, ls_strcat);
  DirectOpConversion(rite::InternOp, ls_intern_string);
  DirectOpConversion(rite::SendVOp, ls_sendv);

  DirectOpConversion(rite::GetConstOp, ls_get_const);
  DirectOpConversion(rite::SetConstOp, ls_set_const);
  DirectOpConversion(rite::GetGVOp, ls_get_global_variable);
  DirectOpConversion(rite::SetGVOp, ls_set_global_variable);
  DirectOpConversion(rite::GetIVOp, ls_get_instance_variable);
  DirectOpConversion(rite::SetIVOp, ls_set_instance_variable);
  DirectOpConversion(rite::GetCVOp, ls_get_class_variable);
  DirectOpConversion(rite::SetCVOp, ls_set_class_variable);
  DirectOpConversion(rite::GetMCNSTOp, ls_get_module_const);
  DirectOpConversion(rite::SetMCNSTOp, ls_set_module_const);
  DirectOpConversion(rite::ARefOp, ls_aref);
  DirectOpConversion(rite::APostOp, ls_apost);
  DirectOpConversion(rite::SendOp, ls_send);
  DirectOpConversion(rite::HashOp, ls_hash);
  DirectOpConversion(rite::HashCatOp, ls_hash_merge);
  DirectOpConversion(rite::HashAddOp, ls_hash_add);
  DirectOpConversion(rite::ArrayOp, ls_array);
  DirectOpConversion(rite::ArrayPushOp, ls_array_push);
  DirectOpConversion(rite::ArrayCatOp, ls_array_cat);
  DirectOpConversion(rite::RangeIncOp, ls_range_inc);
  DirectOpConversion(rite::RangeExcOp, ls_range_exc);
  DirectOpConversion(rite::AliasOp, ls_alias_method);
  DirectOpConversion(rite::UndefOp, ls_undef_method);

  mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  if (mlir::failed(mlir::applyFullConversion(module.getOperation(), target, frozenPatterns))) {
    module.getOperation()->print(llvm::errs(), mlir::OpPrintingFlags().enableDebugInfo(true, true));
    llvm::errs() << "Cannot apply C conversion\n";
    exit(1);
  }
  if (mlir::failed(mlir::verify(module))) {
    llvm::errs() << "Invalid module after C conversion\n";
    module.print(llvm::errs(), mlir::OpPrintingFlags().enableDebugInfo(true, true));
    exit(1);
  }
}

void lightstorm::convertMLIRToC(const LightstormConfig &config, mlir::MLIRContext &context,
                                mlir::ModuleOp module, llvm::raw_ostream &out) {
  mlir::OpBuilder b(module.getBodyRegion());
  b.create<mlir::emitc::IncludeOp>(module->getLoc(),
                                   config.runtime_header_location + "/lightstorm_runtime.h");

  if (mlir::failed(mlir::emitc::translateToCpp(module, out, true))) {
    llvm::errs() << "Cannot convert MLIR to C\n";
    exit(1);
  }
}
