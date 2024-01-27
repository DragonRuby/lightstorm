#include "lightstorm/conversion/conversion.h"
#include "lightstorm/dialect/rite.h"
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
    auto charType = mlir::emitc::OpaqueType::get(getContext(), "const char");
    auto charPtrType = mlir::emitc::PointerType::get(charType);
    auto quotedSymName = '"' + op.getMidAttr().getSymname() + '"';
    auto symAttr = mlir::emitc::OpaqueAttr::get(getContext(), quotedSymName);
    auto symName = rewriter.create<mlir::emitc::ConstantOp>(op->getLoc(), charPtrType, symAttr);
    auto sizeAttr = rewriter.getI64IntegerAttr(op.getMidAttr().getSymname().size());
    auto size =
        rewriter.create<mlir::emitc::ConstantOp>(op->getLoc(), sizeAttr.getType(), sizeAttr);
    mlir::ValueRange mrbInternOperands{ operands[0], symName, size };
    auto mrbSym = opaqueCallOp(rewriter, op->getLoc(), resultType, "mrb_intern", mrbInternOperands);
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

struct ExecOpConversion : public LightstormConversionPattern<rite::ExecOp> {
  explicit ExecOpConversion(LightstormConversionContext &conversionContext)
      : LightstormConversionPattern(conversionContext) {}

  mlir::LogicalResult matchAndRewrite(rite::ExecOp op, llvm::ArrayRef<mlir::Value> operands,
                                      llvm::ArrayRef<mlir::Type> operandTypes,
                                      mlir::Type resultType,
                                      mlir::ConversionPatternRewriter &rewriter) const final {
    auto mrb_func_t = mlir::emitc::OpaqueType::get(getContext(), "mrb_func_t");
    auto ref = rewriter.create<mlir::emitc::ConstantOp>(
        op->getLoc(),
        mrb_func_t,
        mlir::emitc::OpaqueAttr::get(getContext(), op.getFuncAttr().getAttr()));
    auto newOp = opaqueCallOp(rewriter,
                              op->getLoc(),
                              resultType,
                              "ls_exec",
                              mlir::ValueRange{ operands.front(), operands.back(), ref });
    rewriter.replaceOp(op, newOp.getResult(0));
    return mlir::success();
  }
};

struct MethodOpConversion : public LightstormConversionPattern<rite::MethodOp> {
  explicit MethodOpConversion(LightstormConversionContext &conversionContext)
      : LightstormConversionPattern(conversionContext) {}

  mlir::LogicalResult matchAndRewrite(rite::MethodOp op, llvm::ArrayRef<mlir::Value> operands,
                                      llvm::ArrayRef<mlir::Type> operandTypes,
                                      mlir::Type resultType,
                                      mlir::ConversionPatternRewriter &rewriter) const final {
    auto mrb_func_t = mlir::emitc::OpaqueType::get(getContext(), "mrb_func_t");
    auto ref = rewriter.create<mlir::emitc::ConstantOp>(
        op->getLoc(),
        mrb_func_t,
        mlir::emitc::OpaqueAttr::get(getContext(), op.getMethodAttr().getAttr()));
    auto newOp = opaqueCallOp(rewriter,
                              op->getLoc(),
                              resultType,
                              "ls_create_method",
                              mlir::ValueRange{ operands.front(), ref });
    rewriter.replaceOp(op, newOp.getResult(0));
    return mlir::success();
  }
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
    auto newOp = opaqueCallOp(rewriter, op->getLoc(), resultType, kindName(op.getKind()), operands);
    rewriter.replaceOp(op, newOp.getResult(0));
    return mlir::success();
  }
};

} // namespace lightstorm_conversion

#define DirectOpConversion(Op, function)                                                           \
  patterns.add<lightstorm_conversion::DirectOpConversion<Op>>(loweringContext,                     \
                                                              std::string("" #function))

void lightstorm::convertRiteToEmitC(mlir::MLIRContext &context, mlir::ModuleOp module) {
  mlir::ConversionTarget target(context);
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalOp<mlir::arith::ConstantOp>();
  target.addLegalDialect<mlir::func::FuncDialect>();
  target.addLegalDialect<mlir::BuiltinDialect>();
  target.addLegalDialect<mlir::emitc::EmitCDialect>();

  mlir::TypeConverter typeConverter;

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

  mlir::RewritePatternSet patterns(&context);

  LightstormConversionContext loweringContext{ context, typeConverter };

  mlir::populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, typeConverter);
  mlir::populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  patterns.add<
      ///
      lightstorm_conversion::InternSymOpConversion,
      lightstorm_conversion::LoadStringOpConversion,
      lightstorm_conversion::ExecOpConversion,
      lightstorm_conversion::MethodOpConversion,
      lightstorm_conversion::ReturnOpConversion,
      lightstorm_conversion::KindOpConversion<rite::BranchPredicateOp>,
      lightstorm_conversion::KindOpConversion<rite::ArithOp>,
      lightstorm_conversion::KindOpConversion<rite::CmpOp>,
      lightstorm_conversion::KindOpConversion<rite::LoadValueOp>

      ///
      >(loweringContext);

  DirectOpConversion(rite::LoadIOp, ls_load_i);
  DirectOpConversion(rite::LoadFloatOp, ls_load_f);
  DirectOpConversion(rite::LoadSymOp, ls_load_sym);
  DirectOpConversion(rite::LoadLocalVariableOp, ls_load_local_variable);
  DirectOpConversion(rite::DefOp, ls_define_method);
  DirectOpConversion(rite::SClassOp, ls_load_singleton_class);
  DirectOpConversion(rite::ModuleOp, ls_define_module);
  DirectOpConversion(rite::ClassOp, ls_vm_define_class);
  DirectOpConversion(rite::StrCatOp, ls_strcat);
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
  DirectOpConversion(rite::HashCatOp, ls_hash_merge);
  DirectOpConversion(rite::SendOp, ls_send);
  DirectOpConversion(rite::HashOp, ls_hash);
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

void lightstorm::convertMLIRToC(mlir::MLIRContext &context, mlir::ModuleOp module,
                                llvm::raw_ostream &out) {
  mlir::OpBuilder b(module.getBodyRegion());
  b.create<mlir::emitc::IncludeOp>(module->getLoc(), "lightstorm/runtime/runtime.h");

  if (mlir::failed(mlir::emitc::translateToCpp(module, out, true))) {
    llvm::errs() << "Cannot convert MLIR to C\n";
    exit(1);
  }
}
