#include "lightstorm/conversion/conversion.h"
#include "lightstorm/dialect/rite.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/EmitC/IR/EmitC.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Target/Cpp/CppEmitter.h>
#include <mlir/Transforms/DialectConversion.h>

static mlir::func::FuncOp lookupOrCreateFn(mlir::ModuleOp moduleOp, llvm::StringRef name,
                                           mlir::TypeRange paramTypes = {},
                                           mlir::Type resultType = {}) {
  auto func = moduleOp.lookupSymbol<mlir::func::FuncOp>(name);
  if (func) {
    return func;
  }
  mlir::OpBuilder b(moduleOp.getBodyRegion());
  auto type = mlir::FunctionType::get(moduleOp->getContext(), paramTypes, resultType);
  auto f = b.create<mlir::func::FuncOp>(moduleOp->getLoc(), name, type);
  f.setVisibility(mlir::SymbolTable::Visibility::Private);
  return f;
}

static mlir::func::FuncOp lookupOrCreateFn(mlir::Operation *op, llvm::StringRef name,
                                           mlir::TypeRange paramTypes = {},
                                           mlir::Type resultType = {}) {
  auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
  return lookupOrCreateFn(moduleOp, name, paramTypes, resultType);
}

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

struct BlockArgumentTypeConversion : public LightstormConversionPattern<mlir::func::FuncOp> {
  explicit BlockArgumentTypeConversion(LightstormConversionContext &loweringContext)
      : LightstormConversionPattern(loweringContext) {}

  mlir::LogicalResult matchAndRewrite(mlir::func::FuncOp op, llvm::ArrayRef<mlir::Value> operands,
                                      llvm::ArrayRef<mlir::Type> operandTypes,
                                      mlir::Type resultType,
                                      mlir::ConversionPatternRewriter &rewriter) const final {
    auto newType =
        conversionContext.converter.convertType(op.getFunctionType()).cast<mlir::FunctionType>();
    auto newFunc = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getSymName(), newType);
    rewriter.inlineRegionBefore(op.getBody(), newFunc.addEntryBlock());
    rewriter.eraseBlock(&newFunc.getBlocks().back());
    if (mlir::failed(
            rewriter.convertRegionTypes(&newFunc.getBody(), conversionContext.converter))) {
      llvm::errs() << "failed to convert region types\n";
      return mlir::failure();
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

template <typename Op> struct DirectOpConversion : public LightstormConversionPattern<Op> {
  explicit DirectOpConversion(LightstormConversionContext &conversionContext, std::string name)
      : LightstormConversionPattern<Op>(conversionContext), name(std::move(name)) {}

  mlir::LogicalResult matchAndRewrite(Op op, llvm::ArrayRef<mlir::Value> operands,
                                      llvm::ArrayRef<mlir::Type> operandTypes,
                                      mlir::Type resultType,
                                      mlir::ConversionPatternRewriter &rewriter) const final {
    auto func = lookupOrCreateFn(op, name, operandTypes, resultType);
    auto newOp = rewriter.create<mlir::func::CallOp>(op->getLoc(), func, operands);
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

struct LoadIOpConversion : public LightstormConversionPattern<rite::LoadIOp> {
  explicit LoadIOpConversion(LightstormConversionContext &conversionContext)
      : LightstormConversionPattern(conversionContext) {}

  mlir::LogicalResult matchAndRewrite(rite::LoadIOp op, llvm::ArrayRef<mlir::Value> operands,
                                      llvm::ArrayRef<mlir::Type> operandTypes,
                                      mlir::Type resultType,
                                      mlir::ConversionPatternRewriter &rewriter) const final {
    auto func = lookupOrCreateFn(op, "ls_load_i", operandTypes, resultType);
    auto newOp = rewriter.create<mlir::func::CallOp>(op->getLoc(), func, operands);
    rewriter.replaceOp(op, newOp.getResult(0));
    return mlir::success();
  }
};

struct SendOpConversion : public LightstormConversionPattern<rite::SendOp> {
  explicit SendOpConversion(LightstormConversionContext &conversionContext)
      : LightstormConversionPattern(conversionContext) {}

  mlir::LogicalResult matchAndRewrite(rite::SendOp op, llvm::ArrayRef<mlir::Value> operands,
                                      llvm::ArrayRef<mlir::Type> operandTypes,
                                      mlir::Type resultType,
                                      mlir::ConversionPatternRewriter &rewriter) const final {
    auto mrbValueType = operandTypes[1]; // receiver
    auto charType = mlir::emitc::OpaqueType::get(getContext(), "const char");
    auto symType = mlir::emitc::OpaqueType::get(getContext(), "mrb_sym");
    auto charPtrType = mlir::emitc::PointerType::get(charType);
    mlir::TypeRange mrbInternTypes{
      // MRB_API mrb_sym mrb_intern(mrb_state* mrb, const char* s, size_t size);
      operandTypes[0],      // mrb
      charPtrType,          // s
      rewriter.getI64Type() // size
    };
    auto quotedSymName = '"' + op.getSymbolAttr().getSymname() + '"';
    auto symAttr = mlir::emitc::OpaqueAttr::get(getContext(), quotedSymName);
    auto symName = rewriter.create<mlir::emitc::ConstantOp>(op->getLoc(), charPtrType, symAttr);
    auto sizeAttr = rewriter.getI64IntegerAttr(op.getSymbolAttr().getSymname().size());
    auto size =
        rewriter.create<mlir::emitc::ConstantOp>(op->getLoc(), sizeAttr.getType(), sizeAttr);
    mlir::ValueRange mrbInternOperands{ operands[0], symName, size };
    auto mrbIntern = lookupOrCreateFn(op, "mrb_intern", mrbInternTypes, symType);
    auto mrbSym = rewriter.create<mlir::func::CallOp>(op->getLoc(), mrbIntern, mrbInternOperands);
    std::vector<mlir::Type> newOperandTypes{
      // mimics vararg function
      // ls_funcall_X(mrb_state *mrb, mrb_value recv, mrb_sym name, mrb_int argc, mrb_value i..X);
      operandTypes[0],       // mrb
      operandTypes[1],       // receiver
      symType,               // name
      rewriter.getI64Type(), // argc
    };
    for (int i = 0; i < op.getArgv().size(); i++) {
      newOperandTypes.push_back(mrbValueType);
    }
    auto func = lookupOrCreateFn(
        op, "ls_funcall_" + std::to_string(op.getArgv().size()), newOperandTypes, resultType);
    auto argc = rewriter.create<mlir::emitc::ConstantOp>(
        op->getLoc(), op.getArgcAttr().getType(), op.getArgcAttr());
    std::vector<mlir::Value> newOperands{ operands[0], operands[1], mrbSym.getResult(0), argc };
    for (int i = 0; i < op.getArgv().size(); i++) {
      newOperands.push_back(operands[i + 2]);
    }
    auto newOp = rewriter.create<mlir::func::CallOp>(op->getLoc(), func, newOperands);
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

  target.addDynamicallyLegalOp<mlir::func::FuncOp>(
      [&](mlir::func::FuncOp op) { return typeConverter.isLegal(op.getFunctionType()); });

  typeConverter.addConversion([&](mlir::Type type) -> std::optional<mlir::Type> {
    if (type.isa<rite::mrb_valueType>()) {
      return mlir::emitc::OpaqueType::get(&context, "mrb_value");
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

  patterns.add<
      ///
      lightstorm_conversion::BlockArgumentTypeConversion,
      lightstorm_conversion::SendOpConversion,
      lightstorm_conversion::ReturnOpConversion

      ///
      >(loweringContext);

  DirectOpConversion(rite::LoadSelfOp, ls_load_self);
  DirectOpConversion(rite::LoadIOp, ls_load_i);
  DirectOpConversion(rite::LoadNilOp, ls_load_nil);
  DirectOpConversion(rite::GtOp, ls_compare_gt);
  DirectOpConversion(rite::GeOp, ls_compare_ge);
  DirectOpConversion(rite::LtOp, ls_compare_lt);
  DirectOpConversion(rite::LeOp, ls_compare_le);
  DirectOpConversion(rite::EqOp, ls_compare_eq);

  mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  if (mlir::failed(mlir::applyFullConversion(module.getOperation(), target, frozenPatterns))) {
    module.getOperation()->print(llvm::errs(), mlir::OpPrintingFlags().enableDebugInfo(true, true));
    llvm::errs() << "Cannot apply C conversion\n";
    return;
  }
  if (mlir::failed(mlir::verify(module))) {
    llvm::errs() << "Invalid module after C conversion\n";
    return;
  }
}

void lightstorm::convertMLIRToC(mlir::MLIRContext &context, mlir::ModuleOp module,
                                llvm::raw_ostream &out) {
  // EmitC emits empty function bodies even for private declarations, remove those right before
  // conversion
  std::vector<mlir::func::FuncOp> toRemove;
  auto ops = module.getOps<mlir::func::FuncOp>();
  std::copy_if(ops.begin(), ops.end(), std::back_inserter(toRemove), [&](auto op) {
    return op.isDeclaration();
  });
  std::for_each(toRemove.begin(), toRemove.end(), [&](auto &op) { op->remove(); });

  mlir::OpBuilder b(module.getBodyRegion());
  b.create<mlir::emitc::IncludeOp>(module->getLoc(), "lightstorm/runtime/runtime.h");

  if (mlir::failed(mlir::emitc::translateToCpp(module, out))) {
    llvm::errs() << "Cannot convert MLIR to C\n";
  }
}
