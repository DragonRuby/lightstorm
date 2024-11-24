#include "lightstorm/optimizations/optimizations.h"

#include "lightstorm/dialect/rite.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace lightstorm;

// - Can we have escape analysis?
// - We have escape analysis at home
class EscapeAnalysisAtHome
    : public mlir::PassWrapper<EscapeAnalysisAtHome, mlir::OperationPass<mlir::func::FuncOp>> {
public:
  /*
    In the cases like the following
      def dot
        return @x * b.x + @y * b.y + @z * b.z
      end

    Bytecode has the following form (schematically):
      R1 = OP_MUL(@x, b.x)
      R2 = OP_MUL(@y, b.y)
      R3 = OP_MUL(@z, b.z)
      R4 = OP_ADD(R1, R2)
      R5 = OP_ADD(R3, R4)
      OP_RETURN(R5)

    By default, all the values (R1-R5) will be allocated on the heap as they may escape from the
    local scope in case when a custom class implements their own version of arith op
    (e.g. `def +(x) $SOME GLOBAL << x end`).
    In the case of `RFloat`s, the implementation is hardcoded in the VM, so we can provide an
    opportunity for the runtime to use a stack-allocated slot instead of using GC/heap for
    potentially short-lived objects. In the example above R1-R4 will be allocated on the stack and
    deallocated right after execution leaves the scope without adding useless GC/heap overhead.
    The decision whether to use heap or stack depends on the actual type at runtime, so the decision
    should be made at runtime, but we can pre-allocate some space on the stack and let runtime
    decide which one to use.
   */
  void runOnOperation() override {
    auto function = getOperation();
    auto &entry = function.getBlocks().front();

    std::vector<rite::ArithOp> worklist;
    for (auto arith : function.getOps<rite::ArithOp>()) {
      // If all the uses of the ArithOp are another ArithOp, then this op can have an opportunity to
      // use stack-allocated slot instead of heap allocation
      auto is_arith = [](mlir::Operation *user) { return llvm::isa<rite::ArithOp>(user); };
      if (std::all_of(arith->user_begin(), arith->user_end(), is_arith)) {
        worklist.push_back(arith);
      }
    }

    auto mrb = function.getArgument(0);
    auto valueType = rite::stack_allocated_valueType::get(&getContext());
    mlir::OpBuilder builder(&getContext());
    for (auto arith : worklist) {
      builder.setInsertionPointToStart(&entry);
      auto slot = builder.create<rite::StackAllocationOp>(arith.getLoc(), valueType, mrb);
      builder.setInsertionPointAfter(arith);
      auto no_escape = builder.create<rite::ArithNoEscapeOp>(arith->getLoc(),
                                                             arith.getType(),
                                                             arith.getMrb(),
                                                             arith.getLhs(),
                                                             arith.getRhs(),
                                                             slot,
                                                             arith.getKindAttr());
      arith->replaceAllUsesWith(no_escape);
      arith->erase();
    }
  }
};

void lightstorm::applyOptimizations(const LightstormConfig &config, mlir::MLIRContext &context,
                                    mlir::ModuleOp module) {
  mlir::PassManager pipeline(&context);
  pipeline.addNestedPass<mlir::func::FuncOp>(std::make_unique<EscapeAnalysisAtHome>());
  pipeline.addPass(mlir::createCSEPass());
  if (pipeline.run(module).failed()) {
    module.print(llvm::errs());
    llvm::errs() << "\nFailed to run passes\n";
    exit(1);
  }
}
