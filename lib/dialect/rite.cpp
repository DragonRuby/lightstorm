#include "lightstorm/dialect/rite.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

using namespace mlir;
using namespace rite;

#include "RiteDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "RiteAttrDefs.cpp.inc"

void RiteDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "RiteAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "RiteOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "RiteTypeDefs.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "RiteOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "RiteTypeDefs.cpp.inc"

//
// Mem2Reg support
//

Value rite::VirtualRegisterOp::getDefaultValue(const MemorySlot &slot, RewriterBase &rewriter) {
  return {};
}

llvm::SmallVector<MemorySlot> rite::VirtualRegisterOp::getPromotableSlots() {
  return { MemorySlot{ getResult(), rite::mrb_valueType::get(getContext()) } };
}

void rite::VirtualRegisterOp::handleBlockArgument(const MemorySlot &slot, BlockArgument argument,
                                                  RewriterBase &rewriter) {}

void rite::VirtualRegisterOp::handlePromotionComplete(const MemorySlot &slot, Value defaultValue,
                                                      RewriterBase &rewriter) {
  rewriter.eraseOp(*this);
}

bool rite::LoadOp::canUsesBeRemoved(const MemorySlot &slot,
                                    const SmallPtrSetImpl<OpOperand *> &blockingUses,
                                    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  return blockingUse == slot.ptr && getSlot() == slot.ptr && getResult().getType() == slot.elemType;
}

DeletionKind rite::LoadOp::removeBlockingUses(const MemorySlot &slot,
                                              const SmallPtrSetImpl<OpOperand *> &blockingUses,
                                              RewriterBase &rewriter, Value reachingDefinition) {
  rewriter.replaceAllUsesWith(getResult(), reachingDefinition);
  return DeletionKind::Delete;
}

bool rite::LoadOp::loadsFrom(const MemorySlot &slot) {
  return getSlot() == slot.ptr;
}

bool rite::LoadOp::storesTo(const MemorySlot &slot) {
  return false;
}

Value rite::LoadOp::getStored(const MemorySlot &slot, RewriterBase &rewriter) {
  llvm_unreachable("getStored should not be called on LoadOp");
}

bool rite::StoreOp::canUsesBeRemoved(const MemorySlot &slot,
                                     const SmallPtrSetImpl<OpOperand *> &blockingUses,
                                     SmallVectorImpl<OpOperand *> &newBlockingUses) {
  if (blockingUses.size() != 1)
    return false;

  Value blockingUse = (*blockingUses.begin())->get();
  return blockingUse == slot.ptr && getSlot() == slot.ptr && getValue() != slot.ptr &&
         getValue().getType() == slot.elemType;
}

DeletionKind rite::StoreOp::removeBlockingUses(const MemorySlot &slot,
                                               const SmallPtrSetImpl<OpOperand *> &blockingUses,
                                               RewriterBase &rewriter, Value reachingDefinition) {
  return DeletionKind::Delete;
}

bool rite::StoreOp::loadsFrom(const MemorySlot &slot) {
  return false;
}

bool rite::StoreOp::storesTo(const MemorySlot &slot) {
  return getSlot() == slot.ptr;
}

Value rite::StoreOp::getStored(const MemorySlot &slot, RewriterBase &rewriter) {
  return getValue();
}
