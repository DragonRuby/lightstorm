#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/MemorySlotInterfaces.h>

#include "RiteDialect.h.inc"
#include "RiteEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "RiteTypeDefs.h.inc"

#define GET_ATTRDEF_CLASSES
#include "RiteAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "RiteOps.h.inc"
