include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"

def Rite_Dialect : Dialect {
  let name = "rite";
  let summary = "A one-to-one mapping from mruby RITE VM bytecode to MLIR";

  let cppNamespace = "rite";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

class RiteType<string name> : TypeDef<Rite_Dialect, name> {
  let summary = name;
  let mnemonic = name;
  let hasCustomAssemblyFormat = 0;
}

def StateType : RiteType<"mrb_state"> {}
def ValueType : RiteType<"mrb_value"> {}
def ValuePtrType : RiteType<"mrb_value_ptr"> {}

class Rite_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Rite_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def SymAttr : Rite_Attr<"mrb_sym", "Symbol"> {
  let summary = "An Attribute wrapping an mrb_sym and its name";
  let parameters = (ins "std::string":$symname);
  let assemblyFormat = [{
    `<`$symname`>`
  }];
}

class Rite_Op<string mnemonic, list<Trait> traits = []> :
    Op<Rite_Dialect, mnemonic, traits>;

//
// Loads
//

// OPCODE(LOADSELF,   B)        /* R(a) = self */
def LoadSelfOp : Rite_Op<"OP_LOADSELF"> {
  let summary = "OP_LOADSELF";
  let arguments = (ins IndexAttr:$address, StateType:$mrb);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` `{` `address` `=` $address `}` attr-dict `->` type(results)
  }];
}

// OPCODE(LOADI,      BB)       /* R(a) = mrb_int(b) */
def LoadIOp : Rite_Op<"OP_LOADI"> {
  let summary = "OP_LOADI";
  let arguments = (ins IndexAttr:$address, StateType:$mrb, I64:$value);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` `{` `address` `=` $address `}` attr-dict `->` type(results)
  }];
}

// OPCODE(LOADNIL,    B)        /* R(a) = nil */
def LoadNilOp : Rite_Op<"OP_LOADNIL"> {
  let summary = "OP_LOADNIL";
  let arguments = (ins IndexAttr:$address, StateType:$mrb);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` `{` `address` `=` $address `}` attr-dict `->` type(results)
  }];
}

//
// Comparisons
//

// OPCODE(EQ,         B)        /* R(a) = R(a)==R(a+1) */
def EqOp : Rite_Op<"OP_EQ"> {
  let summary = "OP_EQ";
  let arguments = (ins IndexAttr:$address, StateType:$mrb, ValueType:$lhs, ValueType:$rhs);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` `{` `address` `=` $address `}` attr-dict `->` type(results)
  }];
}

// OPCODE(GT,         B)        /* R(a) = R(a)>R(a+1) */
def GtOp : Rite_Op<"OP_GT"> {
  let summary = "OP_GT";
  let arguments = (ins IndexAttr:$address, StateType:$mrb, ValueType:$lhs, ValueType:$rhs);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` `{` `address` `=` $address `}` attr-dict `->` type(results)
  }];
}

// OPCODE(GE,         B)        /* R(a) = R(a)>=R(a+1) */
def GeOp : Rite_Op<"OP_GE"> {
  let summary = "OP_GE";
  let arguments = (ins IndexAttr:$address, StateType:$mrb, ValueType:$lhs, ValueType:$rhs);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` `{` `address` `=` $address `}` attr-dict `->` type(results)
  }];
}

// OPCODE(LT,         B)        /* R(a) = R(a)<R(a+1) */
def LtOp : Rite_Op<"OP_LT"> {
  let summary = "OP_LT";
  let arguments = (ins IndexAttr:$address, StateType:$mrb, ValueType:$lhs, ValueType:$rhs);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` `{` `address` `=` $address `}` attr-dict `->` type(results)
  }];
}

// OPCODE(LE,         B)        /* R(a) = R(a)<=R(a+1) */
def LeOp : Rite_Op<"OP_LE"> {
  let summary = "OP_LE";
  let arguments = (ins IndexAttr:$address, StateType:$mrb, ValueType:$lhs, ValueType:$rhs);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` `{` `address` `=` $address `}` attr-dict `->` type(results)
  }];
}

//
// Calls
//

// OPCODE(SEND,       BBB)      /* R(a) = call(R(a),Syms(b),R(a+1),...,R(a+c)) */
def SendOp : Rite_Op<"OP_SEND"> {
  let summary = "OP_SEND";
  let arguments = (ins IndexAttr:$address, StateType:$mrb, ValueType:$receiver, SymAttr:$symbol, I64Attr:$argc, Variadic<ValueType>:$argv);
  let results = (outs ValueType);
  let assemblyFormat = [{
    ``$symbol `(` $mrb `,` $receiver (`,` $argv^)? `)`
        `{` `address` `=` $address `,` `argc` `=` $argc `}`
        attr-dict `->` type(results)
  }];
}

// OPCODE(RETURN,     B)        /* return R(a) (normal) */
def ReturnOp : Rite_Op<"OP_RETURN", [Terminator]> {
  let summary = "OP_RETURN";
  let arguments = (ins IndexAttr:$address, StateType:$mrb, ValueType:$src);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` `{` `address` `=` $address `}` attr-dict `->` type(results)
  }];
}

//
// SSA/Load/Store
//

def VirtualRegisterOp : Rite_Op<"vreg", [DeclareOpInterfaceMethods<PromotableAllocationOpInterface>]> {
  let summary = "Mimics alloca behavior. Used for SSA/Mem2Reg.";
  let arguments = (ins IndexAttr:$regnum);
  let results = (outs ValuePtrType);
  let assemblyFormat = [{
    `(` $regnum `)` attr-dict `->` type(results)
  }];
}

def LoadOp : Rite_Op<"load", [DeclareOpInterfaceMethods<PromotableMemOpInterface>]> {
  let summary = "Load value from a virtual stack ref";
  let arguments = (ins ValuePtrType:$slot);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` $slot `)` attr-dict `->` type(results)
  }];
}

def StoreOp : Rite_Op<"store", [DeclareOpInterfaceMethods<PromotableMemOpInterface>]> {
  let summary = "Store value to a virtual stack ref";
  let arguments = (ins ValuePtrType:$slot, ValueType:$value);
  let assemblyFormat = [{
    `(` $slot `,` $value  `)` attr-dict
  }];
}

//
// Jumps and predicates
//

// OP_JMPIF
def BranchPredicateTrue : I32EnumAttrCase<"bp_true", 1>;
// OP_JMPNOT
def BranchPredicateFalse : I32EnumAttrCase<"bp_false", 2>;
// OP_JMPNIL
def BranchPredicateNil : I32EnumAttrCase<"bp_nil", 3>;

def BranchPredicate : I32EnumAttr<
    "BranchPredicate",
    "branch predicate",
    [BranchPredicateTrue, BranchPredicateFalse, BranchPredicateNil]> {
  let cppNamespace = "rite";
}

def BranchPredicateOp : Rite_Op<"branch_predicate"> {
  let summary = "branch_predicate";
  let arguments = (ins ValueType:$value, BranchPredicate:$predicate);
  let results = (outs I1);
}

///
/// Arithmetic
///

// OP_ADD
def ArithAdd : I32EnumAttrCase<"add", 1>;
// OP_SUB
def ArithSub : I32EnumAttrCase<"sub", 2>;
// OP_MUL
def ArithMul : I32EnumAttrCase<"mul", 3>;
// OP_DIV
def ArithDiv : I32EnumAttrCase<"div", 4>;

def Arith : I32EnumAttr<
    "Arith",
    "Arithmetic operation",
    [ArithAdd, ArithSub, ArithMul, ArithDiv]> {
  let cppNamespace = "rite";
}

def ArithOp : Rite_Op<"arith"> {
  let summary = "arith";
  let arguments = (ins StateType:$mrb, ValueType:$lhs, ValueType:$rhs, Arith:$kind);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}
