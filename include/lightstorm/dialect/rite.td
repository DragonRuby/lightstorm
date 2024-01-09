include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
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

// OPCODE(LOADSELF,   B)        /* R(a) = self */
def LoadSelfOp : Rite_Op<"OP_LOADSELF"> {
  let summary = "OP_LOADSELF";
  let arguments = (ins IndexAttr:$address, StateType:$mrb);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` $mrb `)` `{` `address` `=` $address `}` attr-dict `->` type(results)
  }];
}

// OPCODE(LOADI,      BB)       /* R(a) = mrb_int(b) */
def LoadIOp : Rite_Op<"OP_LOADI"> {
  let summary = "OP_LOADI";
  let arguments = (ins IndexAttr:$address, StateType:$mrb, I64:$value);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` $mrb `,` $value `)` `{` `address` `=` $address `}` attr-dict `->` type(results)
  }];
}

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
    `(` $mrb `,` $src `)` `{` `address` `=` $address `}` attr-dict `->` type(results)
  }];
}

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
