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
def SymType : RiteType<"mrb_sym"> {}

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

def LoadLocalVariableOp : Rite_Op<"local_variable"> {
  let summary = "Load local variable from the virtual stack";
  let arguments = (ins StateType:$mrb, Index:$regnum);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

// OPCODE(LOADI,      BB)       /* R(a) = mrb_int(b) */
def LoadIOp : Rite_Op<"OP_LOADI"> {
  let summary = "OP_LOADI";
  let arguments = (ins StateType:$mrb, I64:$value);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

// OPCODE(LOADL,      BB)       /* R(a) = Pool(b) */
def LoadFloatOp : Rite_Op<"OP_LOADFLOAT"> {
  let summary = "OP_LOADFLOAT";
  let arguments = (ins StateType:$mrb, F64:$value);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

// OPCODE(LOADSYM,    BB)       /* R(a) = Syms(b) */
def LoadSymOp : Rite_Op<"OP_LOADSYM"> {
  let summary = "OP_LOADSYM";
  let arguments = (ins StateType:$mrb, SymType:$mid);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

// OPCODE(STRING,     BB)       /* R(a) = str_dup(Lit(b)) */
def LoadStringOp : Rite_Op<"OP_STRING"> {
  let summary = "OP_STRING";
  let arguments = (ins StateType:$mrb, StrAttr:$str, I64:$len);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` $mrb `,` $str `,` $len `)` attr-dict `->` type(results)
  }];
}

// OPCODE(STRCAT,     B)        /* str_cat(R(a),R(a+1)) */
def StrCatOp : Rite_Op<"OP_STRCAT"> {
  let summary = "OP_STRCAT";
  let arguments = (ins StateType:$mrb, ValueType:$str, ValueType:$cat);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

// OPCODE(INTERN,     B)        /* R(a) = intern(R(a)) */
def InternOp : Rite_Op<"OP_INTERN"> {
  let summary = "OP_INTERN";
  let arguments = (ins StateType:$mrb, ValueType:$str);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

// OPCODE(SCLASS,     B)        /* R(a) = R(a).singleton_class */
def SClassOp : Rite_Op<"OP_SCLASS"> {
  let summary = "OP_SCLASS";
  let arguments = (ins StateType:$mrb, ValueType:$target);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

// OP_LOADNIL
def LoadValueNil : I32EnumAttrCase<"nil_value", 1>;
// OP_LOADSELF
def LoadValueSelf : I32EnumAttrCase<"self_value", 2>;
// OP_LOADT
def LoadValueTrue : I32EnumAttrCase<"true_value", 3>;
// OP_LOADF
def LoadValueFalse : I32EnumAttrCase<"false_value", 4>;
// OP_TCLASS
def LoadValueTargetClass : I32EnumAttrCase<"target_class_value", 5>;
// OP_OCLASS
def LoadValueObjectClass : I32EnumAttrCase<"object_class_value", 6>;

def LoadValueKind : I32EnumAttr<
    "LoadValueKind",
    "Value kind",
    [LoadValueNil,
     LoadValueSelf,
     LoadValueTrue,
     LoadValueFalse,
     LoadValueTargetClass,
     LoadValueObjectClass
    ]> {
  let cppNamespace = "rite";
}

def LoadValueOp : Rite_Op<"load_value"> {
  let summary = "Load value";
  let arguments = (ins StateType:$mrb, LoadValueKind:$kind);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `[` $kind `]` `(` operands `)` attr-dict `->` type(results)
  }];
}

//
// Comparisons
//

// OP_EQ
def CmpEq : I32EnumAttrCase<"eq", 1>;
// OP_LT
def CmpLt : I32EnumAttrCase<"lt", 2>;
// OP_LE
def CmpLe : I32EnumAttrCase<"le", 3>;
// OP_GT
def CmpGt : I32EnumAttrCase<"gt", 4>;
// OP_GE
def CmpGe : I32EnumAttrCase<"ge", 5>;

def CmpKind : I32EnumAttr<
    "CmpKind",
    "Comparison operation kind",
    [CmpEq, CmpLt, CmpLe, CmpGt, CmpGe]> {
  let cppNamespace = "rite";
}

def CmpOp : Rite_Op<"cmp"> {
  let summary = "cmp";
  let arguments = (ins StateType:$mrb, ValueType:$lhs, ValueType:$rhs, CmpKind:$kind);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

//
// Calls
//

def InternSymOp : Rite_Op<"intern_symbol"> {
  let summary = "Converts string symbol into a numerical value";
  let arguments = (ins StateType:$mrb, SymAttr:$mid);
  let results = (outs SymType);
  let assemblyFormat = [{
    `(` $mrb `,` $mid `)` attr-dict `->` type(results)
  }];
}

// OPCODE(ENTER,      W)        /* arg setup according to flags (23=m5:o5:r1:m5:k5:d1:b1) */
def EnterOp : Rite_Op<"OP_ENTER"> {
  let summary = "OP_ENTER";
  let arguments = (ins StateType:$mrb, I64:$requiredArgs);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

// OPCODE(MODULE,     BB)       /* R(a) = newmodule(R(a),Syms(b)) */
def ModuleOp : Rite_Op<"OP_MODULE"> {
  let summary = "OP_MODULE";
  let arguments = (ins StateType:$mrb, ValueType:$target, SymType:$symbol);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

// OPCODE(METHOD,     BB)       /* R(a) = lambda(SEQ[b],L_METHOD) */
def MethodOp : Rite_Op<"OP_METHOD"> {
  let summary = "OP_METHOD";
  let arguments = (ins StateType:$mrb, FlatSymbolRefAttr:$method);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` $mrb `,` $method `)` attr-dict
  }];
}

// OPCODE(DEF,        BB)       /* R(a).newmethod(Syms(b),R(a+1)) */
def DefOp : Rite_Op<"OP_DEF"> {
  let summary = "OP_DEF";
  let arguments = (ins StateType:$mrb, ValueType:$target, ValueType:$method, SymType:$mid);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict
  }];
}

// OPCODE(SEND,       BBB)      /* R(a) = call(R(a),Syms(b),R(a+1),...,R(a+c)) */
def SendOp : Rite_Op<"OP_SEND"> {
  let summary = "OP_SEND";
  let arguments = (ins StateType:$mrb, ValueType:$receiver, SymType:$symbol, I64:$argc, Variadic<ValueType>:$argv);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `,` type($argv) `)` attr-dict `->` type(results)
  }];
}

// OPCODE(SENDV,      BB)       /* R(a) = call(R(a),Syms(b),*R(a+1)) */
def SendVOp : Rite_Op<"OP_SENDV"> {
  let summary = "OP_SENDV";
  let arguments = (ins StateType:$mrb, ValueType:$receiver, SymType:$symbol, ValueType:$argv);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

// OPCODE(EXEC,       BB)       /* R(a) = blockexec(R(a),SEQ[b]) */
def ExecOp : Rite_Op<"OP_EXEC"> {
  let summary = "OP_EXEC";
  let arguments = (ins StateType:$mrb, ValueType:$receiver, FlatSymbolRefAttr:$method);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `,`  $method `)` attr-dict `->` type(results)
  }];
}

// OPCODE(RETURN,     B)        /* return R(a) (normal) */
def ReturnOp : Rite_Op<"OP_RETURN", [Terminator]> {
  let summary = "OP_RETURN";
  let arguments = (ins StateType:$mrb, ValueType:$src);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
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
def BranchPredicateTrue : I32EnumAttrCase<"is_true", 1>;
// OP_JMPNOT
def BranchPredicateFalse : I32EnumAttrCase<"is_false", 2>;
// OP_JMPNIL
def BranchPredicateNil : I32EnumAttrCase<"is_nil", 3>;

def BranchPredicateKind : I32EnumAttr<
    "BranchPredicateKind",
    "branch predicate",
    [BranchPredicateTrue, BranchPredicateFalse, BranchPredicateNil]> {
  let cppNamespace = "rite";
}

def BranchPredicateOp : Rite_Op<"branch_predicate"> {
  let summary = "branch_predicate";
  let arguments = (ins ValueType:$value, BranchPredicateKind:$kind);
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

def ArithKind : I32EnumAttr<
    "ArithKind",
    "Arithmetic operation",
    [ArithAdd, ArithSub, ArithMul, ArithDiv]> {
  let cppNamespace = "rite";
}

def ArithOp : Rite_Op<"arith"> {
  let summary = "arith";
  let arguments = (ins StateType:$mrb, ValueType:$lhs, ValueType:$rhs, ArithKind:$kind);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

///
/// Array Ops
///

// OPCODE(ARRAY,      BB)       /* R(a) = ary_new(R(a),R(a+1)..R(a+b)) */
def ArrayOp : Rite_Op<"OP_ARRAY"> {
  let summary = "OP_ARRAY";
  let arguments = (ins StateType:$mrb, I64:$argc, Variadic<ValueType>:$argv);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `,` type($argv) `)` attr-dict `->` type(results)
  }];
}

// OPCODE(AREF,       BBB)      /* R(a) = R(b)[c] */
def ARefOp : Rite_Op<"OP_AREF"> {
  let summary = "OP_AREF";
  let arguments = (ins StateType:$mrb, ValueType:$array, Index:$index);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

// OPCODE(APOST,      BBB)      /* *R(a),R(a+1)..R(a+c) = R(a)[b..] */
def APostOp : Rite_Op<"OP_APOST"> {
  let summary = "OP_APOST";
  let arguments = (ins StateType:$mrb, ValueType:$array, Index:$pre, Index:$post);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

// OPCODE(ARYPUSH,    B)        /* ary_push(R(a),R(a+1)) */
def ArrayPushOp : Rite_Op<"OP_ARYPUSH"> {
  let summary = "OP_ARYPUSH";
  let arguments = (ins StateType:$mrb, ValueType:$array, ValueType:$value);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

// OPCODE(ARYCAT,     B)        /* ary_cat(R(a),R(a+1)) */
def ArrayCatOp : Rite_Op<"OP_ARYCAT"> {
  let summary = "OP_ARYCAT";
  let arguments = (ins StateType:$mrb, ValueType:$lhs, ValueType:$rhs);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

///
/// Hash Ops
///

// OPCODE(HASH,       BB)       /* R(a) = hash_new(R(a),R(a+1)..R(a+b*2-1)) */
def HashOp : Rite_Op<"OP_HASH"> {
  let summary = "OP_HASH";
  let arguments = (ins StateType:$mrb, I64:$argc, Variadic<ValueType>:$argv);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `,` type($argv) `)` attr-dict `->` type(results)
  }];
}

// OPCODE(HASHCAT,    B)        /* R(a) = hash_cat(R(a),R(a+1)) */
def HashCatOp : Rite_Op<"OP_HASHCAT"> {
  let summary = "OP_HASHCAT";
  let arguments = (ins StateType:$mrb, ValueType:$h1, ValueType:$h2);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

// OPCODE(HASHADD,    BB)       /* R(a) = hash_push(R(a),R(a+1)..R(a+b*2)) */
def HashAddOp : Rite_Op<"OP_HASHADD"> {
  let summary = "OP_HASHADD";
  let arguments = (ins StateType:$mrb, ValueType:$hash, I64:$argc, Variadic<ValueType>:$argv);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `,` type($argv) `)` attr-dict `->` type(results)
  }];
}

///
/// Const Ops
///

// OPCODE(GETCONST,   BB)       /* R(a) = constget(Syms(b)) */
def GetConstOp : Rite_Op<"OP_GETCONST"> {
  let summary = "OP_GETCONST";
  let arguments = (ins StateType:$mrb, SymType:$symbol);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

// OPCODE(SETCONST,   BB)       /* constset(Syms(b),R(a)) */
def SetConstOp : Rite_Op<"OP_SETCONST"> {
  let summary = "OP_SETCONST";
  let arguments = (ins StateType:$mrb, SymType:$symbol, ValueType:$value);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

// OPCODE(GETGV,      BB)       /* R(a) = getglobal(Syms(b)) */
def GetGVOp : Rite_Op<"OP_GETGV"> {
  let summary = "OP_GETGV";
  let arguments = (ins StateType:$mrb, SymType:$symbol);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

// OPCODE(SETGV,      BB)       /* setglobal(Syms(b), R(a)) */
def SetGVOp : Rite_Op<"OP_SETGV"> {
  let summary = "OP_SETGV";
  let arguments = (ins StateType:$mrb, SymType:$symbol, ValueType:$value);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

// OPCODE(GETIV,      BB)       /* R(a) = ivget(Syms(b)) */
def GetIVOp : Rite_Op<"OP_GETIV"> {
  let summary = "OP_GETIV";
  let arguments = (ins StateType:$mrb, ValueType:$self, SymType:$symbol);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

// OPCODE(SETIV,      BB)       /* ivset(Syms(b),R(a)) */
def SetIVOp : Rite_Op<"OP_SETIV"> {
  let summary = "OP_SETIV";
  let arguments = (ins StateType:$mrb, ValueType:$self, SymType:$symbol, ValueType:$value);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

// OPCODE(GETCV,      BB)       /* R(a) = cvget(Syms(b)) */
def GetCVOp : Rite_Op<"OP_GETCV"> {
  let summary = "OP_GETCV";
  let arguments = (ins StateType:$mrb, SymType:$symbol);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

// OPCODE(SETCV,      BB)       /* cvset(Syms(b),R(a)) */
def SetCVOp : Rite_Op<"OP_SETCV"> {
  let summary = "OP_SETCV";
  let arguments = (ins StateType:$mrb, SymType:$symbol, ValueType:$value);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

// OPCODE(GETMCNST,   BB)       /* R(a) = R(a)::Syms(b) */
def GetMCNSTOp : Rite_Op<"OP_GETMCNST"> {
  let summary = "OP_GETMCNST";
  let arguments = (ins StateType:$mrb, ValueType:$receiver, SymType:$symbol);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

// OPCODE(SETMCNST,   BB)       /* R(a+1)::Syms(b) = R(a) */
def SetMCNSTOp : Rite_Op<"OP_SETMCNST"> {
  let summary = "OP_SETMCNST";
  let arguments = (ins StateType:$mrb, ValueType:$receiver, SymType:$symbol, ValueType:$value);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

///
/// Classes/Modules
///

def ClassOp : Rite_Op<"OP_CLASS"> {
  let summary = "OP_CLASS";
  let arguments = (ins StateType:$mrb, ValueType:$base_class, ValueType:$super_class, SymType:$symbol);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

// OPCODE(ALIAS,      BB)       /* alias_method(target_class,Syms(a),Syms(b)) */
def AliasOp : Rite_Op<"OP_ALIAS"> {
  let summary = "OP_ALIAS";
  let arguments = (ins StateType:$mrb, SymType:$sym1, SymType:$sym2);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

// OPCODE(UNDEF,      B)        /* undef_method(target_class,Syms(a)) */
def UndefOp : Rite_Op<"OP_UNDEF"> {
  let summary = "OP_UNDEF";
  let arguments = (ins StateType:$mrb, SymType:$sym);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

///
/// Ranges
///

// OPCODE(RANGE_INC,  B)        /* R(a) = range_new(R(a),R(a+1),FALSE) */
def RangeIncOp : Rite_Op<"OP_RANGE_INC"> {
  let summary = "OP_RANGE_INC";
  let arguments = (ins StateType:$mrb, ValueType:$rbegin, ValueType:$rend);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}

// OPCODE(RANGE_EXC,  B)        /* R(a) = range_new(R(a),R(a+1),TRUE) */
def RangeExcOp : Rite_Op<"OP_RANGE_EXC"> {
  let summary = "OP_RANGE_EXC";
  let arguments = (ins StateType:$mrb, ValueType:$rbegin, ValueType:$rend);
  let results = (outs ValueType);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `->` type(results)
  }];
}
