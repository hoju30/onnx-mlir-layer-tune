#ifndef POSIT_OPS
#define POSIT_OPS

include "mlir/IR/OpBase.td"
// include "posit/PositDialect.td"
include "PositTypes.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"  // 在.h檔中也要include


// 所有 posit 方言 op 的共同基底
class Posit_Op<string mnemonic, list<Trait> traits = []>
  : Op<Posit_Dialect, mnemonic, traits>;

// 12/3 改成能讀懂DenseElementsAttr tensor
def Posit_ConstantOp : Posit_Op<"constant", [Pure]> {
  let summary = "create a posit constant";
  let description = [{
    Create a scalar `!posit.type<nbits,es>` constant, where the `value` holds
    the raw bit-pattern as a signed 64-bit integer.
  }];

  let results   = (outs AnyType:$result);
  let arguments = (ins AnyAttr:$value);

  let assemblyFormat = "attr-dict `:` type($result)";
}


def Posit_AddOp : Posit_Op<"add", [Pure, SameOperandsAndResultType]> {
  let summary = "add two posit scalars";
  let description = [{
    Add two `!posit.type<nbits,es>` values element.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results   = (outs AnyType:$result);

  // 自訂語法： 'posit.add %a, %b : !posit.type<8,0>'
  let assemblyFormat = "$lhs  `,` $rhs  attr-dict `:` type($result)";
  let hasFolder = 0; 
}

// sub mul div 新增
def Posit_SubOp : Posit_Op<"sub", [Pure, SameOperandsAndResultType]> {
  let summary = "subtract two posit scalars";
  let description = [{
    Subtract two `!posit.type<nbits,es>` values element.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results   = (outs AnyType:$result);

  // 自訂語法： 'posit.sub %a, %b : !posit.type<8,0>'
  let assemblyFormat = "$lhs  `,` $rhs  attr-dict `:` type($result)";
  let hasFolder = 0;
}

def Posit_MulOp : Posit_Op<"mul", [Pure, SameOperandsAndResultType]> {
  let summary = "multiply two posit scalars";
  let description = [{
    Multiply two `!posit.type<nbits,es>` values element.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results   = (outs AnyType:$result);

  // 自訂語法： 'posit.mul %a, %b : !posit.type<8,0>'
  let assemblyFormat = "$lhs  `,` $rhs  attr-dict `:` type($result)";
  let hasFolder = 0;
}

def Posit_DivOp : Posit_Op<"div", [Pure, SameOperandsAndResultType]> {
  let summary = "divide two posit scalars";
  let description = [{
    Divide two `!posit.type<nbits,es>` values element.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results   = (outs AnyType:$result);

  // 自訂語法： 'posit.div %a, %b : !posit.type<8,0>'
  let assemblyFormat = "$lhs  `,` $rhs  attr-dict `:` type($result)";
  let hasFolder = 0;
}

#endif // POSIT_OPS