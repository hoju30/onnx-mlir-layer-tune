#ifndef POSIT_OPS
#define POSIT_OPS

include "mlir/IR/OpBase.td"
include "PositTypes.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"  // 在.h檔中也要include

// 所有 posit 方言 op 的共同基底
class Posit_Op<string mnemonic, list<Trait> traits = []>
  : Op<Posit_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

// 12/3 改成能讀懂DenseElementsAttr tensor
def Posit_ConstantOp : Posit_Op<"constant", [Pure]> {
  let summary = "create a posit constant";
  let description = [{
    Create a `!posit.type<nbits,es>` constant tensor, where the `value` holds
    the raw bit-pattern as a signed integer tensor (e.g., i8 for posit<8,0>).
  }];

  let results   = (outs AnyType:$result);
  let arguments = (ins AnyAttr:$value);

  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Elementwise math
//===----------------------------------------------------------------------===//

// Add/Sub/Mul/Div allow NumPy-style broadcasting between operands (needed for
// dynamic-shape models, e.g. GPT-2 attention mask: scalar/lower-rank vs tensor).
// The trait was previously SameOperandsAndResultType, which forced onnx->posit to
// insert rank-changing casts that broke posit->krnl. Operands keep their natural
// ranks; the result carries the broadcast shape. The assembly format therefore
// prints all operand types via functional-type. The runtime elementwise kernels
// handle the broadcast (offset_with_broadcast, incl. rank-0 scalars).
def Posit_AddOp : Posit_Op<"add", [Pure]> {
  let summary = "add two posit values (broadcasting)";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results   = (outs AnyType:$result);
  let assemblyFormat = "$lhs  `,` $rhs  attr-dict `:` functional-type(operands, results)";
  let hasFolder = 0;
}

def Posit_SubOp : Posit_Op<"sub", [Pure]> {
  let summary = "subtract two posit values (broadcasting)";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results   = (outs AnyType:$result);
  let assemblyFormat = "$lhs  `,` $rhs  attr-dict `:` functional-type(operands, results)";
  let hasFolder = 0;
}

def Posit_MulOp : Posit_Op<"mul", [Pure]> {
  let summary = "multiply two posit values (broadcasting)";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results   = (outs AnyType:$result);
  let assemblyFormat = "$lhs  `,` $rhs  attr-dict `:` functional-type(operands, results)";
  let hasFolder = 0;
}

def Posit_DivOp : Posit_Op<"div", [Pure]> {
  let summary = "divide two posit values (broadcasting)";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results   = (outs AnyType:$result);
  let assemblyFormat = "$lhs  `,` $rhs  attr-dict `:` functional-type(operands, results)";
  let hasFolder = 0;
}

def Posit_ClipOp : Posit_Op<"clip", [Pure, SameOperandsAndResultType]> {
  let summary = "clip posit values by optional min/max";
  let arguments = (ins AnyType:$input);
  let results   = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
  let hasFolder = 0;
}

// 01/14 新增：f32 <-> posit conversion，讓 function signature 保持 f32， 但計算可在 posit 內進行。

def Posit_FromF32Op : Posit_Op<"from_f32", [Pure]> {
  let summary = "convert f32 tensor/scalar to posit";
  let arguments = (ins AnyTensor:$input);
  let results   = (outs AnyTensor:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasFolder = 0;
}

def Posit_ToF32Op : Posit_Op<"to_f32", [Pure]> {
  let summary = "convert posit tensor/scalar to f32";
  let arguments = (ins AnyTensor:$input);
  let results   = (outs AnyTensor:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasFolder = 0;
}

def Posit_DequantizeLinearOp : Posit_Op<"dequantize_linear", [Pure]> {
  let summary = "dequantize integer tensor to posit tensor, with optional f32 reference";
  let description = [{
    Dequantize an integer QDQ tensor into posit storage or f32 output.
    The optional `orig_ref` operand is the pre-QuantizeLinear f32 tensor used
    only for qalign/full-reference collection and probing. It lets runtime collect
    paired samples under the same qalign key:
      orig_samples = orig_ref
      dq_samples   = (q - zero_point) * scale
  }];
  let arguments = (ins AnyTensor:$input, Variadic<AnyTensor>:$orig_ref);
  let results   = (outs AnyTensor:$result);
  // Use generic assembly so older one-operand IR and newer two-operand IR both
  // parse without a fragile optional custom format.
  let hasFolder = 0;
}

def Posit_ReluOp : Posit_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "ReLU for posit";
  let arguments = (ins AnyType:$input);
  let results   = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
  let hasFolder = 0;
}

// NN ops for MNIST (NCHW)

// 01/14 新增：Conv/MaxPool/Reshape/Gemm 先以 MNIST 為主。

def Posit_Conv2DOp : Posit_Op<"conv2d", [Pure]> {
  let summary = "Conv2D NCHW (posit)";
  let arguments = (ins AnyTensor:$X, AnyTensor:$W, AnyTensor:$B);
  let results   = (outs AnyTensor:$result);

  let assemblyFormat = "$X `,` $W `,` $B attr-dict `:`"
                       " type($X) `,` type($W) `,` type($B) `->` type($result)";
  let hasFolder = 0;
}

def Posit_MaxPool2DOp : Posit_Op<"maxpool2d", [Pure]> {
  let summary = "MaxPool2D NCHW (posit)";
  let arguments = (ins AnyTensor:$X);
  let results   = (outs AnyTensor:$result);

  let assemblyFormat = "$X attr-dict `:` type($X) `->` type($result)";
  let hasFolder = 0;
}

def Posit_ReshapeOp : Posit_Op<"reshape", [Pure]> {
  let summary = "Reshape (view) for posit";
  let arguments = (ins AnyTensor:$data, AnyTensor:$shape);
  let results   = (outs AnyTensor:$result);

  let assemblyFormat = "$data `,` $shape attr-dict `:`"
                       " type($data) `,` type($shape) `->` type($result)";
  let hasFolder = 0;
}

def Posit_FlattenOp : Posit_Op<"flatten", [Pure]> {
  let summary = "Flatten tensor for posit";
  let arguments = (ins AnyTensor:$data);
  let results   = (outs AnyTensor:$result);
  let assemblyFormat = "$data attr-dict `:` type($data) `->` type($result)";
  let hasFolder = 0;
}

def Posit_GemmOp : Posit_Op<"gemm", [Pure]> {
  let summary = "Gemm (MatMul + bias) for posit";
  let arguments = (ins AnyTensor:$A, AnyTensor:$B, AnyTensor:$C);
  let results   = (outs AnyTensor:$result);

  let assemblyFormat = "$A `,` $B `,` $C attr-dict `:`"
                       " type($A) `,` type($B) `,` type($C) `->` type($result)";
  let hasFolder = 0;
}

def Posit_ReduceMeanOp : Posit_Op<"reduce_mean", [Pure]> {
  let summary = "ReduceMean for posit";
  let arguments = (ins AnyTensor:$input);
  let results   = (outs AnyTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasFolder = 0;
}

#endif // POSIT_OPS
