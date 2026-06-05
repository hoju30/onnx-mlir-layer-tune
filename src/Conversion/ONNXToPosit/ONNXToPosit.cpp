#include "onnx-mlir/Conversion/ONNXToPosit/ONNXToPosit.hpp"
#include "TypeConverters.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

// 11/26 pipeline
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

// 改成TypeConverters.hpp
#include "src/Conversion/ONNXToPosit/TypeConverters.hpp"

// 讓mlir::posit可以被讀懂
#include "src/Dialect/Posit/PositDialect.h"
#include "src/Dialect/Posit/PositPasses.h"
#include "src/Dialect/Posit/PositOps.h"

// ONNX
#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "mlir/IR/BuiltinAttributes.h"

#include "src/Conversion/ONNXToPosit/Pattern/Math.cpp"
#include "src/Pass/Passes.hpp"

using namespace onnx_mlir;
using namespace mlir;
using namespace posit;

namespace onnx_mlir {
namespace {

static bool parseONNXToPositEnvBool(const char *name, bool fallback = false) {
  const char *e = std::getenv(name);
  if (!e || !*e)
    return fallback;
  std::string v(e);
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return v == "1" || v == "true" || v == "on" || v == "yes" || v == "y";
}

static Type toF32TensorLike(Type t, Builder &b) {
  if (auto rtt = llvm::dyn_cast<RankedTensorType>(t))
    return RankedTensorType::get(rtt.getShape(), b.getF32Type());
  if (llvm::isa<UnrankedTensorType>(t))
    return UnrankedTensorType::get(b.getF32Type());
  return Type();
}

static Operation *createGenericOp(IRRewriter &rewriter, Location loc,
    StringRef name, TypeRange resultTypes, ValueRange operands,
    ArrayRef<NamedAttribute> attrs = {}) {
  OperationState st(loc, name);
  st.addOperands(operands);
  st.addTypes(resultTypes);
  st.addAttributes(attrs);
  return rewriter.create(st);
}

static bool decomposeQLinearLikeOps(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  Builder builder(ctx);
  IRRewriter rewriter(ctx);
  bool changed = false;
  auto si64Attr = [&](int64_t v) -> IntegerAttr {
    Type si64Ty = builder.getIntegerType(64, true);
    return builder.getIntegerAttr(si64Ty, v);
  };

  SmallVector<Operation *, 16> qlinearConvOps;
  SmallVector<Operation *, 16> qlinearAddCustomOps;
  module.walk([&](Operation *op) {
    // Keep QLinearConv as a quantized operator. It is lowered by the
    // ONNXDequantizeLinear pattern when used as DQ(QLinearConv(...)), so the
    // compute path can stay in the posit domain instead of being pre-expanded
    // to DQ -> f32 Conv -> Q. This preserves QOperator semantics for target-B.
    if (op->getName().getStringRef() == "onnx.QLinearConv")
      return;
    if (op->getName().getStringRef() != "onnx.Custom")
      return;
    auto fnAttr = op->getAttrOfType<StringAttr>("function_name");
    if (fnAttr && fnAttr.getValue() == "QLinearAdd")
      qlinearAddCustomOps.push_back(op);
  });

  auto copyAttrIfPresent = [&](Operation *src, StringRef attrName,
                               SmallVectorImpl<NamedAttribute> &outAttrs) {
    if (Attribute a = src->getAttr(attrName))
      outAttrs.emplace_back(builder.getStringAttr(attrName), a);
  };

  for (Operation *op : qlinearConvOps) {
    if (!op || op->getBlock() == nullptr)
      continue;
    if (op->getNumOperands() < 8 || op->getNumResults() != 1)
      continue;

    Location loc = op->getLoc();
    rewriter.setInsertionPoint(op);

    Value x = op->getOperand(0);
    Value xScale = op->getOperand(1);
    Value xZero = op->getOperand(2);
    Value w = op->getOperand(3);
    Value wScale = op->getOperand(4);
    Value wZero = op->getOperand(5);
    Value yScale = op->getOperand(6);
    Value yZero = op->getOperand(7);
    Value bias = op->getNumOperands() >= 9 ? op->getOperand(8) : Value();

    Type xF32Ty = toF32TensorLike(x.getType(), builder);
    Type wF32Ty = toF32TensorLike(w.getType(), builder);
    Type yQTy = op->getResult(0).getType();
    Type yF32Ty = toF32TensorLike(yQTy, builder);
    if (!xF32Ty || !wF32Ty || !yF32Ty)
      continue;

    SmallVector<NamedAttribute, 4> dqXAttrs;
    dqXAttrs.emplace_back(builder.getStringAttr("axis"),
        si64Attr(1));
    Operation *dqXOp = createGenericOp(rewriter, loc, "onnx.DequantizeLinear",
        TypeRange{xF32Ty}, ValueRange{x, xScale, xZero}, dqXAttrs);

    SmallVector<NamedAttribute, 4> dqWAttrs;
    dqWAttrs.emplace_back(builder.getStringAttr("axis"),
        si64Attr(0));
    Operation *dqWOp = createGenericOp(rewriter, loc, "onnx.DequantizeLinear",
        TypeRange{wF32Ty}, ValueRange{w, wScale, wZero}, dqWAttrs);

    Value convBias;
    if (bias && !llvm::isa<NoneType>(bias.getType())) {
      Type bF32Ty = toF32TensorLike(bias.getType(), builder);
      if (bF32Ty) {
        bool foldedBias = false;
        RankedTensorType bRtt;
        SmallVector<int64_t, 64> bVals;
        RankedTensorType xScaleRtt;
        SmallVector<double, 8> xScaleVals;
        RankedTensorType wScaleRtt;
        SmallVector<double, 64> wScaleVals;
        if (getDenseIntegerTensorFromValue(bias, bRtt, bVals) &&
            getDenseFloatTensorFromValue(xScale, xScaleRtt, xScaleVals) &&
            getDenseFloatTensorFromValue(wScale, wScaleRtt, wScaleVals) &&
            !xScaleVals.empty() && !wScaleVals.empty()) {
          int64_t biasElems = static_cast<int64_t>(bVals.size());
          int64_t wScaleElems = static_cast<int64_t>(wScaleVals.size());
          if (biasElems > 0 && (wScaleElems == 1 || wScaleElems == biasElems)) {
            auto f32Ty = builder.getF32Type();
            auto outTy = RankedTensorType::get(bRtt.getShape(), f32Ty);
            SmallVector<Attribute, 64> outAttrs;
            outAttrs.reserve(static_cast<size_t>(biasElems));
            double xS = xScaleVals.front();
            for (int64_t i = 0; i < biasElems; ++i) {
              double wS = (wScaleElems == 1) ? wScaleVals.front() : wScaleVals[i];
              double v = static_cast<double>(bVals[i]) * xS * wS;
              outAttrs.push_back(FloatAttr::get(f32Ty, static_cast<float>(v)));
            }
            DenseElementsAttr foldedAttr = DenseElementsAttr::get(outTy, outAttrs);
            OperationState cstSt(loc, "onnx.Constant");
            cstSt.addTypes({outTy});
            cstSt.addAttribute("value", foldedAttr);
            convBias = rewriter.create(cstSt)->getResult(0);
            foldedBias = true;
          }
        }

        if (!foldedBias) {
          SmallVector<NamedAttribute, 2> castAttrs;
          castAttrs.emplace_back(
              builder.getStringAttr("to"), TypeAttr::get(builder.getF32Type()));
          Operation *castBias = createGenericOp(rewriter, loc, "onnx.Cast",
              TypeRange{bF32Ty}, ValueRange{bias}, castAttrs);

          Operation *scaleMul = createGenericOp(rewriter, loc, "onnx.Mul",
              TypeRange{bF32Ty}, ValueRange{xScale, wScale});
          Operation *biasMul = createGenericOp(rewriter, loc, "onnx.Mul",
              TypeRange{bF32Ty},
              ValueRange{castBias->getResult(0), scaleMul->getResult(0)});
          convBias = biasMul->getResult(0);
        }
      }
    }

    SmallVector<Value, 3> convOperands{dqXOp->getResult(0), dqWOp->getResult(0)};
    if (convBias)
      convOperands.push_back(convBias);

    SmallVector<NamedAttribute, 8> convAttrs;
    copyAttrIfPresent(op, "auto_pad", convAttrs);
    copyAttrIfPresent(op, "dilations", convAttrs);
    copyAttrIfPresent(op, "group", convAttrs);
    copyAttrIfPresent(op, "kernel_shape", convAttrs);
    copyAttrIfPresent(op, "pads", convAttrs);
    copyAttrIfPresent(op, "strides", convAttrs);
    copyAttrIfPresent(op, "onnx_node_name", convAttrs);
    Operation *convOp = createGenericOp(rewriter, loc, "onnx.Conv",
        TypeRange{yF32Ty}, convOperands, convAttrs);

    SmallVector<NamedAttribute, 4> qAttrs;
    qAttrs.emplace_back(builder.getStringAttr("axis"),
        si64Attr(1));
    qAttrs.emplace_back(builder.getStringAttr("saturate"),
        si64Attr(1));
    if (Attribute node = op->getAttr("onnx_node_name"))
      qAttrs.emplace_back(builder.getStringAttr("onnx_node_name"), node);
    Operation *qOp = createGenericOp(rewriter, loc, "onnx.QuantizeLinear",
        TypeRange{yQTy}, ValueRange{convOp->getResult(0), yScale, yZero}, qAttrs);

    rewriter.replaceOp(op, qOp->getResults());
    changed = true;
  }

  for (Operation *op : qlinearAddCustomOps) {
    if (!op || op->getBlock() == nullptr)
      continue;
    if (op->getNumOperands() < 8 || op->getNumResults() != 1)
      continue;

    Location loc = op->getLoc();
    rewriter.setInsertionPoint(op);

    Value a = op->getOperand(0);
    Value aScale = op->getOperand(1);
    Value aZero = op->getOperand(2);
    Value b = op->getOperand(3);
    Value bScale = op->getOperand(4);
    Value bZero = op->getOperand(5);
    Value yScale = op->getOperand(6);
    Value yZero = op->getOperand(7);

    Type yQTy = op->getResult(0).getType();
    Type yF32Ty = toF32TensorLike(yQTy, builder);
    Type aF32Ty = toF32TensorLike(a.getType(), builder);
    Type bF32Ty = toF32TensorLike(b.getType(), builder);
    if (!yF32Ty || !aF32Ty || !bF32Ty)
      continue;

    SmallVector<NamedAttribute, 4> dqAAttrs;
    dqAAttrs.emplace_back(builder.getStringAttr("axis"),
        si64Attr(1));
    Operation *dqA = createGenericOp(rewriter, loc, "onnx.DequantizeLinear",
        TypeRange{aF32Ty}, ValueRange{a, aScale, aZero}, dqAAttrs);

    SmallVector<NamedAttribute, 4> dqBAttrs;
    dqBAttrs.emplace_back(builder.getStringAttr("axis"),
        si64Attr(1));
    Operation *dqB = createGenericOp(rewriter, loc, "onnx.DequantizeLinear",
        TypeRange{bF32Ty}, ValueRange{b, bScale, bZero}, dqBAttrs);

    Operation *addOp = createGenericOp(rewriter, loc, "onnx.Add",
        TypeRange{yF32Ty}, ValueRange{dqA->getResult(0), dqB->getResult(0)});

    SmallVector<NamedAttribute, 4> qAttrs;
    qAttrs.emplace_back(builder.getStringAttr("axis"),
        si64Attr(1));
    qAttrs.emplace_back(builder.getStringAttr("saturate"),
        si64Attr(1));
    if (Attribute node = op->getAttr("onnx_node_name"))
      qAttrs.emplace_back(builder.getStringAttr("onnx_node_name"), node);
    Operation *qOp = createGenericOp(rewriter, loc, "onnx.QuantizeLinear",
        TypeRange{yQTy}, ValueRange{addOp->getResult(0), yScale, yZero}, qAttrs);

    rewriter.replaceOp(op, qOp->getResults());
    changed = true;
  }

  return changed;
}

struct ConvertONNXToPositPass
    : public PassWrapper<ConvertONNXToPositPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertONNXToPositPass)

  ConvertONNXToPositPass() = default;
  ConvertONNXToPositPass(
      unsigned nbits, unsigned es, bool alignToInt8QDomain,
      bool preferDirectF32FromQDQ)
      : nbits(nbits), es(es), alignToInt8QDomain(alignToInt8QDomain),
        preferDirectF32FromQDQ(preferDirectF32FromQDQ) {}

  // When true: rewriteDirectFromQSource() is tried even for f32-boundary DQ
  // nodes (boundaryF32Flow=true). This produces f32→posit round-trip from
  // the original pre-quantization f32 value, bypassing the INT8 DQ step.
  // Set at build time via env var POSIT_PREFER_DIRECT_FROM_QDQ=1.
  // This is Variant-A in the qdq benchmark: direct posit storage, no INT8.
  bool preferDirectPositFromQDQ = false;

  StringRef getArgument() const final { return "convert-onnx-to-posit"; }
  StringRef getDescription() const final {
    return "Lower ONNX ops and types to the Posit dialect.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<posit::PositDialect, arith::ArithDialect,
                    tensor::TensorDialect, func::FuncDialect>();
  }

  unsigned nbits = 8;
  unsigned es = 0;
  bool alignToInt8QDomain = false;
  bool preferDirectF32FromQDQ = true;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext &ctx = getContext();

    // Pre-decompose selected QLinear/custom patterns into standard ONNX ops.
    // This keeps QDQ models usable even when specific QLinear ops lack direct
    // lowering support in the current pipeline.
    bool localChanged = true;
    while (localChanged)
      localChanged = decomposeQLinearLikeOps(module);

    const unsigned localNbits = nbits;
    const unsigned localEs = es;
    PositTypeConverter typeConverter(localNbits, localEs, &ctx);

    // For the nqdq/f32 -> posit path, force selected numerical ONNX ops
    // (Conv/Gemm/MatMul/Add/etc.) to lower into the Posit dialect even though
    // their original operands/results are f32. The QDQ path leaves this unset
    // so only QDQ/posit islands are converted and ordinary f32 compute remains
    // legal. build_model11_sos.sh sets this for --posit-source nqdq.
    const bool forceAllNumericOpsToPosit =
        parseONNXToPositEnvBool("ONNX_MLIR_POSIT_FORCE_NQDQ", false) ||
        parseONNXToPositEnvBool("ONNX_MLIR_POSIT_FORCE_ALL_OPS", false) ||
        parseONNXToPositEnvBool("POSIT_FORCE_NQDQ_POSIT", false);

    const bool localPreferDirectPositFromQDQ =
        preferDirectPositFromQDQ ||
        parseONNXToPositEnvBool("POSIT_PREFER_DIRECT_FROM_QDQ", false) ||
        parseONNXToPositEnvBool("ONNX_MLIR_POSIT_PREFER_DIRECT_FROM_QDQ", false);

    // POSIT_STORE_DQ_AS_POSIT: emit posit bits from DQ (not f32), insert
    // posit.to_f32 before downstream onnx.Conv. Posit bits are persistently
    // stored at layer boundaries (Variant B persistent in qdq benchmark).
    const bool localPreferStoreAsPosit =
        parseONNXToPositEnvBool("POSIT_STORE_DQ_AS_POSIT", false) ||
        parseONNXToPositEnvBool("ONNX_MLIR_POSIT_STORE_DQ_AS_POSIT", false);

    RewritePatternSet patterns(&ctx);

    // [MOD][2026-01-14] 不做 func signature type-conversion
    // 讓 entry function 仍然是 f32 I/O，靠 posit.from_f32 / posit.to_f32 bridging
    // [EXTEND] Pass (nbits, es) through so pattern code no longer hardcodes p8e0.
    populateONNXToPositConversionPattern(
        typeConverter, patterns, &ctx, localNbits, localEs,
        alignToInt8QDomain, preferDirectF32FromQDQ,
        localPreferDirectPositFromQDQ, localPreferStoreAsPosit);

    ConversionTarget target(ctx);

    target.addLegalDialect<posit::PositDialect, arith::ArithDialect,
                           tensor::TensorDialect, func::FuncDialect>();

    target.addLegalOp<UnrealizedConversionCastOp>();

    // Keep ONNX graph terminators/metadata ops legal.
    target.addLegalOp<ONNXReturnOp, ONNXEntryPointOp, ONNXNoneOp,
                      ONNXQuantizeLinearOp>();
    target.addDynamicallyLegalOp<ONNXConstantOp>([&](ONNXConstantOp op) {
      if (!forceAllNumericOpsToPosit)
        return true;
      // Helper constants materialized during ONNX->Posit rewrites must stay
      // legal even when large float ONNX constants are otherwise forced
      // through posit lowering.
      if (op->hasAttr("onnx-mlir.posit-helper-const"))
        return true;
      auto valueAttr = op->getAttr("value");
      auto elements = llvm::dyn_cast_or_null<ElementsAttr>(valueAttr);
      if (!elements)
        return true;
      auto elemTy = elements.getElementType();
      if (!llvm::isa<FloatType>(elemTy))
        return true;
      // Keep scalar floating constants in ONNX; they are typically attributes/
      // small scalar helpers. Larger tensor constants (weights/bias) are
      // lowered to posit constants or from_f32 boundaries.
      return elements.getNumElements() <= 1;
    });
    target.addIllegalOp<ONNXQLinearConvOp>();

    // Force all DequantizeLinear ops through ONNX->Posit rewrite so we can
    // avoid the ONNX QDQ int8 lowering path entirely.
    target.addIllegalOp<ONNXDequantizeLinearOp>();
    // Convert only subgraphs that actually carry posit tensors.
    // This keeps DQ->f32 compute chains (Conv/Gemm in f32 domain) legal while
    // still lowering posit islands through posit->krnl.
    auto hasPositElemType = [](Type ty) -> bool {
      if (auto shapedTy = llvm::dyn_cast<ShapedType>(ty))
        return llvm::isa<posit::PositType>(shapedTy.getElementType());
      return llvm::isa<posit::PositType>(ty);
    };
    auto opNeedsPositConversion = [&](Operation *op) -> bool {
      for (Type t : op->getResultTypes())
        if (hasPositElemType(t))
          return true;
      for (Value v : op->getOperands())
        if (hasPositElemType(v.getType()))
          return true;
      return false;
    };

    target.addDynamicallyLegalOp<ONNXAddOp>(
        [&](ONNXAddOp op) { return !forceAllNumericOpsToPosit && !opNeedsPositConversion(op.getOperation()); });
    target.addDynamicallyLegalOp<ONNXSubOp>(
        [&](ONNXSubOp op) { return !forceAllNumericOpsToPosit && !opNeedsPositConversion(op.getOperation()); });
    target.addDynamicallyLegalOp<ONNXMulOp>(
        [&](ONNXMulOp op) { return !forceAllNumericOpsToPosit && !opNeedsPositConversion(op.getOperation()); });
    target.addDynamicallyLegalOp<ONNXDivOp>(
        [&](ONNXDivOp op) { return !forceAllNumericOpsToPosit && !opNeedsPositConversion(op.getOperation()); });
    target.addDynamicallyLegalOp<ONNXReshapeOp>([&](ONNXReshapeOp op) {
      return !forceAllNumericOpsToPosit && !opNeedsPositConversion(op.getOperation());
    });
    target.addDynamicallyLegalOp<ONNXUnsqueezeOp>([&](ONNXUnsqueezeOp op) {
      return !forceAllNumericOpsToPosit && !opNeedsPositConversion(op.getOperation());
    });
    target.addDynamicallyLegalOp<ONNXReluOp>(
        [&](ONNXReluOp op) { return !forceAllNumericOpsToPosit && !opNeedsPositConversion(op.getOperation()); });
    target.addDynamicallyLegalOp<ONNXClipOp>(
        [&](ONNXClipOp op) { return !forceAllNumericOpsToPosit && !opNeedsPositConversion(op.getOperation()); });
    target.addDynamicallyLegalOp<ONNXMaxPoolSingleOutOp>(
        [&](ONNXMaxPoolSingleOutOp op) {
          return !forceAllNumericOpsToPosit && !opNeedsPositConversion(op.getOperation());
        });
    target.addDynamicallyLegalOp<ONNXFlattenOp>([&](ONNXFlattenOp op) {
      return !forceAllNumericOpsToPosit && !opNeedsPositConversion(op.getOperation());
    });
    target.addDynamicallyLegalOp<ONNXConvOp>(
        [&](ONNXConvOp op) { return !forceAllNumericOpsToPosit && !opNeedsPositConversion(op.getOperation()); });
    target.addDynamicallyLegalOp<ONNXMatMulOp>([&](ONNXMatMulOp op) {
      return !forceAllNumericOpsToPosit && !opNeedsPositConversion(op.getOperation());
    });
    target.addDynamicallyLegalOp<ONNXGemmOp>(
        [&](ONNXGemmOp op) { return !forceAllNumericOpsToPosit && !opNeedsPositConversion(op.getOperation()); });
    target.addDynamicallyLegalOp<ONNXReduceMeanV13Op>(
        [&](ONNXReduceMeanV13Op op) {
          return !forceAllNumericOpsToPosit && !opNeedsPositConversion(op.getOperation());
        });

    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return true; });

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();

    // QDQ rewrite removes DequantizeLinear; QuantizeLinear becomes dead in the
    // common Q->DQ pattern. Erase dead Quantize ops so they do not leak to
    // later lowering stages.
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<ONNXQuantizeLinearOp, 8> deadQuantOps;
      module.walk([&](ONNXQuantizeLinearOp qop) {
        if (qop.getResult().use_empty())
          deadQuantOps.push_back(qop);
      });
      for (ONNXQuantizeLinearOp qop : deadQuantOps) {
        qop.erase();
        changed = true;
      }

      SmallVector<ONNXQLinearMatMulOp, 4> deadQLinearMatMulOps;
      module.walk([&](ONNXQLinearMatMulOp qmm) {
        if (qmm.getResult().use_empty())
          deadQLinearMatMulOps.push_back(qmm);
      });
      for (ONNXQLinearMatMulOp qmm : deadQLinearMatMulOps) {
        qmm.erase();
        changed = true;
      }

      SmallVector<ONNXQLinearConvOp, 4> deadQLinearConvOps;
      module.walk([&](ONNXQLinearConvOp qconv) {
        if (qconv.getResult().use_empty())
          deadQLinearConvOps.push_back(qconv);
      });
      for (ONNXQLinearConvOp qconv : deadQLinearConvOps) {
        qconv.erase();
        changed = true;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createConvertONNXToPositPass(
    unsigned nbits, unsigned es, bool alignToInt8QDomain,
    bool preferDirectF32FromQDQ) {
  return std::make_unique<ConvertONNXToPositPass>(
      nbits, es, alignToInt8QDomain, preferDirectF32FromQDQ);
}
std::unique_ptr<mlir::Pass> createConvertONNXToPositPass() {
  return createConvertONNXToPositPass(8, 0, false, true);
}

} // namespace onnx_mlir
