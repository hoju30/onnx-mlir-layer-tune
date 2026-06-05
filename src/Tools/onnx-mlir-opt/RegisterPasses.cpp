/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- RegisterPasses.cpp -------------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// Beware that we cannot access CompilerOptions or other command line options
// the functions below because they are called before command line options are
// parsed, because passes must be registered first as they instruct command
// line parsing: registered passes can be expressed as command line flags.
//
// In particular the --O flag value is passed as a function argument optLevel
// so we can avoid reading the OptimizationLevel command-line option here.
//
//===----------------------------------------------------------------------===//

#include "RegisterPasses.hpp"

#include <limits>
#include <cstdlib>

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "src/Accelerators/Accelerator.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerPasses.hpp"

#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "src/Pass/Passes.hpp"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"

#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"

#include "src/Dialect/Posit/PositOps.h"

using namespace mlir;
using namespace mlir;
using namespace posit;

namespace onnx_mlir {

namespace {
struct PositFormatConfig {
  unsigned nbits;
  unsigned es;
};

llvm::cl::opt<unsigned> positNbitsOpt("posit-nbits",
    llvm::cl::desc(
        "Posit total bit-width used by convert-onnx-to-posit and convert-posit-to-krnl"),
    llvm::cl::cat(OnnxMlirOptOptions), llvm::cl::init(8));

llvm::cl::opt<unsigned> positEsOpt("posit-es",
    llvm::cl::desc(
        "Posit exponent-size used by convert-onnx-to-posit and convert-posit-to-krnl"),
    llvm::cl::cat(OnnxMlirOptOptions), llvm::cl::init(0));

llvm::cl::opt<std::string> positFormatOpt("posit-format",
    llvm::cl::desc(
        "Posit format override in p<nbits>e<es> form "
        "(supported: p8/p16/p32 with e0/e1/e2, plus p4..p7 and p9 with e0..e3); "
        "when set, overrides --posit-nbits/--posit-es"),
    llvm::cl::cat(OnnxMlirOptOptions), llvm::cl::NotHidden,
    llvm::cl::init(""));

llvm::cl::opt<bool> alignToInt8QDomainOpt("align-to-int8-qdomain",
    llvm::cl::desc(
        "Require ONNX Q->DQ lowering to stay aligned with INT8 quantization "
        "domain (scale/zero_point) in convert-onnx-to-posit"),
    llvm::cl::cat(OnnxMlirOptOptions), llvm::cl::init(false));

llvm::cl::opt<bool> strictQDQModeOpt("strict-qdq-lowering",
    llvm::cl::desc(
        "Use strict ONNX QDQ semantics first in convert-onnx-to-posit; "
        "when OFF (default), prefer direct f32->posit from Q source"),
    llvm::cl::cat(OnnxMlirOptOptions), llvm::cl::init(false));

static bool isSupportedPositFormat(unsigned nbits, unsigned es) {
  // Existing stable set.
  if (nbits == 8 || nbits == 16 || nbits == 32)
    return es <= 2;
  // Extra low-bit experimental set requested by user flow.
  if ((nbits >= 4 && nbits <= 7) || nbits == 9)
    return es <= 3;
  return false;
}

static bool parsePositFormat(
    llvm::StringRef format, PositFormatConfig &cfg, std::string &errMsg) {
  llvm::StringRef s = format.trim();
  if (s.empty()) {
    errMsg = "empty format";
    return false;
  }
  if (!(s.consume_front("p") || s.consume_front("P"))) {
    errMsg = "missing 'p' prefix";
    return false;
  }

  size_t ePos = s.find_first_of("eE");
  if (ePos == llvm::StringRef::npos) {
    errMsg = "missing 'e' separator";
    return false;
  }

  llvm::StringRef nbitsStr = s.substr(0, ePos);
  llvm::StringRef esStr = s.substr(ePos + 1);
  if (nbitsStr.empty() || esStr.empty()) {
    errMsg = "both nbits and es must be provided";
    return false;
  }

  unsigned long long nbitsLL = 0;
  unsigned long long esLL = 0;
  if (nbitsStr.getAsInteger(10, nbitsLL) || esStr.getAsInteger(10, esLL)) {
    errMsg = "nbits/es must be unsigned integers";
    return false;
  }

  if (nbitsLL == 0) {
    errMsg = "nbits must be > 0";
    return false;
  }
  if (nbitsLL > std::numeric_limits<unsigned>::max() ||
      esLL > std::numeric_limits<unsigned>::max()) {
    errMsg = "nbits/es out of range for unsigned";
    return false;
  }

  cfg.nbits = static_cast<unsigned>(nbitsLL);
  cfg.es = static_cast<unsigned>(esLL);
  return true;
}

static PositFormatConfig getPositFormatConfig() {
  auto validateOrExit = [](PositFormatConfig cfg) -> PositFormatConfig {
    if (!isSupportedPositFormat(cfg.nbits, cfg.es)) {
      llvm::errs()
          << "error: unsupported posit format p" << cfg.nbits << "e" << cfg.es
          << ". Supported formats are: "
          << "p8/p16/p32 with e0..e2, and p4..p7/p9 with e0..e3.\n";
      std::exit(1);
    }
    return cfg;
  };

  if (positFormatOpt.empty())
    return validateOrExit({positNbitsOpt, positEsOpt});

  PositFormatConfig cfg{0, 0};
  std::string errMsg;
  if (!parsePositFormat(positFormatOpt, cfg, errMsg)) {
    llvm::errs() << "error: invalid --posit-format='" << positFormatOpt
                 << "': " << errMsg
                 << ". Expected p<nbits>e<es>, e.g. p8e0 or p32e2.\n";
    std::exit(1);
  }
  return validateOrExit(cfg);
}
} // namespace

void registerOMPasses(int optLevel) {
  // All passes implemented within onnx-mlir should register within this
  // function to make themselves available as a command-line option.

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createScrubDisposablePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createONNXOpTransformPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createDecomposeONNXToONNXPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createRecomposeONNXToONNXPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createConvOptONNXToONNXPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createReplaceOpWithItsOperandPass(/*nodeNameRegexList*/ {});
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createONNXHybridTransformPass(/*recompose ops*/ true);
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createShapeInferencePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createConstPropONNXToONNXPass();
  });

  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createInstrumentPass(); });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createInstrumentCleanupPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createInstrumentONNXSignaturePass("NONE", "NONE");
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createSetONNXNodeNamePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createONNXPreKrnlVerifyPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return krnl::createConvertKrnlToAffinePass();
  });

  mlir::registerPass([optLevel]() -> std::unique_ptr<mlir::Pass> {
    return createLowerToKrnlPass(/*enableTiling*/ optLevel >= 3,
        /*enableSIMD, should consider disableSimdOption*/ optLevel >= 3,
        /*enableParallel*/ false,
        /*enableFastMath*/ false, /*default is still off*/
        /*opsForCall*/ "");
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    PositFormatConfig cfg = getPositFormatConfig();
    return createConvertONNXToPositPass(
        cfg.nbits, cfg.es, alignToInt8QDomainOpt, !strictQDQModeOpt);
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    PositFormatConfig cfg = getPositFormatConfig();
    return onnx_mlir::createConvertPositToKrnlPass(cfg.nbits, cfg.es);
  });


  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createProcessScfParallelPrivatePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createProcessKrnlParallelClausePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return krnl::createConvertSeqToMemrefPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return krnl::createLowerKrnlRegionPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return krnl::createConvertKrnlToLLVMPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createSimplifyShapeRelatedOpsPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createStandardFuncReturnPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createONNXDimAnalysisPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createConvertONNXToTOSAPass();
  });

#ifdef ONNX_MLIR_ENABLE_STABLEHLO
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createLowerToStablehloPass();
  });
#endif
}

void registerMLIRPasses() {
  // Register passes created from Passes.td
  mlir::registerTransformsPasses();
  onnx_mlir::registerTransformsPasses();

  affine::registerAffinePasses();
  func::registerFuncPasses();
  registerLinalgPasses();
  memref::registerMemRefPasses();
  registerSCFPasses();
  bufferization::registerBufferizationPasses();

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createConvertVectorToSCFPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createLowerAffinePass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createSCFToControlFlowPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createConvertVectorToLLVMPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createPrintOpStatsPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createConvertSCFToOpenMPPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createConvertOpenMPToLLVMPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createFinalizeMemRefToLLVMConversionPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createReconcileUnrealizedCastsPass();
  });
}

void registerPasses(int optLevel) {
  registerMLIRPasses();

  registerOMPasses(optLevel);

  // Register passes for accelerators.
  for (auto *accel : accel::Accelerator::getAccelerators())
    accel->registerPasses(optLevel);
}

} // namespace onnx_mlir
