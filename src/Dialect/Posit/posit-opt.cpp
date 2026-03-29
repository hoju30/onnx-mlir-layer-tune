#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/InitLLVM.h"

#include "mlir/Tools/mlir-opt/MlirOptMain.h" // MlirOptMain 驅動
#include "src/Dialect/Posit/PositDialect.h" // 註冊自訂方言
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "src/Dialect/Posit/PositPasses.h"  // 註冊自訂pipeline
//#define GEN_PASS_REGISTRATION             // 新增
//#include "posit/PositPasses.h.inc"

#include "mlir/Pass/PassRegistry.h"

// 10/7 new
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

// 10/15 new
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

// 11/12 new
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/Posit/PositOps.h"




using namespace mlir;

// 10/14
namespace {
namespace cl = llvm::cl;

// 位置參數：輸入 MLIR（預設 stdin）
static cl::opt<std::string> inputFilename(
    cl::Positional, cl::desc("<input mlir>"), cl::init("-"),
    cl::value_desc("filename"));

// -o：輸出檔（預設 stdout）
static cl::opt<std::string> outputFilename(
    "o", cl::desc("Output filename"), cl::value_desc("filename"),
    cl::init("-"));

// 是否執行 posit → arith 的 lowering（預設開）
static cl::opt<bool> lowerToArithOpt(
    "lower-posit-to-arith",
    cl::desc("Lower posit dialect to arith/func/builtin"),
    cl::init(true));

// 是否輸出 LLVM IR（預設否；否則輸出 MLIR）
static cl::opt<bool> emitLLVMIR(
    "emit-llvmir", cl::desc("Translate to and print LLVM IR"),
    cl::init(false));

// new (arith lowering to llvm)
static cl::opt<bool> lowerArithToLLVM(
  "lower-arith-to-llvm",
  llvm::cl::desc("Lower arith/func/memref to LLVM dialect"),
  llvm::cl::init(false));

} // namespace

// 10/29 new
std::unique_ptr<mlir::Pass> createConvertONNXToPositPass();



// 10/14 new
static LogicalResult runPipelineAndPrint(ModuleOp module, MLIRContext &ctx) {
  PassManager pm(&ctx);
  // 讓 -pm- 前綴的 PM 旗標可用（例如列印每步 IR/計時）。:contentReference[oaicite:0]{index=0}
  applyPassManagerCLOptions(pm);

  if (lowerToArithOpt)
    pm.addPass(mlir::posit::createLowerPositToArithPass());


  // lowering arith to llvm 10/15 new
  if (lowerArithToLLVM) {
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  } 

  if (failed(pm.run(module)))
    return failure();

  // 準備輸出
  auto out = mlir::openOutputFile(outputFilename);
  if (!out) {
    llvm::errs() << "cannot open output: " << outputFilename << "\n";
    return failure();
  }

  if (!emitLLVMIR) {
    module->print(out->os());
    out->keep();
    return success();
  }


  ctx.loadDialect<mlir::LLVM::LLVMDialect>();
  // 需要將 MLIR 轉 LLVM IR：先把 Builtin/LLVM 的翻譯器掛進 context。
  mlir::registerBuiltinDialectTranslation(ctx);
  mlir::registerLLVMDialectTranslation(ctx);

  llvm::LLVMContext llvmCtx;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmCtx); 
  if (!llvmModule) {
    llvm::errs() << "translateModuleToLLVMIR failed\n";
    return failure();
  }

  // 設定目標 triple / datalayout（可讓印出的 LLVM IR 更完整）。
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  auto tmBuilderOrErr = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrErr) {
    llvm::errs() << "detectHost() failed\n";
    return failure();
  }
  auto tmOrErr = tmBuilderOrErr->createTargetMachine();
  if (!tmOrErr) {
    llvm::errs() << "createTargetMachine() failed\n";
    return failure();
  }
  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(
      llvmModule.get(), tmOrErr.get().get());

  out->os() << *llvmModule << "\n";
  out->keep();
  return success();
}


//old
int main(int argc, char **argv) {
  //mlir::registerMLIRContextCLOptions();
  //mlir::registerPassManagerCLOptions(); // 10/14 try hello simple
  llvm::InitLLVM y(argc, argv);

  mlir::registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "posit-opt (hello-style)\n");

  DialectRegistry registry;
  registry.insert<mlir::posit::PositDialect,
                  mlir::func::FuncDialect,
                  mlir::arith::ArithDialect,
                  mlir::tensor::TensorDialect,
                  mlir::ONNX::ONNXDialect,
                  mlir::LLVM::LLVMDialect>();

  
                  
  //mlir::registerAllPasses();
  //mlir::registerPositPasses();  //  10/7 找不到此pass  positpasses.h.inc有順利產出 93行 但依然報錯

  // 10/14 new
  mlir::MLIRContext ctx;
  ctx.appendDialectRegistry(registry);
  ctx.loadDialect<mlir::posit::PositDialect,
                  mlir::func::FuncDialect,
                  mlir::arith::ArithDialect>();

  auto file = mlir::openInputFile(inputFilename);
  if (!file) {
    llvm::errs() << "cannot open input: " << inputFilename << "\n";
    return 1;
  }
  llvm::SourceMgr sm;
  sm.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sm, &ctx);  // 標準用法。
  if (!module) {
    llvm::errs() << "failed to parse input\n";
    return 1;
  }

  // 執行 pass pipeline 並輸出（MLIR 或 LLVM IR）
  if (failed(runPipelineAndPrint(*module, ctx)))
    return 2;

  return 0;
 
  /*return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Posit optimizer driver", registry));*/
}