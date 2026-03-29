SOFTPOSIT_DIR=/home/lai/mlir_toy/SoftPosit/SoftPosit
SOFTPOSIT_BUILD=/home/lai/mlir_toy/SoftPosit/SoftPosit/build/Linux_x86_64_GCC
SOFTPOSIT_INC=$SOFTPOSIT_DIR/source/include

LLVM_SRC=/home/lai/mlir_toy/llvm-project
LLVM_BUILD=/home/lai/mlir_toy/llvm-project/build


../../../../llvm-project/build/bin/mlir-translate --mlir-to-llvmir ../../../positllvm2.mlir  -o test2.ll

clang++ -O2 -fPIC -shared \
  test.ll posit_runtime.cpp \
  -I$SOFTPOSIT_INC \
  -I/home/lai/mlir_toy/llvm-project/mlir/include \
  -I$LLVM_BUILD/include \
  -L$SOFTPOSIT_BUILD -lsoftposit \
  -Wl,-rpath,$SOFTPOSIT_BUILD \
  -o libtest_posit.so
  
// run ll .so檔
clang++ -O2 run.fixed.cpp \
  -I$SOFTPOSIT_INC \
  -I$LLVM_SRC/mlir/include \
  -I$LLVM_BUILD/tools/mlir/include \
  -I$LLVM_BUILD/include \
  -L$SOFTPOSIT_BUILD -lsoftposit \
  -Wl,-rpath,$SOFTPOSIT_BUILD \
  -ldl \
  -o run
// clang++ -O2 run.cpp -ldl -o run
./run

// run model
./run ./mnist-12-posit.so(so) mnist.txt(測資) --warmup 0 --iters 200

// 多個比較
hyperfine --warmup 5 --runs 30 "指令一" "指令二"
// 多比較 自行寫的sh
./scripts/build_softposit_px1_lib.sh
./scripts/build_px1_targets.sh mnist-12-qdq-p8e0.ll(檔名要換)



//rebuild onnx-mlir-opt
cmake -G Ninja .. -DMLIR_DIR=/home/lai/mlir_toy/llvm-project/build/lib/cmake/mlir -DLLVM_DIR=/home/lai/mlir_toy/llvm-project/build/lib/cmake/llvm -DCMAKE_BUILD_TYPE=Release

ninja onnx-mlir-opt


//build model to onnx
./onnx-mlir --EmitONNXIR mnist-12.onnx(檔名) 
不省略：
./onnx-mlir --EmitONNXIR --mlir-elide-resource-strings-if-larger=1000000000 --mlir-elide-elementsattrs-if-larger=1000000000 

//每一階段單獨看
./onnx-mlir-opt --convert-onnx-to-posit ../../../testonnxtoposit2.mlir -o -
./onnx-mlir-opt --convert-posit-to-krnl ../../../posittokrnl4.mlir -o -
./onnx-mlir-opt --convert-krnl-to-llvm --canonicalize --cse  ../../../testkrnl4.mlir -o ../../../test3.ll

//一次生成11.so
bash scripts/build_mobilenet11_sos.sh 可加生成檔案位置
./build_mobilenet11_sos.sh ./temp/mobilenet11_temp
# 一次跑完整 11 模型 pairwise 
bash scripts/compare_model11_dataset.sh \
  --model-name mobilenetv2-12 \
  --out-dir /tmp/mobilenet11 \
  --txt-dir datasets/vision/processed/imagenette_val_224 \
  --shape 1,3,224,224 --limit 100 --warmup 2 --iters 5 --progress 2

# 單so測試
bash ./time_model_single_format.sh \  
    --model-name mobilenetv2-12   \
    --format p8e0   \
    --out-dir ./temp/mobilenet11   \
    --txt-dir ./temp/imagenette_val_224   \
    --shape 1x3x224x224   --limit 100   \
    --label-map ./temp/imagenette_val_224_labels.txt   \
    --baseline none   --no-benchmark   --progress 1   \
    --log-file ./temp/mobilenet11/p8e0_limit100_label_nobase.log

# resnet
bash ./time_model_single_format.sh   \
    --model-name resnet50-v1-12   \
    --format p8e0   \
    --out-dir ./temp/resnet50-11_temp   \
    --txt-dir ./temp/imagenette_val_224   \
    --shape 1x3x224x224   --limit 1000   \
    --label-map ./temp/imagenette_val_224_labels.txt   \
    --baseline none   --no-benchmark   --progress 1   \
    --log-file ./temp/resnet50-11_temp/p8e0_limit100_label_nobase.log

# 全so 平行多核
bash/time_mobilenet11_dataset_parallel.sh \
  --out-dir ./temp/mobilenet11_temp \
  --txt-dir ./temp/imagenette_val_224 \
  --label-map ./temp/imagenette_val_224_labels.txt \
  --jobs 4 \
  --limit 100 \
  --warmup 0 \
  --iters 1 \
  --no-benchmark \
  --progress 10

bash/time_resnet50_11_dataset_parallel.sh \
  --out-dir ./temp/resnet50-11_temp \
  --txt-dir ./temp/imagenette_val_224 \
  --label-map ./temp/imagenette_val_224_labels.txt \
  --jobs 4 \
  --limit 1 \
  --warmup 0 \
  --iters 1 \
  --no-benchmark \
  --progress 1
