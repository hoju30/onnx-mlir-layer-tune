<!--- SPDX-License-Identifier: Apache-2.0 -->

# Installing `third_party ONNX` for Backend Tests or Rebuilding ONNX Operations

Backend tests are triggered by `make check-onnx-backend` in the build directory and require a few preliminary steps to run successfully. Similarly, rebuilding the ONNX operations in ONNX-MLIR from their ONNX descriptions is triggered by `make OMONNXOpsIncTranslation`.

You will need to install python 3.x if its not default in your environment, and possibly set the cmake `PYTHON_EXECUTABLE` variable in your top cmake file.

You will also need `pybind11` which may need to be installed (mac: `brew install pybind11` or linux: `apt -y install python3-pybind11` for example) and you may need to indicate where to find the software (Mac, POWER, possibly other platforms: `export pybind11_DIR=<your path to pybind>`). Then install the `third_party/onnx` software (Mac: `pip install third_party/onnx`) typed in the top directory.

## Regenerating ONNXOps.td.inc / OpBuildTable.inc reproducibly

`utils/gen_onnx_mlir.py` (invoked by `make OMONNXOpsIncTranslation`, and by
every build via the `add_custom_command` in `utils/CMakeLists.txt`) refuses
to run against any onnx package version other than the one it's pinned to
(the same version `requirements.txt` pins onnx to). If the Python
interpreter CMake picked (`Python3::Interpreter`) already has a different
onnx installed -- easy to end up with if that interpreter is shared with
other projects, or just has a newer onnx than this pin -- the generation
step fails, and prior to this being fixed it failed *silently*: the script
exited 0 on a version mismatch, so the build reported success while quietly
leaving whatever `ONNXOps.td.inc`/`OpBuildTable.inc` already happened to be
on disk (stale, or simply missing on a fresh checkout).

Two ways to fix it:

- Run `pip install -r requirements.txt` for the interpreter CMake is using,
  so its onnx matches the pin exactly.
- Or, if you'd rather not change that interpreter's onnx (e.g. it's shared
  with another project that wants a different version), run
  `utils/regen-onnx-ops.sh` once by hand. It installs the pinned onnx
  version into an isolated `build/.onnx-codegen-env` directory (via `pip
  install --target`, touching nothing else), runs `gen_onnx_mlir.py` against
  just that, and writes `src/Dialect/ONNX/ONNXOps.td.inc` and
  `src/Builder/OpBuildTable.inc` directly -- safe to re-run any time, and
  the install step is skipped on subsequent runs once the isolated
  directory already has the right version. It forwards any arguments to
  `gen_onnx_mlir.py`, so e.g. `utils/regen-onnx-ops.sh
  --check-operation-version` works too.

 ## Upgrading ONNX in ONNX-MLIR

Here are the steps taken to upgrade the ONNX version:

1.	Create your own branch

2.	"cd" into `third_party/onnx` and checkout the commit for the latest version of onnx (You can find the latest commit here: https://github.com/onnx/onnx/releases)

3.	"pip uninstall onnx" (remove older version)

4.	In `onnx-mlir/` directory, "pip install third_party/onnx" (install onnx from the commit and not online version)

5.	Update `utils/gen_onnx_mlir.py` file with the correct version number

6.	Build onnx in the `build/` directory using: set CMAKE_ARGS=-DONNX_USE_LITE_PROTO=ON

7.	Run in the `build/` directory : "make OMONNXOpsIncTranslation" 

8.	Run in `build/` directory : "make onnx-mlir-docs"

9.	Run in `build/` directory : "make check-onnx-backend-case"

10.	Update the [new backend tests](https://github.com/onnx/onnx-mlir/blob/main/test/backend/all_test_names.txt) based on the results from `step 9`

11.	Update the [Opset documentation for cpu](https://github.com/onnx/onnx-mlir/blob/main/test/backend/inference_backend.py) and then issue the following command in the `build/` directory: "make onnx_mlir_supported_ops_cpu"

12.	Update the [Opset documentation for NNPA](https://github.com/onnx/onnx-mlir/blob/main/test/backend/inference_backend.py) and then issue the following command in the `build/` directory: "make onnx_mlir_supported_ops_NNPA"

13.	Ensure the lit tests and backend tests pass successfully and then you are done!


**Note: Please use `git add <filename>` for files that might have been changed before doing a PR.** 

## Known issues

On Macs/POWER and possibly other platforms, there is currently an issue that arises when installing ONNX. If you get an error during the build, try a fix where you edit the top CMakefile as reported in this PR: `https://github.com/onnx/onnx/pull/2482/files`.

While running `make check-onnx-backend` on a Mac you might encounter the following error:

```shell
Fatal Python error: Aborted

Current thread 0x0000000107919e00 (most recent call first):
  File "/usr/local/Cellar/python@3.9/3.9.7/Frameworks/Python.framework/Versions/3.9/lib/python3.9/urllib/request.py", line 2632 in getproxies_macosx_sysconf
  File "/usr/local/Cellar/python@3.9/3.9.7/Frameworks/Python.framework/Versions/3.9/lib/python3.9/urllib/request.py", line 2650 in getproxies
  File "/usr/local/Cellar/python@3.9/3.9.7/Frameworks/Python.framework/Versions/3.9/lib/python3.9/urllib/request.py", line 795 in __init__
  ...
 ```

 A known workaround is to export the `no_proxy` environment variable in your shell as follow, and rerun the tests.

 ```shell
 % export no_proxy="*"
 ```
 
