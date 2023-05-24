# SimpleInfer

SimpleInfer is a neural network inference framework based on [KuiperInfer](https://github.com/zjhellofss/KuiperInfer).

## Build

SimpleInfer uses [xmake](https://xmake.io/#/) to build library and tests.

```shell
git clone --recursive https://github.com/zpye/SimpleInfer.git
cd SimpleInfer
xmake config -a x64 -m release
xmake -w --all
```
## Run

After building successfully, run test-yolo to check.

```shell
xmake run test-yolo
```

## YOLO Result

Here are visualized results of YOLO detection.

![result_31.jpg](imgs/result_31.jpg)

![result_bus.jpg](imgs/result_bus.jpg)

![result_car.jpg](imgs/result_car.jpg)

![result_zidane.jpg](imgs/result_zidane.jpg)

## Working With Python

1. Set environment `PYTHON_ROOT` where `python` binary exists, pybind11 needs `${PYTHON_ROOT}/include` and `${PYTHON_ROOT}/libs` for compiling and linking.

2. Set `--build_python=true` after `xmake config` and build: 

```shell
xmake config -a x64 -m release --builf_python=true
xmake -w --all
```

3. install package by pip:

```shell
pip install build/python/
```

4. run python test:

```shell
python test/test_python/test_model.py
```

## Reference

[KuiperInfer](https://github.com/zjhellofss/KuiperInfer) -> basic framework

[ncnn](https://github.com/Tencent/ncnn) -> pnnx ir, simpleocv and mat-pixel operations

[Eigen](https://gitlab.com/libeigen/eigen), [tensorflow](https://github.com/tensorflow/tensorflow) -> Eigen tensor

[abseil](https://github.com/abseil/abseil-cpp) -> logging, string format operations

[CGraph](https://github.com/ChunelFeng/CGraph) -> graph based pipeline

[highway](https://github.com/google/highway), [Simd](https://github.com/ermig1979/Simd) -> SIMD, GEMM, Winograd, parallel

[benchmark](https://github.com/google/benchmark), [Catch2](https://github.com/catchorg/Catch2) -> benchmark and unit tests

[pybind11](https://github.com/pybind/pybind11) -> python bindings of c++

[tmp](https://github.com/zjhellofss/tmp) -> pnnx models
