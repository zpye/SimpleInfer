# SimpleInfer

SimpleInfer is a neural network inference framework based on [KuiperInfer](https://github.com/zjhellofss/KuiperInfer).

## Build

SimpleInfer uses [xmake](https://xmake.io/#/) to build library and tests.

```shell
git clone --recursive 
cd SimpleInfer
xmake config -m release
xmake
```
## Run

After building successfully, run test-engin to check.

```shell
xmake run test-engine
```

## Reference

[KuiperInfer](https://github.com/zjhellofss/KuiperInfer) -> basic framework

[ncnn](https://github.com/Tencent/ncnn) -> pnnx ir, simpleocv and mat-pixel operations

[Eigen](https://gitlab.com/libeigen/eigen), [tensorflow](https://github.com/tensorflow/tensorflow) -> Eigen tensor

[abseil](https://github.com/abseil/abseil-cpp) -> logging, string format operations

[CGraph](https://github.com/ChunelFeng/CGraph) -> graph based pipeline

[benchmark](https://github.com/google/benchmark), [Catch2](https://github.com/catchorg/Catch2) -> benchmark and unit tests
