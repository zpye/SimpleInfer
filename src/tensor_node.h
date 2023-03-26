#ifndef SIMPLE_INFER_SRC_TENSOR_NODE_H_
#define SIMPLE_INFER_SRC_TENSOR_NODE_H_

#include "pnnx/ir.h"
#include "tensor.h"

namespace SimpleInfer {

struct TensorNode {
    pnnx::Operand* operand = nullptr;
    Tensor tensor;
};

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_SRC_TENSOR_NODE_H_
