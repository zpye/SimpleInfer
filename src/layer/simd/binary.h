#ifndef SIMPLE_INFER_SRC_LAYER_SIMD_BINARY_H_
#define SIMPLE_INFER_SRC_LAYER_SIMD_BINARY_H_

#include <cstddef>

namespace SimpleInfer {

void AddBiasNHWC(const float* bias, size_t spatial, size_t oc, float* dst);

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_SRC_LAYER_SIMD_BINARY_H_
