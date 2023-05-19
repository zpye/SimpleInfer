#ifndef SIMPLE_INFER_SRC_LAYER_SIMD_GEMM_H_
#define SIMPLE_INFER_SRC_LAYER_SIMD_GEMM_H_

#include <cstddef>

namespace SimpleInfer {

void GemmPack4F32(size_t M,
                  size_t N,
                  size_t K,
                  const float* A,
                  size_t lda,
                  const float* B,
                  float* C,
                  size_t ldc);

void GemmPack4F32Ref(size_t M,
                     size_t N,
                     size_t K,
                     const float* A,
                     size_t lda,
                     const float* B,
                     float* C,
                     size_t ldc);

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_SRC_LAYER_SIMD_GEMM_H_
