#ifndef SIMPLE_INFER_SRC_LAYER_SIMD_WINOGRAD_HELPER_H_
#define SIMPLE_INFER_SRC_LAYER_SIMD_WINOGRAD_HELPER_H_

#include <cstddef>

namespace SimpleInfer {

void Conv3x3s1Winograd23TransformKernelPack4(const float* src,
                                             size_t ic,
                                             size_t oc,
                                             float* dst);

void Conv3x3s1Winograd23TransformInput(const float* src,
                                       size_t ih,
                                       size_t iw,
                                       size_t ic,
                                       bool pad,
                                       float* dst,
                                       size_t dst_stride);

void Conv3x3s1Winograd23TransformOutput(const float* src,
                                        size_t src_stride,
                                        float* dst,
                                        size_t oh,
                                        size_t ow,
                                        size_t oc);

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_SRC_LAYER_SIMD_WINOGRAD_HELPER_H_
