/*
 * Simd Library (http://ermig1979.github.io/Simd).
 *
 * Copyright (c) 2011-2021 Yermalayeu Ihar.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "winograd_helper.h"

#include <cassert>
#include <vector>

#include "hwy/highway.h"

namespace hwy {
namespace HWY_NAMESPACE {

static const Full128<float> d;
static_assert(4 == Lanes(d), "Lanes(Full128<float>) should be 4");
using f32x4_t = VFromD<Full128<float>>;

// [3(kh)][3(kw)][ic][oc] -> [4(gh)][4(gw)][oc/4][ic][4(oc)]
void Conv3x3s1Winograd23TransformKernelPack4(const float* src,
                                             size_t ic,
                                             size_t oc,
                                             float* dst) {
    // padding oc with 4
    size_t oc_up4 = (oc + 3) / 4 * 4;

    // 1. [3(kh)][3(kw)][ic][oc] -> [3(kh)][3(kw)][ic][oc4]
    std::vector<float> hwio(9 * ic * oc_up4);
    for (size_t k = 0; k < 9; ++k) {
        for (size_t i = 0; i < ic; ++i) {
            size_t j = 0;
            for (; j < oc; ++j) {
                hwio[k * ic * oc_up4 + i * oc_up4 + j] =
                    src[k * ic * oc + i * oc + j];
            }
            for (; j < oc_up4; ++j) {
                hwio[k * ic * oc_up4 + i * oc_up4 + j] = 0.0f;
            }
        }
    }

    // 2. [3(kh)][3(kw)][ic][oc4] -> [4(gh)][4(gw)][ic][oc4]
    std::vector<float> ghwio(16 * ic * oc_up4);
    size_t stride = ic * oc_up4;  // stride % 4 == 0

    {
        const f32x4_t r2 = Set(d, 1.0f / 2.0f);
        const f32x4_t r4 = Set(d, 1.0f / 4.0f);

        f32x4_t temp[9];
        for (size_t i = 0; i < stride; i += 4) {
            const float* in = hwio.data() + i;
            float* out      = ghwio.data() + i;

            // load 3x3
            temp[0] = LoadU(d, in + 0 * stride);
            temp[1] = LoadU(d, in + 1 * stride);
            temp[2] = LoadU(d, in + 2 * stride);
            temp[3] = LoadU(d, in + 3 * stride);
            temp[4] = LoadU(d, in + 4 * stride);
            temp[5] = LoadU(d, in + 5 * stride);
            temp[6] = LoadU(d, in + 6 * stride);
            temp[7] = LoadU(d, in + 7 * stride);
            temp[8] = LoadU(d, in + 8 * stride);

            // compute GgGT
            {
                StoreU(temp[0], d, out + 0 * stride);
                const f32x4_t _0a2 = Add(temp[0], temp[2]);
                StoreU(Mul(Add(_0a2, temp[1]), r2), d, out + 1 * stride);
                StoreU(Mul(Sub(_0a2, temp[1]), r2), d, out + 2 * stride);
                StoreU(temp[2], d, out + 3 * stride);
            }

            {
                const f32x4_t _0a6a3 = Add(Add(temp[0], temp[6]), temp[3]);
                StoreU(Mul(_0a6a3, r2), d, out + 4 * stride);
                const f32x4_t _2a8a5 = Add(Add(temp[2], temp[8]), temp[5]);
                const f32x4_t _1a7a4 = Add(Add(temp[1], temp[7]), temp[4]);
                StoreU(Mul(Add(Add(_0a6a3, _2a8a5), _1a7a4), r4),
                       d,
                       out + 5 * stride);
                StoreU(Mul(Sub(Add(_0a6a3, _2a8a5), _1a7a4), r4),
                       d,
                       out + 6 * stride);
                StoreU(Mul(_2a8a5, r2), d, out + 7 * stride);
            }

            {
                const f32x4_t _0a6s3 = Sub(Add(temp[0], temp[6]), temp[3]);
                StoreU(Mul(_0a6s3, r2), d, out + 8 * stride);
                const f32x4_t _2a8s5 = Sub(Add(temp[2], temp[8]), temp[5]);
                const f32x4_t _1a7s4 = Sub(Add(temp[1], temp[7]), temp[4]);
                StoreU(Mul(Add(Add(_0a6s3, _2a8s5), _1a7s4), r4),
                       d,
                       out + 9 * stride);
                StoreU(Mul(Sub(Add(_0a6s3, _2a8s5), _1a7s4), r4),
                       d,
                       out + 10 * stride);
                StoreU(Mul(_2a8s5, r2), d, out + 11 * stride);
            }

            {
                StoreU(temp[6], d, out + 12 * stride);
                const f32x4_t _6a8 = Add(temp[6], temp[8]);
                StoreU(Mul(Add(_6a8, temp[7]), r2), d, out + 13 * stride);
                StoreU(Mul(Sub(_6a8, temp[7]), r2), d, out + 14 * stride);
                StoreU(temp[8], d, out + 15 * stride);
            }
        }
    }

    // 3. [4(gh)][4(gw)][ic][oc4] -> [4(gh)][4(gw)][oc/4][ic][4(oc)]
    for (size_t k = 0; k < 16; ++k) {
        for (size_t j = 0; j < oc_up4; j += 4) {
            const float* in = ghwio.data() + k * stride + j;
            for (size_t i = 0; i < ic; ++i) {
                StoreU(LoadU(d, in + i * oc_up4), d, dst);
                dst += 4;
            }
        }
    }
}

// load full 4x4 input
inline void Conv3x3s1Winograd23TransformInput4tLoad(const float* src,
                                                    size_t src_stride,
                                                    size_t ic,
                                                    f32x4_t dst[16]) {
    dst[0]  = LoadU(d, src + 0 * src_stride + 0 * ic);
    dst[1]  = LoadU(d, src + 0 * src_stride + 1 * ic);
    dst[2]  = LoadU(d, src + 0 * src_stride + 2 * ic);
    dst[3]  = LoadU(d, src + 0 * src_stride + 3 * ic);
    dst[4]  = LoadU(d, src + 1 * src_stride + 0 * ic);
    dst[5]  = LoadU(d, src + 1 * src_stride + 1 * ic);
    dst[6]  = LoadU(d, src + 1 * src_stride + 2 * ic);
    dst[7]  = LoadU(d, src + 1 * src_stride + 3 * ic);
    dst[8]  = LoadU(d, src + 2 * src_stride + 0 * ic);
    dst[9]  = LoadU(d, src + 2 * src_stride + 1 * ic);
    dst[10] = LoadU(d, src + 2 * src_stride + 2 * ic);
    dst[11] = LoadU(d, src + 2 * src_stride + 3 * ic);
    dst[12] = LoadU(d, src + 3 * src_stride + 0 * ic);
    dst[13] = LoadU(d, src + 3 * src_stride + 1 * ic);
    dst[14] = LoadU(d, src + 3 * src_stride + 2 * ic);
    dst[15] = LoadU(d, src + 3 * src_stride + 3 * ic);
}

// load partial 4x4 input
inline void Conv3x3s1Winograd23TransformInput4tLoad(const float* src,
                                                    size_t src_stride,
                                                    size_t ic,
                                                    size_t row_start,
                                                    size_t row_end,
                                                    size_t col_start,
                                                    size_t col_end,
                                                    f32x4_t dst[16]) {
    for (size_t i = 0; i < 16; ++i) {
        dst[i] = Zero(d);
    }

    for (size_t row = row_start; row < row_end; ++row) {
        for (size_t col = col_start; col < col_end; ++col) {
            dst[row * 4 + col] = LoadU(d, src + row * src_stride + col * ic);
        }
    }
}

inline void Conv3x3s1Winograd23TransformInput4tStore(const f32x4_t src[16],
                                                     float* dst,
                                                     size_t dst_stride) {
    StoreU(Sub(Sub(src[0], src[8]), Sub(src[2], src[10])),
           d,
           dst + 0 * dst_stride);
    StoreU(Add(Sub(src[1], src[9]), Sub(src[2], src[10])),
           d,
           dst + 1 * dst_stride);
    StoreU(Sub(Sub(src[2], src[10]), Sub(src[1], src[9])),
           d,
           dst + 2 * dst_stride);
    StoreU(Sub(Sub(src[1], src[9]), Sub(src[3], src[11])),
           d,
           dst + 3 * dst_stride);
    StoreU(Sub(Add(src[4], src[8]), Add(src[6], src[10])),
           d,
           dst + 4 * dst_stride);
    StoreU(Add(Add(src[5], src[9]), Add(src[6], src[10])),
           d,
           dst + 5 * dst_stride);
    StoreU(Sub(Add(src[6], src[10]), Add(src[5], src[9])),
           d,
           dst + 6 * dst_stride);
    StoreU(Sub(Add(src[5], src[9]), Add(src[7], src[11])),
           d,
           dst + 7 * dst_stride);
    StoreU(Sub(Sub(src[8], src[4]), Sub(src[10], src[6])),
           d,
           dst + 8 * dst_stride);
    StoreU(Add(Sub(src[9], src[5]), Sub(src[10], src[6])),
           d,
           dst + 9 * dst_stride);
    StoreU(Sub(Sub(src[10], src[6]), Sub(src[9], src[5])),
           d,
           dst + 10 * dst_stride);
    StoreU(Sub(Sub(src[9], src[5]), Sub(src[11], src[7])),
           d,
           dst + 11 * dst_stride);
    StoreU(Sub(Sub(src[4], src[12]), Sub(src[6], src[14])),
           d,
           dst + 12 * dst_stride);
    StoreU(Add(Sub(src[5], src[13]), Sub(src[6], src[14])),
           d,
           dst + 13 * dst_stride);
    StoreU(Sub(Sub(src[6], src[14]), Sub(src[5], src[13])),
           d,
           dst + 14 * dst_stride);
    StoreU(Sub(Sub(src[5], src[13]), Sub(src[7], src[15])),
           d,
           dst + 15 * dst_stride);
}

inline void Conv3x3s1Winograd23TransformInput1tStore(const float src[16],
                                                     float* dst,
                                                     size_t dst_stride) {
    dst[0 * dst_stride]  = (src[0] - src[8]) - (src[2] - src[10]);
    dst[1 * dst_stride]  = (src[1] - src[9]) + (src[2] - src[10]);
    dst[2 * dst_stride]  = (src[2] - src[10]) - (src[1] - src[9]);
    dst[3 * dst_stride]  = (src[1] - src[9]) - (src[3] - src[11]);
    dst[4 * dst_stride]  = (src[4] + src[8]) - (src[6] + src[10]);
    dst[5 * dst_stride]  = (src[5] + src[9]) + (src[6] + src[10]);
    dst[6 * dst_stride]  = (src[6] + src[10]) - (src[5] + src[9]);
    dst[7 * dst_stride]  = (src[5] + src[9]) - (src[7] + src[11]);
    dst[8 * dst_stride]  = (src[8] - src[4]) - (src[10] - src[6]);
    dst[9 * dst_stride]  = (src[9] - src[5]) + (src[10] - src[6]);
    dst[10 * dst_stride] = (src[10] - src[6]) - (src[9] - src[5]);
    dst[11 * dst_stride] = (src[9] - src[5]) - (src[11] - src[7]);
    dst[12 * dst_stride] = (src[4] - src[12]) - (src[6] - src[14]);
    dst[13 * dst_stride] = (src[5] - src[13]) + (src[6] - src[14]);
    dst[14 * dst_stride] = (src[6] - src[14]) - (src[5] - src[13]);
    dst[15 * dst_stride] = (src[5] - src[13]) - (src[7] - src[15]);
}

template<size_t F>
void Conv3x3s1Winograd23TransformInputFt(const float* src,
                                         size_t iw,
                                         size_t ic,
                                         float* dst,
                                         size_t dst_stride);

template<size_t F>
void Conv3x3s1Winograd23TransformInputFt(const float* src,
                                         size_t iw,
                                         size_t ic,
                                         size_t row_start,
                                         size_t row_end,
                                         size_t col_start,
                                         size_t col_end,
                                         float* dst,
                                         size_t dst_stride);

template<>
void Conv3x3s1Winograd23TransformInputFt<4>(const float* src,
                                            size_t iw,
                                            size_t ic,
                                            float* dst,
                                            size_t dst_stride) {
    size_t src_stride = iw * ic;
    size_t ic4        = ic / 4 * 4;

    f32x4_t temp[16];

    for (size_t c = 0; c < ic4; c += 4) {
        Conv3x3s1Winograd23TransformInput4tLoad(src + c, src_stride, ic, temp);
        Conv3x3s1Winograd23TransformInput4tStore(temp, dst + c, dst_stride);
    }

    if (ic4 < ic) {
        Conv3x3s1Winograd23TransformInput4tLoad(src + ic - 4,
                                                src_stride,
                                                ic,
                                                temp);
        Conv3x3s1Winograd23TransformInput4tStore(temp,
                                                 dst + ic - 4,
                                                 dst_stride);
    }
}

template<>
void Conv3x3s1Winograd23TransformInputFt<4>(const float* src,
                                            size_t iw,
                                            size_t ic,
                                            size_t row_start,
                                            size_t row_end,
                                            size_t col_start,
                                            size_t col_end,
                                            float* dst,
                                            size_t dst_stride) {
    size_t src_stride = iw * ic;
    size_t ic4        = ic / 4 * 4;

    f32x4_t temp[16];

    for (size_t c = 0; c < ic4; c += 4) {
        Conv3x3s1Winograd23TransformInput4tLoad(src + c,
                                                src_stride,
                                                ic,
                                                row_start,
                                                row_end,
                                                col_start,
                                                col_end,
                                                temp);
        Conv3x3s1Winograd23TransformInput4tStore(temp, dst + c, dst_stride);
    }

    if (ic4 < ic) {
        Conv3x3s1Winograd23TransformInput4tLoad(src + ic - 4,
                                                src_stride,
                                                ic,
                                                row_start,
                                                row_end,
                                                col_start,
                                                col_end,
                                                temp);
        Conv3x3s1Winograd23TransformInput4tStore(temp,
                                                 dst + ic - 4,
                                                 dst_stride);
    }
}

template<>
void Conv3x3s1Winograd23TransformInputFt<1>(const float* src,
                                            size_t iw,
                                            size_t ic,
                                            float* dst,
                                            size_t dst_stride) {
    size_t src_stride = iw * ic;

    for (size_t c = 0; c < ic; ++c) {
        float temp[16] = {0};

        temp[0]  = src[0 * src_stride + 0 * ic];
        temp[1]  = src[0 * src_stride + 1 * ic];
        temp[2]  = src[0 * src_stride + 2 * ic];
        temp[3]  = src[0 * src_stride + 3 * ic];
        temp[4]  = src[1 * src_stride + 0 * ic];
        temp[5]  = src[1 * src_stride + 1 * ic];
        temp[6]  = src[1 * src_stride + 2 * ic];
        temp[7]  = src[1 * src_stride + 3 * ic];
        temp[8]  = src[2 * src_stride + 0 * ic];
        temp[9]  = src[2 * src_stride + 1 * ic];
        temp[10] = src[2 * src_stride + 2 * ic];
        temp[11] = src[2 * src_stride + 3 * ic];
        temp[12] = src[3 * src_stride + 0 * ic];
        temp[13] = src[3 * src_stride + 1 * ic];
        temp[14] = src[3 * src_stride + 2 * ic];
        temp[15] = src[3 * src_stride + 3 * ic];

        Conv3x3s1Winograd23TransformInput1tStore(temp, dst, dst_stride);

        src += 1;
        dst += 1;
    }
}

template<>
void Conv3x3s1Winograd23TransformInputFt<1>(const float* src,
                                            size_t iw,
                                            size_t ic,
                                            size_t row_start,
                                            size_t row_end,
                                            size_t col_start,
                                            size_t col_end,
                                            float* dst,
                                            size_t dst_stride) {
    size_t src_stride = iw * ic;

    for (size_t c = 0; c < ic; ++c) {
        float temp[16] = {0};

        for (size_t row = row_start; row < row_end; ++row) {
            for (size_t col = col_start; col < col_end; ++col) {
                temp[row * 4 + col] = src[row * src_stride + col * ic];
            }
        }

        Conv3x3s1Winograd23TransformInput1tStore(temp, dst, dst_stride);

        src += 1;
        dst += 1;
    }
}

template<size_t F>
void Conv3x3s1Winograd23TransformInput(const float* src,
                                       size_t ih,
                                       size_t iw,
                                       size_t ic,
                                       bool pad,
                                       float* dst,
                                       size_t dst_stride) {
    assert(1 == F || 4 == F);

    if (ic < F) {
        return Conv3x3s1Winograd23TransformInput<1>(src,
                                                    ih,
                                                    iw,
                                                    ic,
                                                    pad,
                                                    dst,
                                                    dst_stride);
    }

    size_t oh  = (pad ? ih : ih - 2);
    size_t ow  = (pad ? iw : iw - 2);
    size_t oh2 = oh / 2 * 2;
    size_t ow2 = ow / 2 * 2;

    size_t tile_h = (oh + 1) / 2;
    size_t tile_w = (ow + 1) / 2;

    size_t nose_h = (std::min)((size_t)4, oh + 1);
    size_t nose_w = (std::min)((size_t)4, ow + 1);

    size_t start = (pad ? 2 : 0);
    if (pad) {
        if (oh == oh2) {
            oh2 -= 2;
        }

        if (ow == ow2) {
            ow2 -= 2;
        }

        src -= (iw + 1) * ic;
    }

    size_t tail_h = oh - oh2 + (pad ? 1 : 2);
    size_t tail_w = ow - ow2 + (pad ? 1 : 2);

    size_t row = 0;
    size_t col = 0;

    if (pad) {
        if (pad) {
            Conv3x3s1Winograd23TransformInputFt<F>(src,
                                                   iw,
                                                   ic,
                                                   1,
                                                   nose_h,
                                                   1,
                                                   nose_w,
                                                   dst,
                                                   dst_stride);
            dst += ic;
        }

        for (col = start; col < ow2; col += 2) {
            Conv3x3s1Winograd23TransformInputFt<F>(src + col * ic,
                                                   iw,
                                                   ic,
                                                   1,
                                                   nose_h,
                                                   0,
                                                   4,
                                                   dst,
                                                   dst_stride);
            dst += ic;
        }

        if (col < ow) {
            Conv3x3s1Winograd23TransformInputFt<F>(src + col * ic,
                                                   iw,
                                                   ic,
                                                   1,
                                                   nose_h,
                                                   0,
                                                   tail_w,
                                                   dst,
                                                   dst_stride),
                dst += ic;
        }
    }

    for (row = start; row < oh2; row += 2) {
        if (pad) {
            Conv3x3s1Winograd23TransformInputFt<F>(src + row * iw * ic,
                                                   iw,
                                                   ic,
                                                   0,
                                                   4,
                                                   1,
                                                   nose_w,
                                                   dst,
                                                   dst_stride);
            dst += ic;
        }

        for (col = start; col < ow2; col += 2) {
            Conv3x3s1Winograd23TransformInputFt<F>(src + (row * iw + col) * ic,
                                                   iw,
                                                   ic,
                                                   dst,
                                                   dst_stride);
            dst += ic;
        }

        if (col < ow) {
            Conv3x3s1Winograd23TransformInputFt<F>(src + (row * iw + col) * ic,
                                                   iw,
                                                   ic,
                                                   0,
                                                   4,
                                                   0,
                                                   tail_w,
                                                   dst,
                                                   dst_stride);
            dst += ic;
        }
    }

    if (row < ow) {
        if (pad) {
            Conv3x3s1Winograd23TransformInputFt<F>(src + row * iw * ic,
                                                   iw,
                                                   ic,
                                                   0,
                                                   tail_h,
                                                   1,
                                                   nose_w,
                                                   dst,
                                                   dst_stride);
            dst += ic;
        }

        for (col = start; col < ow2; col += 2) {
            Conv3x3s1Winograd23TransformInputFt<F>(src + (row * iw + col) * ic,
                                                   iw,
                                                   ic,
                                                   0,
                                                   tail_h,
                                                   0,
                                                   4,
                                                   dst,
                                                   dst_stride);
            dst += ic;
        }

        if (col < ow) {
            Conv3x3s1Winograd23TransformInputFt<F>(src + (row * iw + col) * ic,
                                                   iw,
                                                   ic,
                                                   0,
                                                   tail_h,
                                                   0,
                                                   tail_w,
                                                   dst,
                                                   dst_stride);
            dst += ic;
        }
    }
}

void Conv3x3s1Winograd23TransformOutputLoad4(const float* src,
                                             size_t src_stride,
                                             f32x4_t dst[2]) {
    f32x4_t s0 = LoadU(d, src + 0 * src_stride);
    f32x4_t s1 = LoadU(d, src + 1 * src_stride);
    f32x4_t s2 = LoadU(d, src + 2 * src_stride);
    f32x4_t s3 = LoadU(d, src + 3 * src_stride);

    dst[0] = Add(Add(s0, s1), s2);
    dst[1] = Sub(Sub(s1, s2), s3);
}

void Conv3x3s1Winograd23TransformOutputLoad16(const float* src,
                                              size_t src_stride,
                                              f32x4_t dst[4]) {
    f32x4_t temp[8];
    Conv3x3s1Winograd23TransformOutputLoad4(src + 0 * src_stride,
                                            src_stride,
                                            temp + 0);
    Conv3x3s1Winograd23TransformOutputLoad4(src + 4 * src_stride,
                                            src_stride,
                                            temp + 2);
    Conv3x3s1Winograd23TransformOutputLoad4(src + 8 * src_stride,
                                            src_stride,
                                            temp + 4);
    Conv3x3s1Winograd23TransformOutputLoad4(src + 12 * src_stride,
                                            src_stride,
                                            temp + 6);

    dst[0] = Add(Add(temp[0], temp[2]), temp[4]);
    dst[1] = Add(Add(temp[1], temp[3]), temp[5]);
    dst[2] = Sub(Sub(temp[2], temp[4]), temp[6]);
    dst[3] = Sub(Sub(temp[3], temp[5]), temp[7]);
}

void Conv3x3s1Winograd23TransformOutputLoad1(const float* src,
                                             size_t src_stride,
                                             float dst[4]) {
    float s[16];
    s[0]  = src[0 * src_stride];
    s[1]  = src[1 * src_stride];
    s[2]  = src[2 * src_stride];
    s[3]  = src[3 * src_stride];
    s[4]  = src[4 * src_stride];
    s[5]  = src[5 * src_stride];
    s[6]  = src[6 * src_stride];
    s[7]  = src[7 * src_stride];
    s[8]  = src[8 * src_stride];
    s[9]  = src[9 * src_stride];
    s[10] = src[10 * src_stride];
    s[11] = src[11 * src_stride];
    s[12] = src[12 * src_stride];
    s[13] = src[13 * src_stride];
    s[14] = src[14 * src_stride];
    s[15] = src[15 * src_stride];

    float temp[8];
    temp[0] = s[0] + s[1] + s[2];
    temp[1] = s[1] - s[2] - s[3];
    temp[2] = s[4] + s[5] + s[6];
    temp[3] = s[5] - s[6] - s[7];
    temp[4] = s[8] + s[9] + s[10];
    temp[5] = s[9] - s[10] - s[11];
    temp[6] = s[12] + s[13] + s[14];
    temp[7] = s[13] - s[14] - s[15];

    dst[0] = temp[0] + temp[2] + temp[4];
    dst[1] = temp[1] + temp[3] + temp[5];
    dst[2] = temp[2] - temp[4] - temp[6];
    dst[3] = temp[3] - temp[5] - temp[7];
}

void Conv3x3s1Winograd23TransformOutputstore4(const f32x4_t src[4],
                                              float* dst,
                                              size_t dst_stride,
                                              size_t oc) {
    StoreU(src[0], d, dst + 0 * dst_stride + 0 * oc);
    StoreU(src[1], d, dst + 0 * dst_stride + 1 * oc);
    StoreU(src[2], d, dst + 1 * dst_stride + 0 * oc);
    StoreU(src[3], d, dst + 1 * dst_stride + 1 * oc);
}

void Conv3x3s1Winograd23TransformOutputstore4(const f32x4_t src[4],
                                              float* dst,
                                              size_t dst_stride,
                                              size_t oc,
                                              size_t row_end,
                                              size_t col_end) {
    for (size_t row = 0; row < row_end; ++row) {
        for (size_t col = 0; col < col_end; ++col) {
            StoreU(src[row * 2 + col], d, dst + row * dst_stride + col * oc);
        }
    }
}

template<size_t F>
void Conv3x3s1Winograd23TransformOutputFt(const float* src,
                                          size_t src_stride,
                                          float* dst,
                                          size_t ow,
                                          size_t oc);

template<size_t F>
void Conv3x3s1Winograd23TransformOutputFt(const float* src,
                                          size_t src_stride,
                                          float* dst,
                                          size_t ow,
                                          size_t oc,
                                          size_t row_end,
                                          size_t col_end);

template<>
void Conv3x3s1Winograd23TransformOutputFt<4>(const float* src,
                                             size_t src_stride,
                                             float* dst,
                                             size_t ow,
                                             size_t oc) {
    size_t dst_stride = ow * oc;
    size_t oc4        = oc / 4 * 4;

    f32x4_t temp[4];

    for (size_t c = 0; c < oc4; c += 4) {
        Conv3x3s1Winograd23TransformOutputLoad16(src + c, src_stride, temp);
        Conv3x3s1Winograd23TransformOutputstore4(temp, dst + c, dst_stride, oc);
    }

    if (oc4 < oc) {
        Conv3x3s1Winograd23TransformOutputLoad16(src + oc - 4,
                                                 src_stride,
                                                 temp);
        Conv3x3s1Winograd23TransformOutputstore4(temp,
                                                 dst + oc - 4,
                                                 dst_stride,
                                                 oc);
    }
}

template<>
void Conv3x3s1Winograd23TransformOutputFt<4>(const float* src,
                                             size_t src_stride,
                                             float* dst,
                                             size_t ow,
                                             size_t oc,
                                             size_t row_end,
                                             size_t col_end) {
    size_t dst_stride = ow * oc;
    size_t oc4        = oc / 4 * 4;

    f32x4_t temp[4];

    for (size_t c = 0; c < oc4; c += 4) {
        Conv3x3s1Winograd23TransformOutputLoad16(src + c, src_stride, temp);
        Conv3x3s1Winograd23TransformOutputstore4(temp,
                                                 dst + c,
                                                 dst_stride,
                                                 oc,
                                                 row_end,
                                                 col_end);
    }

    if (oc4 < oc) {
        Conv3x3s1Winograd23TransformOutputLoad16(src + oc - 4,
                                                 src_stride,
                                                 temp);
        Conv3x3s1Winograd23TransformOutputstore4(temp,
                                                 dst + oc - 4,
                                                 dst_stride,
                                                 oc,
                                                 row_end,
                                                 col_end);
    }
}

template<>
void Conv3x3s1Winograd23TransformOutputFt<1>(const float* src,
                                             size_t src_stride,
                                             float* dst,
                                             size_t ow,
                                             size_t oc) {
    size_t dst_stride = ow * oc;

    for (size_t c = 0; c < oc; ++c) {
        float temp[4];

        Conv3x3s1Winograd23TransformOutputLoad1(src, src_stride, temp);

        dst[0 * dst_stride + 0 * oc] = temp[0];
        dst[0 * dst_stride + 1 * oc] = temp[1];
        dst[1 * dst_stride + 0 * oc] = temp[2];
        dst[1 * dst_stride + 1 * oc] = temp[3];

        src += 1;
        dst += 1;
    }
}

template<>
void Conv3x3s1Winograd23TransformOutputFt<1>(const float* src,
                                             size_t src_stride,
                                             float* dst,
                                             size_t ow,
                                             size_t oc,
                                             size_t row_end,
                                             size_t col_end) {
    size_t dst_stride = ow * oc;

    for (size_t c = 0; c < oc; ++c) {
        float temp[4];

        Conv3x3s1Winograd23TransformOutputLoad1(src, src_stride, temp);

        for (size_t row = 0; row < row_end; ++row) {
            for (size_t col = 0; col < col_end; ++col) {
                dst[row * dst_stride + col * oc] = temp[row * 2 + col];
            }
        }

        src += 1;
        dst += 1;
    }
}

template<size_t F>
void Conv3x3s1Winograd23TransformOutput(const float* src,
                                        size_t src_stride,
                                        float* dst,
                                        size_t oh,
                                        size_t ow,
                                        size_t oc) {
    assert(1 == F || 4 == F);

    if (oc < F) {
        return Conv3x3s1Winograd23TransformOutput<1>(src,
                                                     src_stride,
                                                     dst,
                                                     oh,
                                                     ow,
                                                     oc);
    }

    size_t oh2 = oh / 2 * 2;
    size_t ow2 = ow / 2 * 2;

    size_t row = 0;
    size_t col = 0;

    for (row = 0; row < oh2; row += 2) {
        for (col = 0; col < ow2; col += 2) {
            Conv3x3s1Winograd23TransformOutputFt<F>(src,
                                                    src_stride,
                                                    dst + (row * ow + col) * oc,
                                                    ow,
                                                    oc);
            src += oc;
        }

        if (col < ow) {
            Conv3x3s1Winograd23TransformOutputFt<F>(src,
                                                    src_stride,
                                                    dst + (row * ow + col) * oc,
                                                    ow,
                                                    oc,
                                                    2,
                                                    ow - col);
            src += oc;
        }
    }

    if (row < oh) {
        for (col = 0; col < ow2; col += 2) {
            Conv3x3s1Winograd23TransformOutputFt<F>(src,
                                                    src_stride,
                                                    dst + (row * ow + col) * oc,
                                                    ow,
                                                    oc,
                                                    oh - row,
                                                    2);
            src += oc;
        }

        if (col < ow) {
            Conv3x3s1Winograd23TransformOutputFt<F>(src,
                                                    src_stride,
                                                    dst + (row * ow + col) * oc,
                                                    ow,
                                                    oc,
                                                    oh - row,
                                                    ow - col);
            src += oc;
        }
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy

namespace SimpleInfer {

namespace hn = hwy::HWY_NAMESPACE;

void Conv3x3s1Winograd23TransformKernelPack4(const float* src,
                                             size_t ic,
                                             size_t oc,
                                             float* dst) {
    return hn::Conv3x3s1Winograd23TransformKernelPack4(src, ic, oc, dst);
}

void Conv3x3s1Winograd23TransformInput(const float* src,
                                       size_t ih,
                                       size_t iw,
                                       size_t ic,
                                       bool pad,
                                       float* dst,
                                       size_t dst_stride) {
    return hn::Conv3x3s1Winograd23TransformInput<4>(src,
                                                    ih,
                                                    iw,
                                                    ic,
                                                    pad,
                                                    dst,
                                                    dst_stride);
}

void Conv3x3s1Winograd23TransformOutput(const float* src,
                                        size_t src_stride,
                                        float* dst,
                                        size_t oh,
                                        size_t ow,
                                        size_t oc) {
    return hn::Conv3x3s1Winograd23TransformOutput<4>(src,
                                                     src_stride,
                                                     dst,
                                                     oh,
                                                     ow,
                                                     oc);
}

}  // namespace SimpleInfer
