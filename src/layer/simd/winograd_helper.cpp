#include "winograd_helper.h"

#include <cassert>
#include <vector>

#include "hwy/highway.h"

namespace hwy {
namespace HWY_NAMESPACE {

static const Full128<float> d;
static_assert(4 == Lanes(d), "Lanes(Full128<float>) should be 4");
using f32x4_t = VFromD<Full128<float>>;

// [oc][ic][3(kh)][3(kw)] -> [4(gh)][4(gw)][oc/4][ic][4(oc)]
void Conv3x3s1Winograd23TransformKernelPack4(const float* src,
                                             size_t ic,
                                             size_t oc,
                                             float* dst) {
    // padding oc with 4
    size_t oc_up4 = (oc + 3) / 4 * 4;

    // 1. [oc][ic][3(kh)][3(kw)] -> [3(kh)][3(kw)][ic][oc4]
    std::vector<float> hwio(9 * ic * oc_up4);
    for (size_t k = 0; k < 9; ++k) {
        for (size_t i = 0; i < ic; ++i) {
            size_t j = 0;
            for (; j < oc; ++j) {
                hwio[k * ic * oc_up4 + i * oc_up4 + j] =
                    src[j * ic * 9 + i * 9 + k];
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
            const float* in = ghwio.data() + k * stride + j * 4;
            for (size_t i = 0; i < ic; ++i) {
                Store(LoadU(d, in + i * oc_up4), d, dst);
                dst += 4;
            }
        }
    }
}

// load full 4x4 input
void Conv3x3s1Winograd23TransformInput4tLoad(const float* src,
                                             size_t src_stride,
                                             size_t ic,
                                             f32x4_t temp[16]) {
    temp[0]  = LoadU(d, src + 0 * src_stride + 0 * ic);
    temp[1]  = LoadU(d, src + 0 * src_stride + 1 * ic);
    temp[2]  = LoadU(d, src + 0 * src_stride + 2 * ic);
    temp[3]  = LoadU(d, src + 0 * src_stride + 3 * ic);
    temp[4]  = LoadU(d, src + 1 * src_stride + 0 * ic);
    temp[5]  = LoadU(d, src + 1 * src_stride + 1 * ic);
    temp[6]  = LoadU(d, src + 1 * src_stride + 2 * ic);
    temp[7]  = LoadU(d, src + 1 * src_stride + 3 * ic);
    temp[8]  = LoadU(d, src + 2 * src_stride + 0 * ic);
    temp[9]  = LoadU(d, src + 2 * src_stride + 1 * ic);
    temp[10] = LoadU(d, src + 2 * src_stride + 2 * ic);
    temp[11] = LoadU(d, src + 2 * src_stride + 3 * ic);
    temp[12] = LoadU(d, src + 3 * src_stride + 0 * ic);
    temp[13] = LoadU(d, src + 3 * src_stride + 1 * ic);
    temp[14] = LoadU(d, src + 3 * src_stride + 2 * ic);
    temp[15] = LoadU(d, src + 3 * src_stride + 3 * ic);
}

// load partial 4x4 input
void Conv3x3s1Winograd23TransformInput4tLoad(const float* src,
                                             size_t src_stride,
                                             size_t ic,
                                             size_t row_start,
                                             size_t row_end,
                                             size_t col_start,
                                             size_t col_end,
                                             f32x4_t temp[16]) {
    for (size_t i = 0; i < 16; ++i) {
        temp[i] = Zero(d);
    }

    for (size_t row = row_start; row < row_end; ++row) {
        for (size_t col = col_start; col < col_end; ++col) {
            temp[row * 4 + col] = LoadU(d, src + row * src_stride + col * ic);
        }
    }
}

void Conv3x3s1Winograd23TransformInput4tStore(const f32x4_t temp[16],
                                              float* dst,
                                              size_t dst_stride) {
    StoreU(Sub(Sub(temp[0], temp[8]), Sub(temp[2], temp[10])),
           d,
           dst + 0 * dst_stride);
    StoreU(Add(Sub(temp[1], temp[9]), Sub(temp[2], temp[10])),
           d,
           dst + 1 * dst_stride);
    StoreU(Sub(Sub(temp[2], temp[10]), Sub(temp[1], temp[9])),
           d,
           dst + 2 * dst_stride);
    StoreU(Sub(Sub(temp[1], temp[9]), Sub(temp[3], temp[11])),
           d,
           dst + 3 * dst_stride);
    StoreU(Sub(Add(temp[4], temp[8]), Add(temp[6], temp[10])),
           d,
           dst + 4 * dst_stride);
    StoreU(Add(Add(temp[5], temp[9]), Add(temp[6], temp[10])),
           d,
           dst + 5 * dst_stride);
    StoreU(Sub(Add(temp[6], temp[10]), Add(temp[5], temp[9])),
           d,
           dst + 6 * dst_stride);
    StoreU(Sub(Add(temp[5], temp[9]), Add(temp[7], temp[11])),
           d,
           dst + 7 * dst_stride);
    StoreU(Sub(Sub(temp[8], temp[4]), Sub(temp[10], temp[6])),
           d,
           dst + 8 * dst_stride);
    StoreU(Add(Sub(temp[9], temp[5]), Sub(temp[10], temp[6])),
           d,
           dst + 9 * dst_stride);
    StoreU(Sub(Sub(temp[10], temp[6]), Sub(temp[9], temp[5])),
           d,
           dst + 10 * dst_stride);
    StoreU(Sub(Sub(temp[9], temp[5]), Sub(temp[11], temp[7])),
           d,
           dst + 11 * dst_stride);
    StoreU(Sub(Sub(temp[4], temp[12]), Sub(temp[6], temp[14])),
           d,
           dst + 12 * dst_stride);
    StoreU(Add(Sub(temp[5], temp[13]), Sub(temp[6], temp[14])),
           d,
           dst + 13 * dst_stride);
    StoreU(Sub(Sub(temp[6], temp[14]), Sub(temp[5], temp[13])),
           d,
           dst + 14 * dst_stride);
    StoreU(Sub(Sub(temp[5], temp[13]), Sub(temp[7], temp[15])),
           d,
           dst + 15 * dst_stride);
}

void Conv3x3s1Winograd23TransformInput4t(const float* src,
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

void Conv3x3s1Winograd23TransformInput4(const float* src,
                                        size_t ih,
                                        size_t iw,
                                        size_t ic,
                                        bool pad,
                                        float* dst) {
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
        if (pad)
            WinogradKernel3x3Block2x2SetInput4t(src,
                                                srcWidth,
                                                srcChannels,
                                                1,
                                                noseH,
                                                1,
                                                noseW,
                                                dst,
                                                dstStride),
                dst += srcChannels;
        for (col = start; col < dstW2; col += 2)
            WinogradKernel3x3Block2x2SetInput4t(src + col * srcChannels,
                                                srcWidth,
                                                srcChannels,
                                                1,
                                                noseH,
                                                0,
                                                4,
                                                dst,
                                                dstStride),
                dst += srcChannels;
        if (col < dstW)
            WinogradKernel3x3Block2x2SetInput4t(src + col * srcChannels,
                                                srcWidth,
                                                srcChannels,
                                                1,
                                                noseH,
                                                0,
                                                tailW,
                                                dst,
                                                dstStride),
                dst += srcChannels;
    }

    for (row = start; row < dstH2; row += 2) {
        if (pad)
            WinogradKernel3x3Block2x2SetInput4t(
                src + row * srcWidth * srcChannels,
                srcWidth,
                srcChannels,
                0,
                4,
                1,
                noseW,
                dst,
                dstStride),
                dst += srcChannels;
        for (col = start; col < dstW2; col += 2)
            WinogradKernel3x3Block2x2SetInput4t(
                src + (row * srcWidth + col) * srcChannels,
                srcWidth,
                srcChannels,
                dst,
                dstStride),
                dst += srcChannels;
        if (col < dstW)
            WinogradKernel3x3Block2x2SetInput4t(
                src + (row * srcWidth + col) * srcChannels,
                srcWidth,
                srcChannels,
                0,
                4,
                0,
                tailW,
                dst,
                dstStride),
                dst += srcChannels;
    }

    if (row < dstH) {
        if (pad)
            WinogradKernel3x3Block2x2SetInput4t(
                src + row * srcWidth * srcChannels,
                srcWidth,
                srcChannels,
                0,
                tailH,
                1,
                noseW,
                dst,
                dstStride),
                dst += srcChannels;
        for (col = start; col < dstW2; col += 2)
            WinogradKernel3x3Block2x2SetInput4t(
                src + (row * srcWidth + col) * srcChannels,
                srcWidth,
                srcChannels,
                0,
                tailH,
                0,
                4,
                dst,
                dstStride),
                dst += srcChannels;
        if (col < dstW)
            WinogradKernel3x3Block2x2SetInput4t(
                src + (row * srcWidth + col) * srcChannels,
                srcWidth,
                srcChannels,
                0,
                tailH,
                0,
                tailW,
                dst,
                dstStride),
                dst += srcChannels;
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy
