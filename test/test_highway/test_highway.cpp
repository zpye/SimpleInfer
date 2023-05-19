#include <string>

#include "logger.h"

// #undef HWY_TARGET_INCLUDE
// #define HWY_TARGET_INCLUDE "test_highway/test_highway.cpp"
// #include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

// HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

void MulAddLoop(const float* HWY_RESTRICT mul_array,
                const float* HWY_RESTRICT add_array,
                const size_t size,
                float* HWY_RESTRICT x_array) {
    // const ScalableTag<float> d;
    const Full128<float> d;
    for (size_t i = 0; i < size; i += Lanes(d)) {
        const auto mul = Load(d, mul_array + i);
        const auto add = Load(d, add_array + i);
        auto x         = Load(d, x_array + i);
        x              = MulAdd(mul, x, add);
        Store(x, d, x_array + i);
    }
}

void Gemm4x12Pack4F32(size_t K,
                      const float* A,
                      size_t lda,
                      const float* B,
                      size_t ldb,
                      float* C,
                      size_t ldc) {
    const Full128<float> d;
    assert(4 == Lanes(d));
    using f32x4_t = VFromD<Full128<float>>;

    f32x4_t c00 = Zero(d);
    f32x4_t c01 = Zero(d);
    f32x4_t c02 = Zero(d);
    f32x4_t c10 = Zero(d);
    f32x4_t c11 = Zero(d);
    f32x4_t c12 = Zero(d);
    f32x4_t c20 = Zero(d);
    f32x4_t c21 = Zero(d);
    f32x4_t c22 = Zero(d);
    f32x4_t c30 = Zero(d);
    f32x4_t c31 = Zero(d);
    f32x4_t c32 = Zero(d);

    const size_t oa0 = lda * 0;
    const size_t oa1 = lda * 1;
    const size_t oa2 = lda * 2;
    const size_t oa3 = lda * 3;

    const size_t ob0 = ldb * 0;
    const size_t ob1 = ldb * 1;
    const size_t ob2 = ldb * 2;

    // temp
    f32x4_t a0;
    f32x4_t b0;
    f32x4_t b1;
    f32x4_t b2;

    for (size_t k = 0; k < K; ++k) {
        b0 = Load(d, B + ob0);
        b1 = Load(d, B + ob1);
        b2 = Load(d, B + ob2);

        a0  = Set(d, A[oa0]);
        c00 = MulAdd(a0, b0, c00);
        c01 = MulAdd(a0, b1, c01);
        c02 = MulAdd(a0, b2, c02);

        a0  = Set(d, A[oa1]);
        c10 = MulAdd(a0, b0, c10);
        c11 = MulAdd(a0, b1, c11);
        c12 = MulAdd(a0, b2, c12);

        a0  = Set(d, A[oa2]);
        c20 = MulAdd(a0, b0, c20);
        c21 = MulAdd(a0, b1, c21);
        c22 = MulAdd(a0, b2, c22);

        a0  = Set(d, A[oa3]);
        c30 = MulAdd(a0, b0, c30);
        c31 = MulAdd(a0, b1, c31);
        c32 = MulAdd(a0, b2, c32);

        B += 4;
        A += 1;
    }

    Store(Add(Load(d, C + 0), c00), d, C + 0);
    Store(Add(Load(d, C + 4), c01), d, C + 4);
    Store(Add(Load(d, C + 8), c02), d, C + 8);

    C += ldc;

    Store(Add(Load(d, C + 0), c10), d, C + 0);
    Store(Add(Load(d, C + 4), c11), d, C + 4);
    Store(Add(Load(d, C + 8), c12), d, C + 8);

    C += ldc;

    Store(Add(Load(d, C + 0), c20), d, C + 0);
    Store(Add(Load(d, C + 4), c21), d, C + 4);
    Store(Add(Load(d, C + 8), c22), d, C + 8);

    C += ldc;

    Store(Add(Load(d, C + 0), c30), d, C + 0);
    Store(Add(Load(d, C + 4), c31), d, C + 4);
    Store(Add(Load(d, C + 8), c32), d, C + 8);
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy
// HWY_AFTER_NAMESPACE();

#ifndef TEST_ONCE
#define TEST_ONCE

namespace hn = hwy::HWY_NAMESPACE;

int main() {
    SimpleInfer::InitializeLogger();

    float a[16];
    float b[16];
    float c[16];

    for (int i = 0; i < 16; ++i) {
        a[i] = (float)i;
        b[i] = (float)i * 0.1f;
        c[i] = 1.0f;
    }

    hn::MulAddLoop(a, b, 16, c);

    for (int i = 0; i < 16; ++i) {
        LOG(INFO)
            << absl::StrFormat("%d, a(%f), b(%f), c(%f)", i, a[i], b[i], c[i]);
    }

    alignas(16) float A[8 * 36];
    alignas(16) float B[3 * 36 * 4];
    alignas(16) float C[8 * 12];

    for (int i = 0; i < 8 * 36; ++i) {
        A[i] = 1.0f + (float)i * 0.01f;
    }

    for (int i = 0; i < 3 * 36 * 4; ++i) {
        B[i] = 1.0f + (float)i * 0.02f;
    }

    for (int i = 0; i < 8 * 12; ++i) {
        C[i] = 0.0f;
    }

    hn::Gemm4x12Pack4F32(36, A, 36, B, 36 * 4, C, 12);
    hn::Gemm4x12Pack4F32(36, A + 4 * 36, 36, B, 36 * 4, C + 4 * 12, 12);

    // check
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 12; ++j) {
            float temp = 0.0f;
            for (int k = 0; k < 36; ++k) {
                temp += A[i * 36 + k] * B[(j / 4) * 36 * 4 + k * 4 + (j % 4)];
            }

            LOG(INFO) << absl::StrFormat("(%d, %d), (%f, %f)",
                                         i,
                                         j,
                                         C[i * 12 + j],
                                         temp);
        }
    }

    return 0;
}

#endif  // TEST_ONCE
