#include "gemm.h"

#include <cassert>

#include "hwy/highway.h"

namespace hwy {
namespace HWY_NAMESPACE {

static const Full128<float> d;
static_assert(4 == Lanes(d), "Lanes(Full128<float>) should be 4");
using f32x4_t = VFromD<Full128<float>>;

inline void AddToMemory(float* ptr, f32x4_t value) {
    Store(Add(Load(d, ptr), value), d, ptr);
}

inline void AddToMemory(float* ptr, f32x4_t value, size_t tail) {
    if (4 == tail) {
        AddToMemory(ptr, value);
    } else {
        float temp[4];
        Store(Add(Load(d, ptr), value), d, temp);

        for (size_t i = 0; i < tail; ++i) {
            ptr[i] = temp[i];
        }
    }
}

void Gemm4x12Pack4F32(size_t K,
                      const float* A,
                      size_t lda,
                      const float* B,
                      size_t ldb,
                      float* C,
                      size_t ldc) {
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

    const size_t oa0 = 0 * lda;
    const size_t oa1 = 1 * lda;
    const size_t oa2 = 2 * lda;
    const size_t oa3 = 3 * lda;

    const size_t ob0 = 0 * ldb;
    const size_t ob1 = 1 * ldb;
    const size_t ob2 = 2 * ldb;

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

    AddToMemory(C + 0, c00);
    AddToMemory(C + 4, c01);
    AddToMemory(C + 8, c02);

    C += ldc;

    AddToMemory(C + 0, c10);
    AddToMemory(C + 4, c11);
    AddToMemory(C + 8, c12);

    C += ldc;

    AddToMemory(C + 0, c20);
    AddToMemory(C + 4, c21);
    AddToMemory(C + 8, c22);

    C += ldc;

    AddToMemory(C + 0, c30);
    AddToMemory(C + 4, c31);
    AddToMemory(C + 8, c32);
}

void GemmMx12Pack4F32(size_t M,
                      size_t K,
                      const float* A,
                      size_t lda,
                      const float* B,
                      size_t ldb,
                      float* C,
                      size_t ldc) {
    assert(M <= 4);

    f32x4_t c[4][3];
    size_t oa[4];

    for (size_t i = 0; i < M; ++i) {
        c[i][0] = Zero(d);
        c[i][1] = Zero(d);
        c[i][2] = Zero(d);

        oa[i] = i * lda;
    }

    const size_t ob0 = 0 * ldb;
    const size_t ob1 = 1 * ldb;
    const size_t ob2 = 2 * ldb;

    // temp
    f32x4_t a0;
    f32x4_t b0;
    f32x4_t b1;
    f32x4_t b2;

    for (size_t k = 0; k < K; ++k) {
        b0 = Load(d, B + ob0);
        b1 = Load(d, B + ob1);
        b2 = Load(d, B + ob2);

        for (size_t i = 0; i < M; ++i) {
            a0 = Set(d, A[oa[i]]);

            c[i][0] = MulAdd(a0, b0, c[i][0]);
            c[i][1] = MulAdd(a0, b0, c[i][1]);
            c[i][2] = MulAdd(a0, b0, c[i][2]);
        }

        B += 4;
        A += 1;
    }

    for (size_t i = 0; i < M; ++i) {
        AddToMemory(C + 0, c[i][0]);
        AddToMemory(C + 4, c[i][1]);
        AddToMemory(C + 8, c[i][2]);

        C += ldc;
    }
}

void Gemm4x4Pack4F32(size_t K,
                     const float* A,
                     size_t lda,
                     const float* B,
                     size_t ldb,
                     float* C,
                     size_t ldc,
                     size_t tail) {
    f32x4_t c0 = Zero(d);
    f32x4_t c1 = Zero(d);
    f32x4_t c2 = Zero(d);
    f32x4_t c3 = Zero(d);

    const size_t oa0 = 0 * lda;
    const size_t oa1 = 1 * lda;
    const size_t oa2 = 2 * lda;
    const size_t oa3 = 3 * lda;

    // temp
    f32x4_t a0;
    f32x4_t b0;

    for (size_t k = 0; k < K; ++k) {
        b0 = Load(d, B);

        c0 = MulAdd(Set(d, A[oa0]), b0, c0);
        c1 = MulAdd(Set(d, A[oa1]), b0, c1);
        c2 = MulAdd(Set(d, A[oa2]), b0, c2);

        B += 4;
        A += 1;
    }

    AddToMemory(C + 0 * ldc, c0, tail);
    AddToMemory(C + 1 * ldc, c1, tail);
    AddToMemory(C + 2 * ldc, c2, tail);
    AddToMemory(C + 3 * ldc, c3, tail);
}

void GemmMx4Pack4F32(size_t M,
                     size_t K,
                     const float* A,
                     size_t lda,
                     const float* B,
                     size_t ldb,
                     float* C,
                     size_t ldc,
                     size_t tail) {
    assert(M <= 4);

    f32x4_t c[4];
    size_t oa[4];

    for (size_t i = 0; i < M; ++i) {
        c[i] = Zero(d);
        c[i] = Zero(d);
        c[i] = Zero(d);

        oa[i] = i * lda;
    }

    // temp
    f32x4_t b0;

    for (size_t k = 0; k < K; ++k) {
        b0 = Load(d, B);

        for (size_t i = 0; i < M; ++i) {
            c[i] = MulAdd(Set(d, A[oa[i]]), b0, c[i]);
        }

        B += 4;
        A += 1;
    }

    for (size_t i = 0; i < M; ++i) {
        AddToMemory(C + i * ldc, c[i]);
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy
