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

#include "gemm.h"

#include <cassert>

#include "hwy/highway.h"

namespace hwy {
namespace HWY_NAMESPACE {

static const Full128<float> d;
static_assert(4 == Lanes(d), "Lanes(Full128<float>) should be 4");
using f32x4_t = VFromD<Full128<float>>;

inline void AddToMemory(float* ptr, f32x4_t value) {
    StoreU(Add(LoadU(d, ptr), value), d, ptr);
}

inline void AddToMemory(float* ptr, f32x4_t value, size_t tail) {
    if (4 == tail) {
        AddToMemory(ptr, value);
    } else {
        float temp[4];
        StoreU(Add(LoadU(d, ptr), value), d, temp);

        for (size_t i = 0; i < tail; ++i) {
            ptr[i] = temp[i];
        }
    }
}

inline void SetToMemory(float* ptr, f32x4_t value) {
    StoreU(value, d, ptr);
}

inline void SetToMemory(float* ptr, f32x4_t value, size_t tail) {
    if (4 == tail) {
        SetToMemory(ptr, value);
    } else {
        float temp[4];
        StoreU(value, d, temp);

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
        b0 = LoadU(d, B + ob0);
        b1 = LoadU(d, B + ob1);
        b2 = LoadU(d, B + ob2);

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

    SetToMemory(C + 0, c00);
    SetToMemory(C + 4, c01);
    SetToMemory(C + 8, c02);

    C += ldc;

    SetToMemory(C + 0, c10);
    SetToMemory(C + 4, c11);
    SetToMemory(C + 8, c12);

    C += ldc;

    SetToMemory(C + 0, c20);
    SetToMemory(C + 4, c21);
    SetToMemory(C + 8, c22);

    C += ldc;

    SetToMemory(C + 0, c30);
    SetToMemory(C + 4, c31);
    SetToMemory(C + 8, c32);
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
        b0 = LoadU(d, B + ob0);
        b1 = LoadU(d, B + ob1);
        b2 = LoadU(d, B + ob2);

        for (size_t i = 0; i < M; ++i) {
            a0 = Set(d, A[oa[i]]);

            c[i][0] = MulAdd(a0, b0, c[i][0]);
            c[i][1] = MulAdd(a0, b1, c[i][1]);
            c[i][2] = MulAdd(a0, b2, c[i][2]);
        }

        B += 4;
        A += 1;
    }

    for (size_t i = 0; i < M; ++i) {
        SetToMemory(C + 0, c[i][0]);
        SetToMemory(C + 4, c[i][1]);
        SetToMemory(C + 8, c[i][2]);

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
        b0 = LoadU(d, B);

        c0 = MulAdd(Set(d, A[oa0]), b0, c0);
        c1 = MulAdd(Set(d, A[oa1]), b0, c1);
        c2 = MulAdd(Set(d, A[oa2]), b0, c2);
        c3 = MulAdd(Set(d, A[oa3]), b0, c3);

        B += 4;
        A += 1;
    }

    SetToMemory(C + 0 * ldc, c0, tail);
    SetToMemory(C + 1 * ldc, c1, tail);
    SetToMemory(C + 2 * ldc, c2, tail);
    SetToMemory(C + 3 * ldc, c3, tail);
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

        oa[i] = i * lda;
    }

    // temp
    f32x4_t b0;

    for (size_t k = 0; k < K; ++k) {
        b0 = LoadU(d, B);

        for (size_t i = 0; i < M; ++i) {
            c[i] = MulAdd(Set(d, A[oa[i]]), b0, c[i]);
        }

        B += 4;
        A += 1;
    }

    for (size_t i = 0; i < M; ++i) {
        SetToMemory(C + i * ldc, c[i], tail);
    }
}

void GemmPack4F32(size_t M,
                  size_t N,
                  size_t K,
                  const float* A,
                  size_t lda,
                  const float* B,
                  float* C,
                  size_t ldc) {
    size_t ldb = K * 4;

    size_t M4  = M / 4 * 4;
    size_t N12 = N / 12 * 12;
    size_t N4  = N / 4 * 4;

    size_t tail_N = N - N4;

    size_t j = 0;
    for (; j < N12; j += 12) {
        size_t i = 0;
        for (; i < M4; i += 4) {
            Gemm4x12Pack4F32(K, A + i * lda, lda, B, ldb, C + i * ldc + j, ldc);
        }

        if (i < M) {
            GemmMx12Pack4F32(M - i,
                             K,
                             A + i * lda,
                             lda,
                             B,
                             ldb,
                             C + i * ldc + j,
                             ldc);
        }

        B += 3 * ldb;
    }

    for (; j < N4; j += 4) {
        size_t i = 0;
        for (; i < M4; i += 4) {
            Gemm4x4Pack4F32(K,
                            A + i * lda,
                            lda,
                            B,
                            ldb,
                            C + i * ldc + j,
                            ldc,
                            4);
        }

        if (i < M) {
            GemmMx4Pack4F32(M - i,
                            K,
                            A + i * lda,
                            lda,
                            B,
                            ldb,
                            C + i * ldc + j,
                            ldc,
                            4);
        }

        B += ldb;
    }

    if (j < N) {
        size_t i = 0;
        for (; i < M4; i += 4) {
            Gemm4x4Pack4F32(K,
                            A + i * lda,
                            lda,
                            B,
                            ldb,
                            C + i * ldc + j,
                            ldc,
                            tail_N);
        }

        if (i < M) {
            GemmMx4Pack4F32(M - i,
                            K,
                            A + i * lda,
                            lda,
                            B,
                            ldb,
                            C + i * ldc + j,
                            ldc,
                            tail_N);
        }
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy

namespace SimpleInfer {

namespace hn = hwy::HWY_NAMESPACE;

void GemmPack4F32(size_t M,
                  size_t N,
                  size_t K,
                  const float* A,
                  size_t lda,
                  const float* B,
                  float* C,
                  size_t ldc) {
    return hn::GemmPack4F32(M, N, K, A, lda, B, C, ldc);
}

void GemmPack4F32Ref(size_t M,
                     size_t N,
                     size_t K,
                     const float* A,
                     size_t lda,
                     const float* B,
                     float* C,
                     size_t ldc) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            size_t n4   = n / 4;
            size_t nres = n % 4;

            C[m * ldc + n] = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                C[m * ldc + n] += A[m * lda + k] * B[n4 * K * 4 + k * 4 + nres];
            }
        }
    }
}

}  // namespace SimpleInfer
