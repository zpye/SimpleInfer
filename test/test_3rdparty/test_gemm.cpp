#include "common.h"

#include "layer/simd/gemm.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

inline void GenData(size_t M,
                    size_t N,
                    size_t K,
                    std::vector<float>& A,
                    size_t& lda,
                    std::vector<float>& B,
                    std::vector<float>& C,
                    size_t& ldc,
                    std::vector<float>& C_true) {
    A.resize(M * K, 0.0f);
    lda = K;

    size_t N_updiv4 = (N + 3) / 4;
    B.resize(N_updiv4 * K * 4, 0.0f);

    C.resize(M * N, -1.0f);
    C_true.resize(M * N, -1.0f);
    ldc = N;

    // set random value
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> nd;

    for (auto& v : A) {
        // v = nd(gen);
        v = 1.0f;
    }

    for (auto& v : B) {
        // v = nd(gen);
        v = 1.0f;
    }

    // compute Gemm
    SimpleInfer::GemmPack4F32(M, N, K, A.data(), lda, B.data(), C.data(), ldc);
    SimpleInfer::GemmPack4F32Ref(M,
                                 N,
                                 K,
                                 A.data(),
                                 lda,
                                 B.data(),
                                 C_true.data(),
                                 ldc);
}

inline void TestFunc(size_t M, size_t N, size_t K) {
    std::vector<float> A;
    size_t lda;
    std::vector<float> B;
    std::vector<float> C;
    size_t ldc;
    std::vector<float> C_true;

    GenData(M, N, K, A, lda, B, C, ldc, C_true);

    CHECK_EQ(C.size(), C_true.size());

    for (size_t i = 0; i < C.size(); ++i) {
        CHECK_EQ(C[i], C_true[i]);
    }
}

TEST_CASE("Test Gemm", "[Gemm]") {
    TestFunc(1, 1, 1);
    TestFunc(2, 2, 2);
    TestFunc(3, 3, 3);
    TestFunc(4, 4, 4);
    TestFunc(9, 9, 9);
    TestFunc(11, 11, 11);
    TestFunc(12, 12, 12);
    TestFunc(13, 13, 13);
    TestFunc(21, 21, 21);
    TestFunc(97, 97, 97);

    TestFunc(32, 128, 64);
    TestFunc(33, 129, 65);
    TestFunc(64, 128, 32);
    TestFunc(1024, 128, 256);
}
