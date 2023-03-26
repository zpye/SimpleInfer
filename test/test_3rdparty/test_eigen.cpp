#include "eigen_helper.h"

#include "logger.h"

int main() {
    SimpleInfer::InitializeLogger();

    Eigen::Tensor<float, 4> tensor0(1, 3, 2, 2);
    Eigen::Tensor<float, 4, Eigen::RowMajor> tensor1(1, 3, 2, 2);

    int num = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                *(tensor0.data() + num) = (float)num;
                *(tensor1.data() + num) = (float)num;
                num += 1;
            }
        }
    }

    LOG(INFO) << "<<<<<<<<< tensor0 <<<<<<<<";
    LOG(INFO) << "\n" << tensor0;

    num = 0;
    for (int k = 0; k < 2; ++k) {
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 3; ++i) {
                LOG(INFO) << absl::StrFormat("(%d, %d, %d) -> %f",
                                             i,
                                             j,
                                             k,
                                             tensor0(0, i, j, k));
                num += 1;
            }
        }
    }

    LOG(INFO) << "<<<<<<<<< tensor1 <<<<<<<<";
    LOG(INFO) << "\n" << tensor1;

    num = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                LOG(INFO) << absl::StrFormat("(%d, %d, %d) -> %f",
                                             i,
                                             j,
                                             k,
                                             tensor1(0, i, j, k));
                num += 1;
            }
        }
    }

    Eigen::Tensor<float, 4> tensor2 = tensor1.swap_layout();

    LOG(INFO) << "<<<<<<<<< tensor2 <<<<<<<<";
    LOG(INFO) << "\n" << tensor2;

    num = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                LOG(INFO) << absl::StrFormat("(%d, %d, %d) -> %f",
                                             k,
                                             j,
                                             i,
                                             tensor2(k, j, i, 0));
                num += 1;
            }
        }
    }

    Eigen::Tensor<float, 4> tensor3 = SimpleInfer::ConvertLayout(tensor1);

    LOG(INFO) << "<<<<<<<<< tensor3 <<<<<<<<";
    LOG(INFO) << "\n" << tensor3;

    num = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                LOG(INFO) << absl::StrFormat("(%d, %d, %d) -> %f",
                                             i,
                                             j,
                                             k,
                                             tensor3(0, i, j, k));
                num += 1;
            }
        }
    }

    SimpleInfer::EigenDSize<4> start(0, 0, 1, 1);
    SimpleInfer::EigenDSize<4> size(1, 3, 1, 1);
    Eigen::Tensor<float, 4> tensor4 = tensor3.slice(start, size);

    LOG(INFO) << "<<<<<<<<< tensor4 <<<<<<<<";
    LOG(INFO) << "\n" << tensor4;

    num = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 1; ++j) {
            for (int k = 0; k < 1; ++k) {
                LOG(INFO) << absl::StrFormat("(%d, %d, %d) -> %f",
                                             i,
                                             j,
                                             k,
                                             tensor4(0, i, j, k));
                num += 1;
            }
        }
    }

    return 0;
}
