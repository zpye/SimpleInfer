#include "eigen_helper.h"

#include "logger.h"

using namespace SimpleInfer;

int main() {
    InitializeLogger();

    // NHWC
    EigenTensor<float, 4> image(2, 2, 4, 3);
    LOG(INFO) << "image size: " << image.size();

    for (int i = 0; i < (int)image.size(); ++i) {
        *(image.data() + i) = (float)i;
    }
    LOG(INFO) << "image:\n" << image;

    EigenTensor<float, 5> image_patch =
        image.extract_image_patches(2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.0f);
    LOG(INFO) << "image patch:\n" << image_patch;

    int image_matrix_cols = 2 * 2 * 3;  // patch_height * patch_width * channels
    int image_matrix_rows =
        image_patch.size() /
        image_matrix_cols;  // batch * output_height * output_width

    EigenTensor<float, 2> image_matrix = image_patch.reshape(
        std::array<int, 2>{image_matrix_rows, image_matrix_cols});
    LOG(INFO) << "image matrix:\n" << image_matrix;

    // OIHW
    EigenTensor<float, 4> kernel(4, 3, 2, 2);
    for (int i = 0; i < (int)kernel.size(); ++i) {
        *(kernel.data() + i) = (float)i;
    }
    LOG(INFO) << "kernel:\n" << kernel;

    // HWIO
    EigenTensor<float, 4> kernel_shuffle =
        kernel.shuffle(std::array<int, 4>{2, 3, 1, 0});
    LOG(INFO) << "kernel_shuffle:\n" << kernel_shuffle;

    int kernel_matrix_rows =
        2 * 2 * 3;  // kernel_height * kernel_width * input_channels
    int kernel_matrix_cols =
        kernel_shuffle.size() / kernel_matrix_rows;  // output_channels

    EigenTensor<float, 2> kernel_matrix = kernel_shuffle.reshape(
        std::array<int, 2>{kernel_matrix_rows, kernel_matrix_cols});
    LOG(INFO) << "kernel matrix:\n" << kernel_matrix;

    return 0;
}
