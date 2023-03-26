#ifndef SIMPLE_INFER_INCLUDE_EIGEN_HELPER_H_
#define SIMPLE_INFER_INCLUDE_EIGEN_HELPER_H_

#include <array>
#include <cassert>
#include <vector>

#include <unsupported/Eigen/CXX11/Tensor>

namespace SimpleInfer {

// use row major as default
template<typename T, int num_indices, int Options = 0x1>
using EigenTensor = Eigen::Tensor<T, num_indices, Options>;

template<typename T, int num_indices, int Options = 0x1>
using EigenTensorMap = Eigen::TensorMap<EigenTensor<T, num_indices, Options>>;

template<int num_indices>
using EigenDSize = Eigen::DSizes<ptrdiff_t, num_indices>;

template<typename T, int num_indices, int Options>
auto ConvertLayout(const Eigen::Tensor<T, num_indices, Options>& tensor) {
    EigenDSize<num_indices> shuffle;
    for (int i = 0; i < num_indices; ++i) {
        shuffle[i] = num_indices - 1 - i;
    }

    return tensor.swap_layout().shuffle(shuffle);
}

template<int num_indices>
EigenDSize<num_indices> ToEigenDSize(const std::vector<int>& vec) {
    assert(num_indices >= 0);

    EigenDSize<num_indices> dsize;

    if (num_indices <= (int)vec.size()) {
        int index = (int)vec.size() - 1;
        for (int i = num_indices - 1; i > 0; --i) {
            dsize[i] = vec[index];
            index -= 1;
        }

        int d0 = 1;
        for (int i = index; i >= 0; --i) {
            d0 *= vec[i];
        }
        dsize[0] = d0;
    } else {
        int index = num_indices - 1;
        for (int i = (int)vec.size() - 1; i >= 0; --i) {
            dsize[index] = vec[i];
            index -= 1;
        }

        for (int i = index; i >= 0; --i) {
            dsize[i] = 1;
        }
    }

    return dsize;
}

template<int num_indices>
std::vector<int> ToVector(const EigenDSize<num_indices>& dsize) {
    std::vector<int> vec(num_indices);

    for (int i = 0; i < num_indices; ++i) {
        vec[i] = dsize[i];
    }

    return vec;
}

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_INCLUDE_EIGEN_HELPER_H_
