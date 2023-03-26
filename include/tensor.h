#ifndef SIMPLE_INFER_INCLUDE_TENSOR_H_
#define SIMPLE_INFER_INCLUDE_TENSOR_H_

#include <array>
#include <cassert>
#include <vector>

#include "eigen_helper.h"
#include "types.h"

namespace SimpleInfer {

class Tensor {
public:
    Tensor();

    Tensor(const DataType data_type,
           const std::vector<int>& shape,
           const bool allocate = false);

    ~Tensor();

    Tensor(const Tensor& tensor);

    Tensor& operator=(const Tensor& tensor);

public:
    Status Allocate();

    Status Allocate(const DataType data_type, const std::vector<int>& shape);

    Status Deallocate();

    const DataType GetDataType() const;

    const std::vector<int>& Shape() const;

public:
    template<typename T, int num_indices>
    Status SetEigenTensor(const EigenTensorMap<T, num_indices>& tensor_map) {
        if (use_internal_data_) {
            return Status::kFail;
        }

        assert(IsSameDataType<T>(data_type_));
        assert(num_indices >= 0);

        data_ = tensor_map.data();

        return Status::kSuccess;
    }

    template<typename T, int num_indices, int Options = 0x1>
    EigenTensorMap<T, num_indices> GetEigenTensor() const {
        assert(IsSameDataType<T>(data_type_));

        return EigenTensorMap<T, num_indices, Options>(
            static_cast<T*>(data_),
            ToEigenDSize<num_indices>(shape_));
    }

protected:
    DataType data_type_ = DataType::kNone;

    std::vector<int> shape_;

    bool use_internal_data_ = false;
    void* data_             = nullptr;
};

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_INCLUDE_TENSOR_H_
