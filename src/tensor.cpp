#include "tensor.h"

#include <cstdlib>

namespace SimpleInfer {

Tensor::Tensor() {}

Tensor::Tensor(const DataType data_type,
               const std::vector<int>& shape,
               const bool allocate)
    : data_type_(data_type), shape_(shape) {
    if (allocate) {
        Allocate(data_type, shape);
    }
}

Tensor::~Tensor() {
    Deallocate();

    shape_.clear();

    data_type_ = DataType::kNone;
}

Tensor::Tensor(const Tensor& tensor)
    : data_type_(tensor.data_type_),
      shape_(tensor.shape_),
      use_internal_data_(false),
      data_(tensor.data_) {}

Tensor& Tensor::operator=(const Tensor& tensor) {
    data_type_ = tensor.data_type_;

    shape_ = tensor.shape_;

    use_internal_data_ = false;
    data_              = tensor.data_;

    return *this;
}

Status Tensor::Allocate() {
    int total_size = ElementSize(data_type_);
    for (const auto s : shape_) {
        total_size *= s;
    }

    if (total_size > 0) {
        // TODO: use memory pool
        data_ = malloc(total_size);
        if (nullptr != data_) {
            use_internal_data_ = true;

            return Status::kSuccess;
        }
    }

    return Status::kFail;
}

Status Tensor::Allocate(const DataType data_type,
                        const std::vector<int>& shape) {
    if ((data_type_ == data_type) && IsSameShape(shape_, shape) &&
        use_internal_data_ && (nullptr != data_)) {
        return Status::kSuccess;
    }

    Deallocate();

    data_type_ = data_type;

    shape_ = shape;

    return Allocate();
}

Status Tensor::Deallocate() {
    if (use_internal_data_) {
        if (nullptr != data_) {
            free(data_);
            data_ = nullptr;
        }

        use_internal_data_ = false;

        return Status::kSuccess;
    }

    return Status::kFail;
}

const DataType Tensor::GetDataType() const {
    return data_type_;
}

const std::vector<int>& Tensor::Shape() const {
    return shape_;
}

}  // namespace SimpleInfer
