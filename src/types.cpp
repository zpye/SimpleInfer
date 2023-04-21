#include "types.h"

#include <cstddef>
#include <cstdint>

namespace SimpleInfer {

template<>
bool IsSameDataType<float>(const DataType data_type) {
    return (DataType::kFloat32 == data_type);
}

template<>
bool IsSameDataType<double>(const DataType data_type) {
    return (DataType::kFloat64 == data_type);
}

template<>
bool IsSameDataType<int32_t>(const DataType data_type) {
    return (DataType::kInt32 == data_type);
}

template<>
bool IsSameDataType<int64_t>(const DataType data_type) {
    return (DataType::kInt64 == data_type);
}

template<>
bool IsSameDataType<int16_t>(const DataType data_type) {
    return (DataType::kInt16 == data_type);
}

template<>
bool IsSameDataType<int8_t>(const DataType data_type) {
    return (DataType::kInt8 == data_type);
}

template<>
bool IsSameDataType<uint8_t>(const DataType data_type) {
    return (DataType::kUint8 == data_type);
}

template<>
bool IsSameDataType<bool>(const DataType data_type) {
    return (DataType::kBool == data_type);
}

DataType PnnxToDataType(int type) {
    switch (type) {
        case 1:
            return DataType::kFloat32;
        case 2:
            return DataType::kFloat64;
        case 3:
            return DataType::kFloat16;
        case 4:
            return DataType::kInt32;
        case 5:
            return DataType::kInt64;
        case 6:
            return DataType::kInt16;
        case 7:
            return DataType::kInt8;
        case 8:
            return DataType::kUint8;
        case 9:
            return DataType::kBool;
        case 10:
            return DataType::kComplex64;
        case 11:
            return DataType::kComplex128;
        case 12:
            return DataType::kComplex32;
        default:
            return DataType::kNone;
    }

    return DataType::kNone;
}

int ElementSize(const DataType data_type) {
    switch (data_type) {
        case DataType::kInt8:
        case DataType::kUint8:
        case DataType::kBool:
            return 1;
        case DataType::kFloat16:
        case DataType::kInt16:
            return 2;
        case DataType::kFloat32:
        case DataType::kInt32:
        case DataType::kComplex32:
            return 4;
        case DataType::kFloat64:
        case DataType::kInt64:
        case DataType::kComplex64:
            return 8;
        case DataType::kComplex128:
            return 16;
        default:
            return 0;
    }

    return 0;
}

bool IsSameShape(const std::vector<int>& shape0,
                 const std::vector<int>& shape1) {
    if (shape0.size() != shape1.size()) {
        return false;
    }

    for (size_t i = 0; i < shape0.size(); ++i) {
        if (shape0[i] != shape1[i]) {
            return false;
        }
    }

    return true;
}

}  // namespace SimpleInfer
