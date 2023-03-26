#ifndef SIMPLE_INFER_INCLUDE_TYPES_H_
#define SIMPLE_INFER_INCLUDE_TYPES_H_

#include <vector>

namespace SimpleInfer {

enum class DataType {
    kNone = 0,
    kFloat32,
    kFloat64,
    kFloat16,
    kInt32,
    kInt64,
    kInt16,
    kInt8,
    kUint8,
    kBool,
    kComplex64,
    kComplex128,
    kComplex32
};

enum class Status {
    kSuccess = 0,
    kFail,
    kEmpty,
    kErrorShape,
    kErrorContext,
    kUnsupport
};

#define CHECK_BOOL(b)             \
    {                             \
        const bool _b = (b);      \
        if (!_b) {                \
            return Status::kFail; \
        }                         \
    }

#define CHECK_STATUS(s)                    \
    {                                      \
        const Status _status = (s);        \
        if (Status::kSuccess != _status) { \
            return _status;                \
        }                                  \
    }

template<typename T>
bool IsSameDataType(const DataType data_type);

DataType PnnxToDataType(int type);

int ElementSize(const DataType data_type);

bool IsSameShape(const std::vector<int>& shape0,
                 const std::vector<int>& shape1);

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_INCLUDE_TYPES_H_
