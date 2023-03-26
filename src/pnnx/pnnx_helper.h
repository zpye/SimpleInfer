#ifndef SIMPLE_INFER_SRC_PNNX_PNNX_HELPER_H_
#define SIMPLE_INFER_SRC_PNNX_PNNX_HELPER_H_

#include "ir.h"

namespace SimpleInfer {

bool CheckParam(const std::map<std::string, pnnx::Parameter>& params,
                const std::string& name,
                const int type);

bool CheckParam(const pnnx::Operator* op,
                const std::string& name,
                const int type);

bool CheckAttr(const std::map<std::string, pnnx::Attribute>& attrs,
               const std::string& name,
               const int type);

bool CheckAttr(const pnnx::Operator* op,
               const std::string& name,
               const int type);

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_SRC_PNNX_PNNX_HELPER_H_
