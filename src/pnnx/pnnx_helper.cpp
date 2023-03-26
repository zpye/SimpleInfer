#include "pnnx_helper.h"

namespace SimpleInfer {

bool CheckParam(const std::map<std::string, pnnx::Parameter>& params,
                const std::string& name,
                const int type) {
    if (params.count(name) > 0) {
        if (type == params.at(name).type) {
            return true;
        }
    }

    return false;
}

bool CheckParam(const pnnx::Operator* op,
                const std::string& name,
                const int type) {
    return CheckParam(op->params, name, type);
}

bool CheckAttr(const std::map<std::string, pnnx::Attribute>& attrs,
               const std::string& name,
               const int type) {
    if (attrs.count(name) > 0) {
        if (type == attrs.at(name).type) {
            return true;
        }
    }

    return false;
}

bool CheckAttr(const pnnx::Operator* op,
               const std::string& name,
               const int type) {
    return CheckAttr(op->attrs, name, type);
}

}  // namespace SimpleInfer
