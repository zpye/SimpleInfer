#ifndef SIMPLE_INFER_SRC_LAYER_H_
#define SIMPLE_INFER_SRC_LAYER_H_

#include <vector>

#include "context.h"
#include "logger.h"
#include "pnnx/pnnx_helper.h"
#include "tensor.h"
#include "tensor_node.h"
#include "types.h"

namespace SimpleInfer {

class Layer {
public:
    Layer();

    virtual ~Layer();

public:
    virtual Status Init(const pnnx::Operator* op);

    virtual Status Init(const std::map<std::string, pnnx::Parameter>& params,
                        const std::map<std::string, pnnx::Attribute>& attrs);

    virtual void SetContext(Context* context);

    virtual void SetInputNodes(
        const std::vector<TensorNode*>& input_tensor_nodes);

    virtual void SetOutputNodes(
        const std::vector<TensorNode*>& output_tensor_nodes);

    virtual Status Deinit();

    virtual Status Validate();

    virtual Status Forward();

    virtual Status Forward(const Tensor& input, Tensor& output);

    virtual Status Forward(const std::vector<Tensor>& inputs, Tensor& output);

    virtual Status Forward(const Tensor& input, std::vector<Tensor>& outputs);

    virtual Status Forward(const std::vector<Tensor>& inputs,
                           std::vector<Tensor>& outputs);

    const pnnx::Operator* GetOp();

protected:
    virtual Status ValidateShape(const int input_size, const int output_size);

    Eigen::ThreadPoolDevice* GetEigenThreadPoolDevice();

protected:
    Context* context_ = nullptr;

    const pnnx::Operator* op_ = nullptr;

    std::vector<TensorNode*> input_tensor_nodes_;
    std::vector<TensorNode*> output_tensor_nodes_;
};

// get eigen threadpool device
#define GET_EIGEN_THREADPOOL_DEVICE(device)                       \
    Eigen::ThreadPoolDevice* device = GetEigenThreadPoolDevice(); \
    if (nullptr == device) {                                      \
        LOG(ERROR) << "Empty Eigen ThreadPool Device";            \
        return Status::kErrorContext;                             \
    }

// default layer registry entry
#define DEFINE_LAYER_CREATOR(type) \
    Layer* type##_LayerCreator() { \
        return (new type);         \
    }

#define DEFINE_LAYER_DESTROYER(type)           \
    void type##_LayerDestroyer(Layer* layer) { \
        if (nullptr != layer) {                \
            delete layer;                      \
        }                                      \
    }

#define DEFINE_LAYER_REGISTRY(type) \
    DEFINE_LAYER_CREATOR(type)      \
    DEFINE_LAYER_DESTROYER(type)

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_SRC_LAYER_H_
