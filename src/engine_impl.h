#ifndef SIMPLE_INFER_SRC_ENGINE_IMPL_H_
#define SIMPLE_INFER_SRC_ENGINE_IMPL_H_

#include <map>
#include <string>

#include <CGraph.h>

#include "context.h"
#include "layer.h"
#include "pipeline_node.h"
#include "pnnx/pnnx_helper.h"
#include "tensor.h"
#include "tensor_node.h"
#include "types.h"

namespace SimpleInfer {

class EngineImpl {
public:
    EngineImpl();

    ~EngineImpl();

public:
    Status LoadModel(const std::string& parampath, const std::string& binpath);

    Status Release();

    Status CreateContext();
    Status DestroyContext();

    Status CreateGraph(const std::string& parampath,
                       const std::string& binpath);
    Status DestroyGraph();

    Status CreateTensorNodes();
    Status DestroyTensorNodes();

    Status CreateLayers();
    Status DestroyLayers();

    Status CreatePipeline();
    Status DestroyPipeline();

    Status AllocateTensorMemory();
    Status DeallocateTensorMemory();

public:
    const std::vector<std::string> InputNames();
    const std::vector<std::string> OutputNames();

public:
    Status Input(const std::string& name, const Tensor& input);

    Status Forward();

    Status Extract(const std::string& name, Tensor& output);

private:
    Context* context_ = nullptr;

    pnnx::Graph* graph_ = nullptr;

    std::map<std::string, Layer*> layers_;
    std::map<std::string, TensorNode*> tensor_nodes_;

    std::map<std::string, TensorNode*> input_tensor_nodes_;
    std::map<std::string, TensorNode*> output_tensor_nodes_;

    CGraph::GPipelinePtr pipeline_ = nullptr;
    std::map<std::string, PipelineNode*> pipeline_nodes_;
};

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_SRC_ENGINE_IMPL_H_
