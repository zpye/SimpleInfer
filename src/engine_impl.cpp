#include "engine_impl.h"

#include "layer.h"
#include "layer_registry.h"
#include "logger.h"
#include "pnnx/expand_expression.h"

namespace SimpleInfer {

EngineImpl::EngineImpl() {}

EngineImpl::~EngineImpl() {
    Release();
}

Status EngineImpl::LoadModel(const std::string& parampath,
                             const std::string& binpath) {
    {
        Status ret = Release();
        if (Status::kSuccess != ret) {
            LOG(ERROR) << "Release fail";
            return ret;
        }
    }

    {
        Status ret = CreateContext();
        if (Status::kSuccess != ret) {
            LOG(ERROR) << "CreateContext fail";
            return ret;
        }
    }

    {
        Status ret = CreateGraph(parampath, binpath);
        if (Status::kSuccess != ret) {
            LOG(ERROR) << "CreateGraph fail";
            return ret;
        }
    }

    {
        Status ret = CreateTensorNodes();
        if (Status::kSuccess != ret) {
            LOG(ERROR) << "CreateTensorNodes fail";
            return ret;
        }
    }

    {
        Status ret = CreateLayers();
        if (Status::kSuccess != ret) {
            LOG(ERROR) << "CreateLayers fail";
            return ret;
        }
    }

    {
        Status ret = CreatePipeline();
        if (Status::kSuccess != ret) {
            LOG(ERROR) << "CreatePipeline fail";
            return ret;
        }
    }

    {
        Status ret = AllocateTensorMemory();
        if (Status::kSuccess != ret) {
            LOG(ERROR) << "AllocateTensorMemory fail";
            return ret;
        }
    }

    return Status::kSuccess;
}

Status EngineImpl::Release() {
    {
        Status ret = DeallocateTensorMemory();
        if (Status::kSuccess != ret) {
            LOG(ERROR) << "DeallocateTensorMemory fail";
            return ret;
        }
    }

    {
        Status ret = DestroyPipeline();
        if (Status::kSuccess != ret) {
            LOG(ERROR) << "DestroyPipeline fail";
            return ret;
        }
    }

    {
        Status ret = DestroyLayers();
        if (Status::kSuccess != ret) {
            LOG(ERROR) << "DestroyLayers fail";
            return ret;
        }
    }

    {
        Status ret = DestroyTensorNodes();
        if (Status::kSuccess != ret) {
            LOG(ERROR) << "DestroyTensorNodes fail";
            return ret;
        }
    }

    {
        Status ret = DestroyGraph();
        if (Status::kSuccess != ret) {
            LOG(ERROR) << "DestroyGraph fail";
            return ret;
        }
    }

    {
        Status ret = DestroyContext();
        if (Status::kSuccess != ret) {
            LOG(ERROR) << "DestroyContext fail";
            return ret;
        }
    }

    return Status::kSuccess;
}

Status EngineImpl::CreateContext() {
    context_ = new Context;

    // TODO: set from user
    context_->InitEigenThreadPoolDevice(16);

    return Status::kSuccess;
}

Status EngineImpl::DestroyContext() {
    if (nullptr != context_) {
        delete context_;
        context_ = nullptr;
    }

    return Status::kSuccess;
}

Status EngineImpl::CreateGraph(const std::string& parampath,
                               const std::string& binpath) {
    graph_  = new pnnx::Graph;
    int ret = graph_->load(parampath, binpath);
    if (0 != ret) {
        LOG(ERROR) << "load graph fail";
        return Status::kFail;
    }

    pnnx::expand_expression(*graph_);

    return Status::kSuccess;
}

Status EngineImpl::DestroyGraph() {
    if (nullptr != graph_) {
        delete graph_;
        graph_ = nullptr;
    }

    return Status::kSuccess;
}

Status EngineImpl::CreateTensorNodes() {
    for (size_t i = 0; i < graph_->operands.size(); ++i) {
        pnnx::Operand* opd = graph_->operands[i];

        if (tensor_nodes_.count(opd->name) > 0) {
            LOG(ERROR) << "tensor node [" << opd->name << "] already exists";
            return Status::kFail;
        }

        TensorNode* tensor_node = new TensorNode;
        tensor_node->operand    = opd;

        // NCHW -> NHWC
        std::vector<int> shape_nhwc = opd->shape;
        if (shape_nhwc.size() > 3) {
            int shape_dims             = (int)shape_nhwc.size();
            shape_nhwc[shape_dims - 1] = opd->shape[shape_dims - 3];
            shape_nhwc[shape_dims - 2] = opd->shape[shape_dims - 1];
            shape_nhwc[shape_dims - 3] = opd->shape[shape_dims - 2];
        }

        LOG(INFO) << "tensor node [" << opd->name << "]" << opd->type << " "
                  << shape_nhwc[0] << " " << shape_nhwc[1] << " "
                  << shape_nhwc[2] << " " << shape_nhwc[3];

        tensor_node->tensor =
            Tensor(PnnxToDataType(opd->type), shape_nhwc, false);

        tensor_nodes_[opd->name] = tensor_node;

        if (nullptr != opd->producer) {
            // type == "pnnx.Input"
            if (0 == opd->producer->inputs.size()) {
                input_tensor_nodes_[opd->name] = tensor_node;
            }
        }

        for (auto& consumer : opd->consumers) {
            if (nullptr != consumer) {
                // type == "pnnx.Output"
                if (0 == consumer->outputs.size()) {
                    output_tensor_nodes_[opd->name] = tensor_node;
                    break;
                }
            }
        }
    }

    return Status::kSuccess;
}

Status EngineImpl::DestroyTensorNodes() {
    input_tensor_nodes_.clear();
    output_tensor_nodes_.clear();

    for (auto& tensor_node_iter : tensor_nodes_) {
        TensorNode* tensor_node = tensor_node_iter.second;
        if (nullptr != tensor_node) {
            delete tensor_node;
            tensor_node = nullptr;
        }
    }

    return Status::kSuccess;
}

Status EngineImpl::CreateLayers() {
    for (size_t i = 0; i < graph_->ops.size(); ++i) {
        pnnx::Operator* op = graph_->ops[i];

        if ("pnnx.Input" == op->type || "pnnx.Output" == op->type) {
            continue;
        }

        if (layers_.count(op->name) > 0) {
            LOG(ERROR) << "layer [" << op->name << "] already exists";
            return Status::kFail;
        }

        const LayerRegistryEntry* layer_registry_entry =
            GetLayerRegistry(op->type);
        if (nullptr == layer_registry_entry) {
            LOG(ERROR) << "layer type [" << op->type << "] not registered";
            return Status::kEmpty;
        }

        Layer* layer = layer_registry_entry->creator();
        if (nullptr == layer) {
            LOG(ERROR) << "create layer [" << op->type << "] fail";
            return Status::kFail;
        }

        {
            Status ret = layer->Init(op);
            if (Status::kSuccess != ret) {
                LOG(ERROR) << "layer [" << op->name << "] init fail";
                return ret;
            }

            layer->SetContext(context_);
        }

        {
            std::vector<SimpleInfer::TensorNode*> input_tensor_nodes;
            for (size_t j = 0; j < op->inputs.size(); ++j) {
                pnnx::Operand* opd = op->inputs[j];
                if (tensor_nodes_.count(opd->name) > 0) {
                    input_tensor_nodes.push_back(tensor_nodes_[opd->name]);
                } else {
                    LOG(ERROR) << "tensor node [" << op->name << "] not exist";
                    return Status::kEmpty;
                }
            }

            layer->SetInputNodes(input_tensor_nodes);
        }

        {
            std::vector<SimpleInfer::TensorNode*> output_tensor_nodes;
            for (size_t j = 0; j < op->outputs.size(); ++j) {
                pnnx::Operand* opd = op->outputs[j];
                if (tensor_nodes_.count(opd->name) > 0) {
                    output_tensor_nodes.push_back(tensor_nodes_[opd->name]);
                } else {
                    LOG(ERROR) << "tensor node [" << op->name << "] not exist";
                    return Status::kEmpty;
                }
            }

            layer->SetOutputNodes(output_tensor_nodes);
        }

        {
            Status ret = layer->Validate();
            if (Status::kSuccess != ret) {
                LOG(ERROR) << "layer [" << op->name << "] validate fail";
                return ret;
            }
        }

        layers_[op->name] = layer;
    }

    return Status::kSuccess;
}

Status EngineImpl::DestroyLayers() {
    for (auto& layer_iter : layers_) {
        Layer* layer             = layer_iter.second;
        const pnnx::Operator* op = layer->GetOp();

        Status ret = layer->Deinit();
        if (Status::kSuccess != ret) {
            LOG(ERROR) << "layer [" << op->name << "] deinit fail";
            return ret;
        }

        const LayerRegistryEntry* layer_registry_entry =
            GetLayerRegistry(op->type);
        if (nullptr == layer_registry_entry) {
            LOG(ERROR) << "layer type [" << op->type << "] not registered";
            return Status::kEmpty;
        }

        layer_registry_entry->destroyer(layer);
    }

    return Status::kSuccess;
}

Status EngineImpl::CreatePipeline() {
    pipeline_ = CGraph::GPipelineFactory::create();
    if (nullptr == pipeline_) {
        LOG(ERROR) << "create pipeline fail";
        return Status::kEmpty;
    }

    for (auto& layer_iter : layers_) {
        Layer* layer           = layer_iter.second;
        std::string layer_name = layer->GetOp()->name;

        if (pipeline_nodes_.count(layer_name) > 0) {
            LOG(ERROR) << "pipeline node [" << layer_name << "] already exists";
            return Status::kFail;
        }

        CGraph::GElementPtr element = nullptr;
        CStatus ret =
            pipeline_->registerGElement<PipelineNode>(&element, {}, layer_name);
        if (!ret.isOK() || nullptr == element) {
            LOG(ERROR) << "registerGElement fail";
            return Status::kFail;
        }

        PipelineNode* node = static_cast<PipelineNode*>(element);
        node->SetLayer(layer);
        node->setName(layer_name);
        pipeline_nodes_[layer_name] = node;
    }

    for (auto& tensor_node_iter : tensor_nodes_) {
        TensorNode* tensor_node = tensor_node_iter.second;

        pnnx::Operator* producer = tensor_node->operand->producer;
        if ("pnnx.Input" == producer->type) {
            continue;
        }

        std::string producer_name = producer->name;

        if (pipeline_nodes_.count(producer_name) <= 0) {
            LOG(ERROR) << "layer [" << producer_name << "] not in pipeline";
            return Status::kFail;
        }

        PipelineNode* producer_node = pipeline_nodes_[producer_name];
        CGraph::GElementPtrSet element_set{
            static_cast<CGraph::GElementPtr>(producer_node)};

        std::vector<pnnx::Operator*> consumers =
            tensor_node->operand->consumers;
        for (auto& consumer : consumers) {
            if ("pnnx.Output" == consumer->type) {
                continue;
            }

            std::string consumer_name = consumer->name;

            if (pipeline_nodes_.count(consumer_name) > 0) {
                PipelineNode* consumer_node = pipeline_nodes_[consumer_name];
                consumer_node->addDependGElements(element_set);
            } else {
                LOG(ERROR) << "layer [" << consumer_name << "] not in pipeline";
                return Status::kFail;
            }
        }
    }

    {
        // pipiline config
        pipeline_->setGEngineType(CGraph::GEngineType::STATIC);

#if 0
        CSize size;
        CStatus ret = pipeline_->calcMaxPara(size);
        if (!ret.isOK()) {
            LOG(ERROR) << "pipeline calcMaxPara fail";
            return Status::kFail;
        }

        LOG(INFO) << "pipeline calcMaxPara size [" << size << "]";

        pipeline_thread_pool_config_.default_thread_size_ = (int)size;
        pipeline_thread_pool_config_.max_thread_size_     = (int)size;
#else
        pipeline_thread_pool_config_.default_thread_size_ = 2;
        pipeline_thread_pool_config_.max_thread_size_     = 2;
#endif

        pipeline_->setUniqueThreadPoolConfig(pipeline_thread_pool_config_);
    }

    {
        CStatus ret = pipeline_->init();
        if (!ret.isOK()) {
            LOG(ERROR) << "pipeline init fail";
            return Status::kFail;
        }
    }

    return Status::kSuccess;
}

Status EngineImpl::DestroyPipeline() {
    pipeline_nodes_.clear();

    if (nullptr != pipeline_) {
        {
            CStatus ret = pipeline_->destroy();
            if (!ret.isOK()) {
                LOG(ERROR) << "pipeline destroy fail";
                return Status::kFail;
            }
        }

        {
            CStatus ret = CGraph::GPipelineFactory::remove(pipeline_);
            if (!ret.isOK()) {
                LOG(ERROR) << "remove pipeline fail";
                return Status::kFail;
            }
        }

        pipeline_ = nullptr;
    }

    return Status::kSuccess;
}

Status EngineImpl::AllocateTensorMemory() {
    // TODO: memory reuse
    for (auto& tensor_node_iter : tensor_nodes_) {
        if (input_tensor_nodes_.count(tensor_node_iter.first) > 0) {
            // input tensor use external memory
            continue;
        }

        Status ret = tensor_node_iter.second->tensor.Allocate();
        if (Status::kSuccess != ret) {
            LOG(ERROR) << "allocate tensor [" << tensor_node_iter.first
                       << "] memory fail";
            return ret;
        }
    }

    return Status::kSuccess;
}

Status EngineImpl::DeallocateTensorMemory() {
    for (auto& tensor_node_iter : tensor_nodes_) {
        if (input_tensor_nodes_.count(tensor_node_iter.first) > 0) {
            // input tensor use external memory
            continue;
        }

        TensorNode* tensor_node = tensor_node_iter.second;

        Status ret = tensor_node->tensor.Deallocate();
        if (Status::kSuccess != ret) {
            LOG(ERROR) << "deallocate tensor memory ["
                       << tensor_node->operand->name << "] fail";
            return ret;
        }
    }

    return Status::kSuccess;
}

const std::vector<std::string> EngineImpl::InputNames() {
    std::vector<std::string> ret;
    for (auto& input_tensor_node_iter : input_tensor_nodes_) {
        ret.push_back(input_tensor_node_iter.first);
    }

    return ret;
}

const std::vector<std::string> EngineImpl::OutputNames() {
    std::vector<std::string> ret;
    for (auto& output_tensor_node_iter : output_tensor_nodes_) {
        ret.push_back(output_tensor_node_iter.first);
    }

    return ret;
}

Status EngineImpl::Input(const std::string& name, const Tensor& input) {
    if (input_tensor_nodes_.count(name) <= 0) {
        LOG(ERROR) << "tensor [" << name << "] is not an input tensor";
        return Status::kFail;
    }

    input_tensor_nodes_[name]->tensor = input;

    return Status::kSuccess;
}

Status EngineImpl::Forward() {
    // TODO: add runtime options
    {
        CStatus ret = pipeline_->run();
        if (!ret.isOK()) {
            LOG(ERROR) << "pipeline run fail";
            return Status::kFail;
        }
    }

    return Status::kSuccess;
}

Status EngineImpl::Extract(const std::string& name, Tensor& output) {
    if (output_tensor_nodes_.count(name) <= 0) {
        LOG(ERROR) << "tensor [" << name << "] is not an output tensor";
        return Status::kFail;
    }

    output = output_tensor_nodes_[name]->tensor;

    return Status::kSuccess;
}

}  // namespace SimpleInfer
