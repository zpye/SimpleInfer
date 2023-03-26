#include "pipeline_node.h"

namespace SimpleInfer {

void PipelineNode::SetLayer(Layer* layer) {
    layer_ = layer;
}

CStatus PipelineNode::run() {
    if (nullptr == layer_) {
        return CStatus("empty layer");
    }

    {
        Status ret = layer_->Forward();
        if (Status::kSuccess != ret) {
            LOG(ERROR) << "layer [" << layer_->GetOp()->name
                       << "] forward fail";
            return CStatus("layer forward fail");
        }
    }

    return CStatus();
}

}  // namespace SimpleInfer
