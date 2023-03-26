#ifndef SIMPLE_INFER_SRC_PIPELINE_NODE_H_
#define SIMPLE_INFER_SRC_PIPELINE_NODE_H_

#include <CGraph.h>

#include "layer.h"
#include "logger.h"

namespace SimpleInfer {

class PipelineNode : public CGraph::GNode {
public:
    void SetLayer(Layer* layer);

    virtual CStatus run() override;

protected:
    Layer* layer_ = nullptr;
};

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_SRC_PIPELINE_NODE_H_
