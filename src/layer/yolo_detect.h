#ifndef SIMPLE_INFER_SRC_LAYER_YOLO_DETECT_H_
#define SIMPLE_INFER_SRC_LAYER_YOLO_DETECT_H_

#include "layer.h"

#include "conv_2d.h"

namespace SimpleInfer {

class YoloDetect : public Layer {
public:
    YoloDetect();

    virtual ~YoloDetect() override;

public:
    virtual Status Init(const pnnx::Operator* op) override;

    virtual void SetContext(Context* context) override;

    virtual Status Validate() override;

    virtual Status Forward(const std::vector<Tensor>& inputs,
                           Tensor& output) override;

public:
    static const int num_spatial_sizes = 3;
    static constexpr int anchor_index[num_spatial_sizes]{4, 2, 0};
    static constexpr int grid_index[num_spatial_sizes]{6, 3, 1};

    Conv2d conv_2d_layer_[num_spatial_sizes];
    Tensor spatial_output[num_spatial_sizes];
    EigenDSize<3> anchor_grids_shape_[num_spatial_sizes];
    std::vector<char> anchor_grids_[num_spatial_sizes];
    EigenDSize<3> grids_shape_[num_spatial_sizes];
    std::vector<char> grids_[num_spatial_sizes];
    float strides_[num_spatial_sizes];

    int num_elements_           = 255;
    int num_anchor_grid_levels_ = 3;
    int num_classes_info_       = 85;
};

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_SRC_LAYER_YOLO_DETECT_H_
