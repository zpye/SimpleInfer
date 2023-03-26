#include "yolo_detect.h"

namespace SimpleInfer {

DEFINE_LAYER_REGISTRY(YoloDetect);

YoloDetect::YoloDetect() {}

YoloDetect::~YoloDetect() {}

Status YoloDetect::Init(const pnnx::Operator* op) {
    Status ret = Layer::Init(op);
    if (Status::kSuccess != ret) {
        return ret;
    }

    {
        // strides
        CHECK_BOOL(CheckAttr(op->attrs, "pnnx_5", 1));
        CHECK_BOOL(1 == op->attrs.at("pnnx_5").shape.size());
        CHECK_BOOL(3 == op->attrs.at("pnnx_5").shape[0]);

        const float* s =
            reinterpret_cast<const float*>(op->attrs.at("pnnx_5").data.data());

        for (int i = 0; i < num_spatial_sizes; ++i) {
            strides_[i] = s[i];
        }
    }

    for (int i = 0; i < num_spatial_sizes; ++i) {
        {
            // build conv_2d layer
            std::map<std::string, pnnx::Parameter> params;
            std::map<std::string, pnnx::Attribute> attrs;

            // weight
            const std::string weight_name = absl::StrFormat("m.%d.weight", i);
            CHECK_BOOL(CheckAttr(op->attrs, weight_name, 1));
            attrs["weight"] = op->attrs.at(weight_name);

            const std::vector<int>& weight_shape =
                op->attrs.at(weight_name).shape;
            CHECK_BOOL(1 == weight_shape[2] && 1 == weight_shape[3]);

            // bias
            const std::string bias_name = absl::StrFormat("m.%d.bias", i);
            CHECK_BOOL(CheckAttr(op->attrs, bias_name, 1));
            attrs["bias"] = op->attrs.at(bias_name);

            params["bias"]         = pnnx::Parameter(true);
            params["padding_mode"] = pnnx::Parameter("zeros");
            params["padding"]      = pnnx::Parameter({0, 0});
            params["kernel_size"]  = pnnx::Parameter({1, 1});
            params["stride"]       = pnnx::Parameter({1, 1});
            params["dilation"]     = pnnx::Parameter({1, 1});
            params["groups"]       = pnnx::Parameter(1);
            params["in_channels"]  = pnnx::Parameter(weight_shape[1]);
            params["out_channels"] = pnnx::Parameter(weight_shape[0]);

            CHECK_STATUS(conv_2d_layer_[i].Init(params, attrs));

            const std::vector<int>& input_shape = op->inputs[i]->shape;
            const std::vector<int> tensor_shape{input_shape[0],
                                                input_shape[2],
                                                input_shape[3],
                                                weight_shape[0]};
            CHECK_STATUS(
                spatial_output[i].Allocate(DataType::kFloat32, tensor_shape));

            if (0 == i) {
                num_elements_ = weight_shape[0];
            } else {
                CHECK_BOOL(num_elements_ == weight_shape[0]);
            }
        }

        {
            // anchor grids
            const std::string anchor_grid_name =
                absl::StrFormat("pnnx_%d", anchor_index[i]);
            CHECK_BOOL(CheckAttr(op->attrs, anchor_grid_name, 1));
            CHECK_BOOL(5 == op->attrs.at(anchor_grid_name).shape.size());
            CHECK_BOOL(1 == op->attrs.at(anchor_grid_name).shape[0]);
            CHECK_BOOL(2 == op->attrs.at(anchor_grid_name).shape[4]);

            // [1][anchor_grid_levels][H(i)][W(i)][2]
            EigenDSize<5> origin_anchor_grids_shape =
                ToEigenDSize<5>(op->attrs.at(anchor_grid_name).shape);
            std::vector<char> origin_anchor_grids =
                op->attrs.at(anchor_grid_name).data;
            anchor_grids_[i].resize(origin_anchor_grids.size());

            // [1][H(i) * W(i) * anchor_grid_levels][2]
            anchor_grids_shape_[i] = EigenDSize<3>(
                1,
                origin_anchor_grids_shape[2] * origin_anchor_grids_shape[3] *
                    origin_anchor_grids_shape[1],
                2);

            EigenTensorMap<float, 5> origin_anchor_grids_eigen_tensor(
                reinterpret_cast<float*>(origin_anchor_grids.data()),
                origin_anchor_grids_shape);
            EigenTensorMap<float, 3> anchor_grids_eigen_tensor(
                reinterpret_cast<float*>(anchor_grids_[i].data()),
                anchor_grids_shape_[i]);

            // [1][H(i)][W(i)][anchor_grid_levels][2]
            EigenDSize<5> shuffle_index(0, 2, 3, 1, 4);
            anchor_grids_eigen_tensor =
                origin_anchor_grids_eigen_tensor.shuffle(shuffle_index)
                    .reshape(anchor_grids_shape_[i]);

            // grids
            const std::string grid_name =
                absl::StrFormat("pnnx_%d", grid_index[i]);
            CHECK_BOOL(CheckAttr(op->attrs, grid_name, 1));
            CHECK_BOOL(5 == op->attrs.at(grid_name).shape.size());
            CHECK_BOOL(1 == op->attrs.at(grid_name).shape[0]);
            CHECK_BOOL(2 == op->attrs.at(grid_name).shape[4]);

            // [1][anchor_grid_levels][H(i)][W(i)][2]
            EigenDSize<5> origin_grids_shape =
                ToEigenDSize<5>(op->attrs.at(grid_name).shape);
            std::vector<char> origin_grids = op->attrs.at(grid_name).data;
            grids_[i].resize(origin_grids.size());

            // [1][H(i) * W(i) * anchor_grid_levels][2]
            grids_shape_[i] =
                EigenDSize<3>(1,
                              origin_grids_shape[2] * origin_grids_shape[3] *
                                  origin_grids_shape[1],
                              2);

            EigenTensorMap<float, 5> origin_grids_eigen_tensor(
                reinterpret_cast<float*>(origin_grids.data()),
                origin_grids_shape);
            EigenTensorMap<float, 3> grids_eigen_tensor(
                reinterpret_cast<float*>(grids_[i].data()),
                grids_shape_[i]);

            // [1][H(i)][W(i)][anchor_grid_levels][2]
            grids_eigen_tensor =
                origin_grids_eigen_tensor.shuffle(shuffle_index)
                    .reshape(grids_shape_[i]);

            // anchor levels
            CHECK_BOOL(origin_anchor_grids_shape == origin_grids_shape);
            if (0 == i) {
                num_anchor_grid_levels_ = origin_anchor_grids_shape[1];
            } else {
                CHECK_BOOL(num_anchor_grid_levels_ ==
                           origin_anchor_grids_shape[1]);
            }
        }
    }

    {
        // classes
        CHECK_BOOL(0 == num_elements_ % num_anchor_grid_levels_);
        num_classes_info_ = num_elements_ / num_anchor_grid_levels_;
    }

    return Status::kSuccess;
}

void YoloDetect::SetContext(Context* context) {
    Layer::SetContext(context);

    for (int i = 0; i < num_spatial_sizes; ++i) {
        conv_2d_layer_[i].SetContext(context);
    }
}

Status YoloDetect::Validate() {
    {
        Status ret = Layer::Validate();
        if (Status::kSuccess != ret) {
            return ret;
        }
    }

    {
        Status ret = ValidateShape(3, 1);
        if (Status::kSuccess != ret) {
            return ret;
        }
    }

    if (!(IsSameDataType<float>(input_tensor_nodes_[0]->tensor.GetDataType()) &&
          IsSameDataType<float>(
              output_tensor_nodes_[0]->tensor.GetDataType()))) {
        LOG(ERROR) << "YoloDetect::Validate fail ["
                   << "unsupport input/output data type"
                   << "]";
        return Status::kUnsupport;
    }

    // TODO: check shape

    return Status::kSuccess;
}

Status YoloDetect::Forward(const std::vector<Tensor>& inputs, Tensor& output) {
    GET_EIGEN_THREADPOOL_DEVICE(device);

    EigenTensorMap<float, 3> output_eigen_tensor =
        output.GetEigenTensor<float, 3>();

    // TODO: parallel
    int elements_offset = 0;
    for (int i = 0; i < num_spatial_sizes; ++i) {
        // conv1x1 for 3 inputs
        CHECK_STATUS(conv_2d_layer_[i].Forward(inputs[i], spatial_output[i]));

        // [N][H(i)][W(i)][anchor_grid_levels * classes_info]
        EigenTensorMap<float, 4> spatial_output_eigen_tensor =
            spatial_output[i].GetEigenTensor<float, 4>();

        EigenDSize<4> spatial_shape = spatial_output_eigen_tensor.dimensions();
        CHECK_BOOL(num_elements_ == spatial_shape[3]);

        // sigmoid & reshape [N][H(i) * W(i) * anchor_grid_levels][classes_info]
        EigenDSize<3> spatial_shape_new(
            spatial_shape[0],
            spatial_shape[1] * spatial_shape[2] * num_anchor_grid_levels_,
            num_classes_info_);

        // cat
        EigenDSize<3> output_offset(0, elements_offset, 0);
        output_eigen_tensor.slice(output_offset, spatial_shape_new)
            .device(*device) =
            spatial_output_eigen_tensor.sigmoid().reshape(spatial_shape_new);

        // anchor grid
        EigenDSize<3> output_xy_offset(0, elements_offset, 0);
        EigenDSize<3> output_wh_offset(0, elements_offset, 2);
        EigenDSize<3> broadcast_grids_shape(spatial_shape[0],
                                            grids_shape_[i][1],
                                            grids_shape_[i][2]);
        EigenDSize<3> broadcast_anchor_grids_shape(spatial_shape[0],
                                                   anchor_grids_shape_[i][1],
                                                   anchor_grids_shape_[i][2]);

        EigenTensorMap<float, 3> grids_eigen_tensor(
            reinterpret_cast<float*>(grids_[i].data()),
            grids_shape_[i]);
        EigenTensorMap<float, 3> anchor_grids_eigen_tensor(
            reinterpret_cast<float*>(anchor_grids_[i].data()),
            anchor_grids_shape_[i]);

        auto broadcast_grids =
            grids_eigen_tensor.broadcast(EigenDSize<3>(spatial_shape[0], 1, 1));
        auto broadcast_anchor_grids = anchor_grids_eigen_tensor.broadcast(
            EigenDSize<3>(spatial_shape[0], 1, 1));

        auto xy =
            output_eigen_tensor.slice(output_xy_offset, broadcast_grids_shape);
        auto wh = output_eigen_tensor.slice(output_wh_offset,
                                            broadcast_anchor_grids_shape);

        output_eigen_tensor.slice(output_xy_offset, broadcast_grids_shape)
            .device(*device) = (xy * 2.0f + broadcast_grids) * strides_[i];
        output_eigen_tensor
            .slice(output_wh_offset, broadcast_anchor_grids_shape)
            .device(*device) = (wh * 2.0f).pow(2.0f) * broadcast_anchor_grids;

        elements_offset += spatial_shape_new[1];
    }

    return Status::kSuccess;
}

}  // namespace SimpleInfer
