#include <string>

#include "pnnx/expand_expression.h"
#include "pnnx/ir.h"

#include "logger.h"

static const std::string model_path(MODEL_PATH);

void ShowAttr(const std::string& name,
              const pnnx::Attribute& attr,
              const std::string pre_str) {
    std::string show_str = pre_str;
    absl::StrAppendFormat(&show_str,
                          "type [%d], name [%s], ",
                          attr.type,
                          name.c_str());

    absl::StrAppendFormat(&show_str, "shape [");
    for (auto& v : attr.shape) {
        absl::StrAppendFormat(&show_str, " %d", v);
    }
    absl::StrAppendFormat(&show_str, " ], ");

    absl::StrAppendFormat(&show_str, "data size [%d]", (int)attr.data.size());

    LOG(INFO) << show_str;
}

void ShowParam(const std::string& name,
               const pnnx::Parameter& param,
               const std::string pre_str) {
    std::string show_str = pre_str;
    absl::StrAppendFormat(&show_str,
                          "type [%d], name [%s], ",
                          param.type,
                          name.c_str());

    switch (param.type) {
        case 1: {
            absl::StrAppendFormat(&show_str,
                                  "value [%s]",
                                  ((param.b) ? ("True") : ("False")));
            break;
        }
        case 2: {
            absl::StrAppendFormat(&show_str, "value [%d]", param.i);
            break;
        }
        case 3: {
            absl::StrAppendFormat(&show_str, "value [%f]", param.f);
            break;
        }
        case 4: {
            absl::StrAppendFormat(&show_str, "value [%s]", param.s.c_str());
            break;
        }
        case 5: {
            absl::StrAppendFormat(&show_str, "value [");
            for (auto& v : param.ai) {
                absl::StrAppendFormat(&show_str, " %d", v);
            }
            absl::StrAppendFormat(&show_str, " ]");
            break;
        }
        case 6: {
            absl::StrAppendFormat(&show_str, "value [");
            for (auto& v : param.af) {
                absl::StrAppendFormat(&show_str, " %f", v);
            }
            absl::StrAppendFormat(&show_str, " ]");
            break;
        }
        case 7: {
            absl::StrAppendFormat(&show_str, "value [");
            for (auto& v : param.as) {
                absl::StrAppendFormat(&show_str, " %s", v.c_str());
            }
            absl::StrAppendFormat(&show_str, " ]");
            break;
        }
        default: {
            absl::StrAppendFormat(&show_str, "Unknown type");
            break;
        }
    }

    LOG(INFO) << show_str;
}

void ShowOp(const pnnx::Operator* op, const std::string pre_str) {
    LOG(INFO) << pre_str
              << absl::StrFormat("type [%s], name [%s]",
                                 op->type.c_str(),
                                 op->name.c_str());
    LOG(INFO) << pre_str
              << absl::StrFormat("%d inputs:", (int)op->inputs.size());
    int i = 0;
    for (const auto input : op->inputs) {
        LOG(INFO) << pre_str
                  << absl::StrFormat("%d: type [%d], name [%s]",
                                     i,
                                     input->type,
                                     input->name.c_str());
        i += 1;
    }

    LOG(INFO) << pre_str
              << absl::StrFormat("%d outputs:", (int)op->outputs.size());
    i = 0;
    for (const auto output : op->outputs) {
        LOG(INFO) << pre_str
                  << absl::StrFormat("%d: type [%d], name [%s]",
                                     i,
                                     output->type,
                                     output->name.c_str());
        i += 1;
    }

    LOG(INFO) << pre_str
              << absl::StrFormat("%d params:", (int)op->params.size());
    for (const auto& p : op->params) {
        ShowParam(p.first, p.second, pre_str);
    }

    LOG(INFO) << pre_str << absl::StrFormat("%d attrs:", (int)op->attrs.size());
    for (const auto& a : op->attrs) {
        ShowAttr(a.first, a.second, pre_str);
    }
}

void ShowOperand(const pnnx::Operand* operand, const std::string pre_str) {
    LOG(INFO) << pre_str
              << absl::StrFormat("type [%d], name [%s]",
                                 operand->type,
                                 operand->name.c_str());
    if (operand->producer) {
        LOG(INFO) << pre_str
                  << absl::StrFormat("producer: type [%s], name [%s]",
                                     operand->producer->type.c_str(),
                                     operand->producer->name.c_str());
    } else {
        LOG(INFO) << pre_str << absl::StrFormat("empty producer");
    }

    int i = 0;
    for (const auto consumer : operand->consumers) {
        if (consumer) {
            LOG(INFO) << pre_str
                      << absl::StrFormat("consumer %d: type [%s], name [%s]",
                                         i,
                                         consumer->type.c_str(),
                                         consumer->name.c_str());
        } else {
            LOG(INFO) << pre_str << absl::StrFormat("empty consumer %d", i);
        }

        i += 1;
    }

    std::string shape_str = pre_str;
    absl::StrAppendFormat(&shape_str, "shape [");
    for (auto& v : operand->shape) {
        absl::StrAppendFormat(&shape_str, " %d", v);
    }
    absl::StrAppendFormat(&shape_str, " ]");
    LOG(INFO) << shape_str;

    for (const auto& p : operand->params) {
        ShowParam(p.first, p.second, pre_str);
    }
}

void ShowGraph(const pnnx::Graph& graph) {
    // ops
    LOG(INFO) << ">>>>>>>>>> ops >>>>>>>>>>>>";

    int i = 0;
    for (auto op : graph.ops) {
        LOG(INFO) << ">>>>>>>>>>>>>>>>>>>>>>>>>";
        LOG(INFO) << absl::StrFormat("op %d", i);
        ShowOp(op, "");

        i += 1;
    }

    LOG(INFO) << ">>>>>>>>>> operand >>>>>>>>>>>>";

    i = 0;
    for (auto operand : graph.operands) {
        LOG(INFO) << ">>>>>>>>>>>>>>>>>>>>>>>>>";
        LOG(INFO) << absl::StrFormat("operand %d", i);
        ShowOperand(operand, "");

        i += 1;
    }
}

int main() {
    SimpleInfer::InitializeLogger();

    pnnx::Graph graph;
    // int ret = graph.load(model_path + "/resnet/resnet18_batch1.param",
    //                      model_path + "/resnet/resnet18_batch1.pnnx.bin");
    int ret = graph.load(model_path + "/add/resnet_add3.pnnx.param",
                         model_path + "/add/resnet_add3.pnnx.bin");
    // int ret = graph.load(model_path + "/yolo/demo/yolov5s_batch8.pnnx.param",
    //                      model_path + "/yolo/demo/yolov5s_batch8.pnnx.bin");

    // ShowGraph(graph);

    expand_expression(graph);

    ShowGraph(graph);

    return 0;
}
