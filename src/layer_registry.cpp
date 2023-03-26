#include "layer_registry.h"

#include <map>

namespace SimpleInfer {

#define DECLARE_LAYER_REGISTRY(type) \
    Layer* type##_LayerCreator();    \
    void type##_LayerDestroyer(Layer*);

// clang-format off
#define LAYER_REGISTRY_ITEM(pnnx_type, type)           \
    { #pnnx_type, { type##_LayerCreator, type##_LayerDestroyer } }
// clang-format on

// declarations
DECLARE_LAYER_REGISTRY(AdaptiveAvgPool2d)
DECLARE_LAYER_REGISTRY(BatchNorm2d)
DECLARE_LAYER_REGISTRY(BinaryOp)
DECLARE_LAYER_REGISTRY(Cat)
DECLARE_LAYER_REGISTRY(Conv2d)
DECLARE_LAYER_REGISTRY(Flatten)
DECLARE_LAYER_REGISTRY(HardSigmoid)
DECLARE_LAYER_REGISTRY(HardSwish)
DECLARE_LAYER_REGISTRY(Linear)
DECLARE_LAYER_REGISTRY(MaxPool2d)
DECLARE_LAYER_REGISTRY(ReLU)
DECLARE_LAYER_REGISTRY(Sigmoid)
DECLARE_LAYER_REGISTRY(SiLU)
DECLARE_LAYER_REGISTRY(Upsample)
DECLARE_LAYER_REGISTRY(YoloDetect)

static std::map<std::string, LayerRegistryEntry> layer_registry_map = {
    LAYER_REGISTRY_ITEM(nn.AdaptiveAvgPool2d, AdaptiveAvgPool2d),
    LAYER_REGISTRY_ITEM(nn.BatchNorm2d, BatchNorm2d),
    LAYER_REGISTRY_ITEM(BinaryOp, BinaryOp),
    LAYER_REGISTRY_ITEM(torch.cat, Cat),
    LAYER_REGISTRY_ITEM(nn.Conv2d, Conv2d),
    LAYER_REGISTRY_ITEM(torch.flatten, Flatten),
    LAYER_REGISTRY_ITEM(nn.Hardsigmoid, HardSigmoid),
    LAYER_REGISTRY_ITEM(nn.Hardswish, HardSwish),
    LAYER_REGISTRY_ITEM(nn.Linear, Linear),
    LAYER_REGISTRY_ITEM(nn.MaxPool2d, MaxPool2d),
    LAYER_REGISTRY_ITEM(nn.ReLU, ReLU),
    LAYER_REGISTRY_ITEM(nn.Sigmoid, Sigmoid),
    LAYER_REGISTRY_ITEM(nn.SiLU, SiLU),
    LAYER_REGISTRY_ITEM(nn.Upsample, Upsample),
    LAYER_REGISTRY_ITEM(models.yolo.Detect, YoloDetect),
};

const LayerRegistryEntry* GetLayerRegistry(std::string type) {
    if (layer_registry_map.count(type) > 0) {
        return &layer_registry_map[type];
    }

    return nullptr;
}

}  // namespace SimpleInfer
