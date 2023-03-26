#ifndef SIMPLE_INFER_SRC_LAYER_REGISTRY_H_
#define SIMPLE_INFER_SRC_LAYER_REGISTRY_H_

#include <string>

namespace SimpleInfer {

class Layer;

using LayerCreatorFunc   = Layer* (*)();
using LayerDestroyerFunc = void (*)(Layer*);

struct LayerRegistryEntry {
    LayerCreatorFunc creator     = nullptr;
    LayerDestroyerFunc destroyer = nullptr;
};

const LayerRegistryEntry* GetLayerRegistry(std::string type);

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_SRC_LAYER_REGISTRY_H_
