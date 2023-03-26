#include "eigen_helper.h"

#include "logger.h"

using namespace SimpleInfer;

int main() {
    InitializeLogger();

    EigenTensor<float, 4> t(2, 2, 2, 2);
    for (int i = 0; i < (int)t.size(); ++i) {
        *(t.data() + i) = (float)i;
    }
    LOG(INFO) << "t:\n" << t;

    EigenTensor<float, 1> bias(2);
    for (int i = 0; i < (int)bias.size(); ++i) {
        *(bias.data() + i) = (float)(i + 1) * 0.1f;
    }
    LOG(INFO) << "bias:\n" << bias;

    t += bias.reshape(EigenDSize<4>(1, 1, 1, 2))
             .broadcast(EigenDSize<4>(2, 2, 2, 1));
    LOG(INFO) << "t + bias:\n" << t;

    return 0;
}
