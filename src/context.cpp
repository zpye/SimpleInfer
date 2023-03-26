#include "context.h"

namespace SimpleInfer {

Context::Context() {}

Context::~Context() {}

void Context::InitEigenThreadPoolDevice(int num_threads) {
    eigen_threadpool_.reset(new Eigen::ThreadPool(num_threads));
    eigen_threadpool_device_.reset(
        new Eigen::ThreadPoolDevice(eigen_threadpool_.get(), num_threads));
}

Eigen::ThreadPoolDevice* Context::GetEigenThreadPoolDevice() {
    return eigen_threadpool_device_.get();
}

}  // namespace SimpleInfer
