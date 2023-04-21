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

Eigen::ThreadPoolDevice* Context::GetDefaultEigenThreadPoolDevice() {
    static Eigen::ThreadPool default_eigen_threadpool(1);
    static Eigen::ThreadPoolDevice default_eigen_threadpool_device(
        &default_eigen_threadpool,
        1);

    return &default_eigen_threadpool_device;
}

}  // namespace SimpleInfer
