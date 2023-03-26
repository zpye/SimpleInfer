#ifndef SIMPLE_INFER_SRC_CONTEXT_H_
#define SIMPLE_INFER_SRC_CONTEXT_H_

#include <memory>

#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/ThreadPool>

namespace SimpleInfer {

class Context {
public:
    Context();

    virtual ~Context();

public:
    // Eigen ThreadPool Device
    void InitEigenThreadPoolDevice(int num_threads = 8);

    Eigen::ThreadPoolDevice* GetEigenThreadPoolDevice();

protected:
    std::unique_ptr<Eigen::ThreadPool> eigen_threadpool_;
    std::unique_ptr<Eigen::ThreadPoolDevice> eigen_threadpool_device_;
};

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_SRC_CONTEXT_H_
