#include <pybind11/eigen/tensor.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "engine.h"
#include "tensor.h"
#include "types.h"

using namespace SimpleInfer;

namespace py = pybind11;

PYBIND11_MODULE(simpleinfer, m) {
    m.doc() = "pybind11 SimpleInfer";

    m.def("InitializeContext", &InitializeContext);

    py::enum_<DataType>(m, "DataType")
        .value("None", DataType::kNone)
        .value("Float32", DataType::kFloat32);

    py::enum_<Status>(m, "Status")
        .value("Success", Status::kSuccess)
        .value("Fail", Status::kFail)
        .value("Empty", Status::kEmpty)
        .value("ErrorShape", Status::kErrorShape)
        .value("ErrorContext", Status::kErrorContext)
        .value("Unsupport", Status::kUnsupport);

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<DataType, std::vector<int>>())
        .def("GetDataType",
             static_cast<const DataType (Tensor::*)() const>(
                 &Tensor::GetDataType))
        .def("Shape",
             static_cast<const std::vector<int>& (Tensor::*)() const>(
                 &Tensor::Shape))
        .def("SetTensorDim4",
             static_cast<Status (Tensor::*)(const EigenTensorMap<float, 4>&)>(
                 &Tensor::SetEigenTensor<float, 4>),
             py::return_value_policy::reference)
        .def("GetTensorDim4",
             static_cast<EigenTensorMap<float, 4> (Tensor::*)() const>(
                 &Tensor::GetEigenTensor<float, 4>),
             py::return_value_policy::reference);

    py::class_<Engine>(m, "Engine")
        .def(py::init<>())
        .def("LoadModel",
             static_cast<Status (Engine::*)(const std::string&,
                                            const std::string&)>(
                 &Engine::LoadModel))
        .def("Release", static_cast<Status (Engine::*)()>(&Engine::Release))
        .def("InputNames",
             static_cast<const std::vector<std::string> (Engine::*)()>(
                 &Engine::InputNames))
        .def("OutputNames",
             static_cast<const std::vector<std::string> (Engine::*)()>(
                 &Engine::OutputNames))
        .def("Input",
             static_cast<Status (Engine::*)(const std::string&, const Tensor&)>(
                 &Engine::Input))
        .def("Forward", static_cast<Status (Engine::*)()>(&Engine::Forward))
        .def("Extract",
             static_cast<Status (Engine::*)(const std::string&, Tensor&)>(
                 &Engine::Extract));
}
