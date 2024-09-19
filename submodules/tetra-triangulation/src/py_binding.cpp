#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <iostream>
#include <memory>
#include <string>

#include "triangulation.h"

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the `_a` literal

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_DEVICE(x) TORCH_CHECK(x.device() == this->device, #x " must be on the same device")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must have float32 type")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT_DIM3(x) \
    CHECK_INPUT(x);         \
    CHECK_DEVICE(x);        \
    CHECK_FLOAT(x);         \
    TORCH_CHECK(x.size(-1) == 3, #x " must have last dimension with size 3")

torch::Tensor py_triangulate(const torch::Tensor &points) {
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "points must have shape [num_points, 3]");
    const auto points_ = points.cpu().contiguous();

    std::vector<uint4> cells = triangulate(
        points_.size(0),
        reinterpret_cast<float3 *>(points_.data_ptr()));

    if (cells.size() >= (size_t)std::numeric_limits<int>::max) {
        throw Exception("Too many points!");
    }
    auto cells_out = torch::empty({(long)cells.size(), 4}, torch::dtype(torch::kInt32).device(torch::kCPU));
    memcpy(
        cells_out.data_ptr(),
        reinterpret_cast<void *>(cells.data()),
        cells.size() * sizeof(uint4));
    return cells_out.to(points.device());
};


PYBIND11_MODULE(tetranerf_cpp_extension, m) {
    m.def("triangulate", &py_triangulate);
}