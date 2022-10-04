/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatTensor

*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pytensor.hpp>

#define GMATTENSOR_USE_XTENSOR_PYTHON
#include <GMatTensor/version.h>

#include "Cartesian2d.hpp"
#include "Cartesian3d.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_GMatTensor, m)
{
    xt::import_numpy();

    m.doc() = "Tensor operations and unit tensors support GMat models";

    m.def("version", &GMatTensor::version, "Version string.");

    m.def(
        "version_dependencies",
        &GMatTensor::version_dependencies,
        "List of version strings, include dependencies.",
        py::arg("greedy") = true);

    m.def(
        "version_compiler",
        &GMatTensor::version_compiler,
        "Information on the compiler, the platform, the C++ standard, and the compilation data.");

    {
        py::module sm = m.def_submodule("Cartesian2d", "2d Cartesian coordinates");
        init_Cartesian2d(sm);
    }

    {
        py::module sm = m.def_submodule("Cartesian3d", "3d Cartesian coordinates");
        init_Cartesian3d(sm);
    }
}
