/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatTensor

*/

#include <pybind11/pybind11.h>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pytensor.hpp>

#include "Cartesian2d.hpp"
#include "Cartesian3d.hpp"

// Enable basic assertions on matrix shape
// (doesn't cost a lot of time, but avoids segmentation faults)
#define GMATTENSOR_ENABLE_ASSERT

namespace py = pybind11;

PYBIND11_MODULE(GMatTensor, m)
{
    xt::import_numpy();

    m.doc() = "Tensor operations and unit tensors support GMat models";

    m.def("version",
          &GMatTensor::version,
          "Return version string.");

    {
        py::module sm = m.def_submodule("Cartesian2d", "2d Cartesian coordinates");
        init_Cartesian2d(sm);
    }

    {
        py::module sm = m.def_submodule("Cartesian3d", "3d Cartesian coordinates");
        init_Cartesian3d(sm);
    }
}
