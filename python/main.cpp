/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatTensor

*/

#include <pybind11/pybind11.h>
#include <pyxtensor/pyxtensor.hpp>
// #include <GMatTensor/Cartesian2d.h>
#include <GMatTensor/Cartesian3d.h>

// Enable basic assertions on matrix shape
// (doesn't cost a lot of time, but avoids segmentation faults)
#define GMATTENSOR_ENABLE_ASSERT

namespace py = pybind11;

template <size_t rank, class T>
auto add_common_members_Cartesiand2d_Array(T& self)
{
    namespace SM = GMatTensor::Cartesian3d;

    self.def(py::init<std::array<size_t, rank>>(), "Array.", py::arg("shape"))
        .def("shape", &SM::Array<rank>::shape, "Shape of array.")
        .def("I2", &SM::Array<rank>::I2, "Array with 2nd-order unit tensors.")
        .def("II", &SM::Array<rank>::II, "Array with 4th-order tensors = dyadic(I2, I2).")
        .def("I4", &SM::Array<rank>::I4, "Array with 4th-order unit tensors.")
        .def("I4rt", &SM::Array<rank>::I4rt, "Array with 4th-order right-transposed unit tensors.")
        .def("I4s", &SM::Array<rank>::I4s, "Array with 4th-order symmetric projection tensors.")
        .def("I4d", &SM::Array<rank>::I4d, "Array with 4th-order deviatoric projection tensors.")
        .def("__repr__", [](const SM::Array<rank>&) { return "<GMatTensor.Cartesian3d.Array>"; });
}

PYBIND11_MODULE(GMatTensor, m)
{

    m.doc() = "Tensor operations and unit tensors support GMat models";

    // ----------------------
    // GMatTensor.Cartesian3d
    // ----------------------

    py::module sm = m.def_submodule("Cartesian3d", "3d Cartesian coordinates");

    namespace SM = GMatTensor::Cartesian3d;

    // Unit tensors

    sm.def("I2", &SM::I2, "Second order unit tensor.");
    sm.def("II", &SM::II, "Fourth order tensor with the result of the dyadic product II.");
    sm.def("I4", &SM::I4, "Fourth order unit tensor.");
    sm.def("I4rt", &SM::I4rt, "Fourth right-transposed order unit tensor.");
    sm.def("I4s", &SM::I4s, "Fourth order symmetric projection tensor.");
    sm.def("I4d", &SM::I4d, "Fourth order deviatoric projection tensor.");

    // Tensor algebra

    sm.def(
        "Deviatoric",
        static_cast<xt::xtensor<double, 4> (*)(const xt::xtensor<double, 4>&)>(
            &SM::Deviatoric<xt::xtensor<double, 4>>),
        "Deviatoric part of a 2nd-order tensor. Returns matrix of 2nd-order tensors.",
        py::arg("A"));

    sm.def(
        "Deviatoric",
        static_cast<xt::xtensor<double, 3> (*)(const xt::xtensor<double, 3>&)>(
            &SM::Deviatoric<xt::xtensor<double, 3>>),
        "Deviatoric part of a 2nd-order tensor. Returns list of 2nd-order tensors.",
        py::arg("A"));

    sm.def(
        "Deviatoric",
        static_cast<xt::xtensor<double, 2> (*)(const xt::xtensor<double, 2>&)>(
            &SM::Deviatoric<xt::xtensor<double, 2>>),
        "Deviatoric part of a 2nd-order tensor. Returns 2nd-order tensor.",
        py::arg("A"));

    sm.def(
        "Hydrostatic",
        static_cast<xt::xtensor<double, 2> (*)(const xt::xtensor<double, 4>&)>(
            &SM::Hydrostatic<xt::xtensor<double, 4>>),
        "Hydrostatic part of a 2nd-order tensor. Returns matrix (of scalars).",
        py::arg("A"));

    sm.def(
        "Hydrostatic",
        static_cast<xt::xtensor<double, 1> (*)(const xt::xtensor<double, 3>&)>(
            &SM::Hydrostatic<xt::xtensor<double, 3>>),
        "Hydrostatic part of a 2nd-order tensor. Returns list (of scalars).",
        py::arg("A"));

    sm.def(
        "Hydrostatic",
        static_cast<xt::xtensor<double, 0> (*)(const xt::xtensor<double, 2>&)>(
            &SM::Hydrostatic<xt::xtensor<double, 2>>),
        "Hydrostatic part of a 2nd-order tensor. Returns scalar.",
        py::arg("A"));

    sm.def(
        "Equivalent_deviatoric",
        static_cast<xt::xtensor<double, 2> (*)(const xt::xtensor<double, 4>&, double)>(
            &SM::Equivalent_deviatoric<xt::xtensor<double, 4>, double>),
        "Equivalent value of the tensor's deviator. Returns matrix (of scalars).",
        py::arg("A"),
        py::arg("factor"));

    sm.def(
        "Equivalent_deviatoric",
        static_cast<xt::xtensor<double, 1> (*)(const xt::xtensor<double, 3>&, double)>(
            &SM::Equivalent_deviatoric<xt::xtensor<double, 3>, double>),
        "Equivalent value of the tensor's deviator. Returns list (of scalars).",
        py::arg("A"),
        py::arg("factor"));

    sm.def(
        "Equivalent_deviatoric",
        static_cast<xt::xtensor<double, 0> (*)(const xt::xtensor<double, 2>&, double)>(
            &SM::Equivalent_deviatoric<xt::xtensor<double, 2>, double>),
        "Equivalent value of the tensor's deviator. Returns scalar.",
        py::arg("A"),
        py::arg("factor"));

    // Array

    py::class_<SM::Array<1>> array1d(sm, "Array1d");
    add_common_members_Cartesiand2d_Array<1>(array1d);

    py::class_<SM::Array<2>> array2d(sm, "Array2d");
    add_common_members_Cartesiand2d_Array<2>(array2d);

    py::class_<SM::Array<3>> array3d(sm, "Array3d");
    add_common_members_Cartesiand2d_Array<3>(array3d);
}
