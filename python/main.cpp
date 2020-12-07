/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatTensor

*/

#include <pybind11/pybind11.h>
#include <pyxtensor/pyxtensor.hpp>
#include <GMatTensor/Cartesian2d.h>
#include <GMatTensor/Cartesian3d.h>

// Enable basic assertions on matrix shape
// (doesn't cost a lot of time, but avoids segmentation faults)
#define GMATTENSOR_ENABLE_ASSERT

namespace py = pybind11;

// -----------
// Cartesian2d
// -----------

template <class S, class T>
auto construct_Array_2d(T& self)
{
    self.def(py::init<std::array<size_t, S::rank>>(), "Array.", py::arg("shape"))
        .def("shape", &S::shape, "Shape of array.")
        .def("I2", &S::I2, "Array with 2nd-order unit tensors.")
        .def("II", &S::II, "Array with 4th-order tensors = dyadic(I2, I2).")
        .def("I4", &S::I4, "Array with 4th-order unit tensors.")
        .def("I4rt", &S::I4rt, "Array with 4th-order right-transposed unit tensors.")
        .def("I4s", &S::I4s, "Array with 4th-order symmetric projection tensors.")
        .def("I4d", &S::I4d, "Array with 4th-order deviatoric projection tensors.")
        .def("__repr__", [](const S&) { return "<GMatTensor.Cartesian2d.Array>"; });
}

template <class S, class T>
void add_deviatoric_overloads_2d(T& module)
{
    module.def(
        "Deviatoric",
        static_cast<S (*)(const S&)>(&GMatTensor::Cartesian2d::Deviatoric<S>),
        "Deviatoric part of a(n) (array of) tensor(s).",
        py::arg("A"));
}

template <class R, class S, class T>
void add_hydrostatic_overloads_2d(T& module)
{
    module.def(
        "Hydrostatic",
        static_cast<R (*)(const S&)>(&GMatTensor::Cartesian2d::Hydrostatic<S>),
        "Hydrostatic part of a(n) (array of) tensor(s).",
        py::arg("A"));
}

template <class R, class S, class T>
void add_norm_deviatoric_overloads_2d(T& module)
{
    module.def(
        "Norm_deviatoric",
        static_cast<R (*)(const S&)>(&GMatTensor::Cartesian2d::Norm_deviatoric<S>),
        "Norm of the deviatoric part of a(n) (array of) tensor(s).",
        py::arg("A"));
}

// -----------
// Cartesian3d
// -----------

template <class S, class T>
auto construct_Array_3d(T& self)
{
    self.def(py::init<std::array<size_t, S::rank>>(), "Array.", py::arg("shape"))
        .def("shape", &S::shape, "Shape of array.")
        .def("I2", &S::I2, "Array with 2nd-order unit tensors.")
        .def("II", &S::II, "Array with 4th-order tensors = dyadic(I2, I2).")
        .def("I4", &S::I4, "Array with 4th-order unit tensors.")
        .def("I4rt", &S::I4rt, "Array with 4th-order right-transposed unit tensors.")
        .def("I4s", &S::I4s, "Array with 4th-order symmetric projection tensors.")
        .def("I4d", &S::I4d, "Array with 4th-order deviatoric projection tensors.")
        .def("__repr__", [](const S&) { return "<GMatTensor.Cartesian3d.Array>"; });
}

template <class S, class T>
void add_deviatoric_overloads_3d(T& module)
{
    module.def(
        "Deviatoric",
        static_cast<S (*)(const S&)>(&GMatTensor::Cartesian3d::Deviatoric<S>),
        "Deviatoric part of a (array of) tensor(s).",
        py::arg("A"));
}

template <class R, class S, class T>
void add_hydrostatic_overloads_3d(T& module)
{
    module.def(
        "Hydrostatic",
        static_cast<R (*)(const S&)>(&GMatTensor::Cartesian3d::Hydrostatic<S>),
        "Hydrostatic part of a (array of) tensor(s).",
        py::arg("A"));
}

template <class R, class S, class T>
void add_norm_deviatoric_overloads_3d(T& module)
{
    module.def(
        "Norm_deviatoric",
        static_cast<R (*)(const S&)>(&GMatTensor::Cartesian3d::Norm_deviatoric<S>),
        "Norm of the deviatoric part of a(n) (array of) tensor(s).",
        py::arg("A"));
}

PYBIND11_MODULE(GMatTensor, m)
{

    m.doc() = "Tensor operations and unit tensors support GMat models";

    // ----------------------
    // GMatTensor.Cartesian2d
    // ----------------------

    {
        py::module sm = m.def_submodule("Cartesian2d", "2d Cartesian coordinates");
        namespace SM = GMatTensor::Cartesian2d;

        // Unit tensors

        sm.def("O2", &SM::O2, "Second order null tensor.");
        sm.def("O4", &SM::O4, "Fourth order null tensor.");
        sm.def("I2", &SM::I2, "Second order unit tensor.");
        sm.def("II", &SM::II, "Fourth order tensor with the result of the dyadic product II.");
        sm.def("I4", &SM::I4, "Fourth order unit tensor.");
        sm.def("I4rt", &SM::I4rt, "Fourth right-transposed order unit tensor.");
        sm.def("I4s", &SM::I4s, "Fourth order symmetric projection tensor.");
        sm.def("I4d", &SM::I4d, "Fourth order deviatoric projection tensor.");

        // Tensor algebra

        sm.def(
            "trace",
            static_cast<double (*)(const xt::xtensor<double, 2>&)>(
                &SM::trace<xt::xtensor<double, 2>>),
            "Trace.",
            py::arg("A"));

        sm.def(
            "A2_ddot_B2",
            static_cast<double (*)(const xt::xtensor<double, 2>&, const xt::xtensor<double, 2>&)>(
                &SM::A2_ddot_B2<xt::xtensor<double, 2>, xt::xtensor<double, 2>>),
            "A : B.",
            py::arg("A"),
            py::arg("B"));

        sm.def(
            "A2s_ddot_B2s",
            static_cast<double (*)(const xt::xtensor<double, 2>&, const xt::xtensor<double, 2>&)>(
                &SM::A2s_ddot_B2s<xt::xtensor<double, 2>, xt::xtensor<double, 2>>),
            "A : B.",
            py::arg("A"),
            py::arg("B"));

        sm.def(
            "A2_dyadic_B2",
            static_cast<xt::xtensor<double, 4> (*)(
                    const xt::xtensor<double, 2>&,
                    const xt::xtensor<double, 2>&)>(
                &SM::A2_dyadic_B2<xt::xtensor<double, 2>, xt::xtensor<double, 2>>),
            "A * B.",
            py::arg("A"),
            py::arg("B"));

        sm.def(
            "A4_ddot_B2",
            static_cast<xt::xtensor<double, 2> (*)(
                    const xt::xtensor<double, 4>&,
                    const xt::xtensor<double, 2>&)>(
                &SM::A4_ddot_B2<xt::xtensor<double, 4>, xt::xtensor<double, 2>>),
            "A : B.",
            py::arg("A"),
            py::arg("B"));

        // Tensor algebra

        add_deviatoric_overloads_2d<xt::xtensor<double, 4>>(sm);
        add_deviatoric_overloads_2d<xt::xtensor<double, 3>>(sm);
        add_deviatoric_overloads_2d<xt::xtensor<double, 2>>(sm);
        add_hydrostatic_overloads_2d<xt::xtensor<double, 2>, xt::xtensor<double, 4>>(sm);
        add_hydrostatic_overloads_2d<xt::xtensor<double, 1>, xt::xtensor<double, 3>>(sm);
        add_hydrostatic_overloads_2d<xt::xtensor<double, 0>, xt::xtensor<double, 2>>(sm);
        add_norm_deviatoric_overloads_2d<xt::xtensor<double, 2>, xt::xtensor<double, 4>>(sm);
        add_norm_deviatoric_overloads_2d<xt::xtensor<double, 1>, xt::xtensor<double, 3>>(sm);
        add_norm_deviatoric_overloads_2d<xt::xtensor<double, 0>, xt::xtensor<double, 2>>(sm);

        // Array

        py::class_<SM::Array<1>> array1d(sm, "Array1d");
        py::class_<SM::Array<2>> array2d(sm, "Array2d");
        py::class_<SM::Array<3>> array3d(sm, "Array3d");

        construct_Array_2d<SM::Array<1>>(array1d);
        construct_Array_2d<SM::Array<2>>(array2d);
        construct_Array_2d<SM::Array<3>>(array3d);
    }

    // ----------------------
    // GMatTensor.Cartesian3d
    // ----------------------

    {
        py::module sm = m.def_submodule("Cartesian3d", "3d Cartesian coordinates");
        namespace SM = GMatTensor::Cartesian3d;

        // Unit tensors

        sm.def("O2", &SM::O2, "Second order null tensor.");
        sm.def("O4", &SM::O4, "Fourth order null tensor.");
        sm.def("I2", &SM::I2, "Second order unit tensor.");
        sm.def("II", &SM::II, "Fourth order tensor with the result of the dyadic product II.");
        sm.def("I4", &SM::I4, "Fourth order unit tensor.");
        sm.def("I4rt", &SM::I4rt, "Fourth right-transposed order unit tensor.");
        sm.def("I4s", &SM::I4s, "Fourth order symmetric projection tensor.");
        sm.def("I4d", &SM::I4d, "Fourth order deviatoric projection tensor.");

        // Tensor algebra

        sm.def(
            "trace",
            static_cast<double (*)(const xt::xtensor<double, 2>&)>(
                &SM::trace<xt::xtensor<double, 2>>),
            "Trace.",
            py::arg("A"));

        sm.def(
            "det",
            static_cast<double (*)(const xt::xtensor<double, 2>&)>(
                &SM::det<xt::xtensor<double, 2>>),
            "Determinant.",
            py::arg("A"));

        sm.def(
            "inv",
            static_cast<xt::xtensor<double, 2> (*)(const xt::xtensor<double, 2>&)>(
                &SM::inv<xt::xtensor<double, 2>>),
            "Inverse.",
            py::arg("A"));

        sm.def(
            "A2_ddot_B2",
            static_cast<double (*)(const xt::xtensor<double, 2>&, const xt::xtensor<double, 2>&)>(
                &SM::A2_ddot_B2<xt::xtensor<double, 2>, xt::xtensor<double, 2>>),
            "A : B.",
            py::arg("A"),
            py::arg("B"));

        sm.def(
            "A2s_ddot_B2s",
            static_cast<double (*)(const xt::xtensor<double, 2>&, const xt::xtensor<double, 2>&)>(
                &SM::A2s_ddot_B2s<xt::xtensor<double, 2>, xt::xtensor<double, 2>>),
            "A : B.",
            py::arg("A"),
            py::arg("B"));

        sm.def(
            "A2_dyadic_B2",
            static_cast<xt::xtensor<double, 4> (*)(
                    const xt::xtensor<double, 2>&,
                    const xt::xtensor<double, 2>&)>(
                &SM::A2_dyadic_B2<xt::xtensor<double, 2>, xt::xtensor<double, 2>>),
            "A * B.",
            py::arg("A"),
            py::arg("B"));

        sm.def(
            "A4_dot_B2",
            static_cast<xt::xtensor<double, 4> (*)(
                    const xt::xtensor<double, 4>&,
                    const xt::xtensor<double, 2>&)>(
                &SM::A4_dot_B2<xt::xtensor<double, 4>, xt::xtensor<double, 2>>),
            "A . B.",
            py::arg("A"),
            py::arg("B"));

        sm.def(
            "A2_dot_B2",
            static_cast<xt::xtensor<double, 2> (*)(
                    const xt::xtensor<double, 2>&,
                    const xt::xtensor<double, 2>&)>(
                &SM::A2_dot_B2<xt::xtensor<double, 2>, xt::xtensor<double, 2>>),
            "A . B.",
            py::arg("A"),
            py::arg("B"));

        sm.def(
            "A2_dot_A2T",
            static_cast<xt::xtensor<double, 2> (*)(const xt::xtensor<double, 2>&)>(
                &SM::A2_dot_A2T<xt::xtensor<double, 2>>),
            "A . A^T.",
            py::arg("A"));

        sm.def(
            "A4_ddot_B2",
            static_cast<xt::xtensor<double, 2> (*)(
                    const xt::xtensor<double, 4>&,
                    const xt::xtensor<double, 2>&)>(
                &SM::A4_ddot_B2<xt::xtensor<double, 4>, xt::xtensor<double, 2>>),
            "A : B.",
            py::arg("A"),
            py::arg("B"));

        // Tensor algebra

        add_deviatoric_overloads_3d<xt::xtensor<double, 4>>(sm);
        add_deviatoric_overloads_3d<xt::xtensor<double, 3>>(sm);
        add_deviatoric_overloads_3d<xt::xtensor<double, 2>>(sm);
        add_hydrostatic_overloads_3d<xt::xtensor<double, 2>, xt::xtensor<double, 4>>(sm);
        add_hydrostatic_overloads_3d<xt::xtensor<double, 1>, xt::xtensor<double, 3>>(sm);
        add_hydrostatic_overloads_3d<xt::xtensor<double, 0>, xt::xtensor<double, 2>>(sm);
        add_norm_deviatoric_overloads_3d<xt::xtensor<double, 2>, xt::xtensor<double, 4>>(sm);
        add_norm_deviatoric_overloads_3d<xt::xtensor<double, 1>, xt::xtensor<double, 3>>(sm);
        add_norm_deviatoric_overloads_3d<xt::xtensor<double, 0>, xt::xtensor<double, 2>>(sm);

        // Array

        py::class_<SM::Array<1>> array1d(sm, "Array1d");
        py::class_<SM::Array<2>> array2d(sm, "Array2d");
        py::class_<SM::Array<3>> array3d(sm, "Array3d");

        construct_Array_3d<SM::Array<1>>(array1d);
        construct_Array_3d<SM::Array<2>>(array2d);
        construct_Array_3d<SM::Array<3>>(array3d);
    }
}
