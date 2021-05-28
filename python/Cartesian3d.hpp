/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatTensor

*/

#include <pybind11/pybind11.h>
#include <xtensor-python/pytensor.hpp>

#include <GMatTensor/Cartesian3d.h>

// Enable basic assertions on matrix shape
// (doesn't cost a lot of time, but avoids segmentation faults)
#define GMATTENSOR_ENABLE_ASSERT

namespace py = pybind11;

template <class T, class M>
auto construct_Array_3d(M& self)
{
    self.def(py::init<std::array<size_t, T::rank>>(), "Array.", py::arg("shape"))
        .def("shape", &T::shape, "Shape of array.")
        .def("I2", &T::I2, "Array with 2nd-order unit tensors.")
        .def("II", &T::II, "Array with 4th-order tensors = dyadic(I2, I2).")
        .def("I4", &T::I4, "Array with 4th-order unit tensors.")
        .def("I4rt", &T::I4rt, "Array with 4th-order right-transposed unit tensors.")
        .def("I4s", &T::I4s, "Array with 4th-order symmetric projection tensors.")
        .def("I4d", &T::I4d, "Array with 4th-order deviatoric projection tensors.")
        .def("__repr__", [](const T&) { return "<GMatTensor.Cartesian3d.Array>"; });
}

template <class R, class T, class M>
void add3d_Trace(M& module)
{
    module.def(
        "Trace",
        py::overload_cast<const T&>(&GMatTensor::Cartesian3d::Trace<T>),
        "Trace of a(n) (array of) 2nd-order tensor(s).",
        py::arg("A"));
}

template <class R, class T, class M>
void add3d_trace(M& module)
{
    module.def(
        "trace",
        py::overload_cast<const T&, R&>(&GMatTensor::Cartesian3d::trace<T, R>),
        "Trace of a(n) (array of) 2nd-order tensor(s).",
        py::arg("A"),
        py::arg("ret"));
}

template <class R, class T, class M>
void add3d_Hydrostatic(M& module)
{
    module.def(
        "Hydrostatic",
        py::overload_cast<const T&>(&GMatTensor::Cartesian3d::Hydrostatic<T>),
        "Hydrostatic part of a(n) (array of) 2nd-order tensor(s).",
        py::arg("A"));
}

template <class R, class T, class M>
void add3d_hydrostatic(M& module)
{
    module.def(
        "hydrostatic",
        py::overload_cast<const T&, R&>(&GMatTensor::Cartesian3d::hydrostatic<T, R>),
        "Hydrostatic part of a(n) (array of) 2nd-order tensor(s).",
        py::arg("A"),
        py::arg("ret"));
}

template <class R, class T, class M>
void add3d_Det(M& module)
{
    module.def(
        "Det",
        py::overload_cast<const T&>(&GMatTensor::Cartesian3d::Det<T>),
        "Determinant of a(n) (array of) 2nd-order tensor(s).",
        py::arg("A"));
}

template <class R, class T, class M>
void add3d_det(M& module)
{
    module.def(
        "det",
        py::overload_cast<const T&, R&>(&GMatTensor::Cartesian3d::det<T, R>),
        "Determinant of a(n) (array of) 2nd-order tensor(s).",
        py::arg("A"),
        py::arg("ret"));
}

template <class R, class T, class M>
void add3d_A2_ddot_B2(M& module)
{
    module.def(
        "A2_ddot_B2",
        py::overload_cast<const T&, const T&>(&GMatTensor::Cartesian3d::A2_ddot_B2<T>),
        "Product 'A : B' for two (arrays of) 2nd-order tensors.",
        py::arg("A"),
        py::arg("B"));
}

template <class R, class T, class M>
void add3d_A2_ddot_B2_ret(M& module)
{
    module.def(
        "A2_ddot_B2",
        py::overload_cast<const T&, const T&, R&>(&GMatTensor::Cartesian3d::A2_ddot_B2<T, R>),
        "Product 'A : B' for two (arrays of) 2nd-order tensors.",
        py::arg("A"),
        py::arg("B"),
        py::arg("ret"));
}

template <class R, class T, class M>
void add3d_A2s_ddot_B2s(M& module)
{
    module.def(
        "A2s_ddot_B2s",
        py::overload_cast<const T&, const T&>(&GMatTensor::Cartesian3d::A2s_ddot_B2s<T>),
        "Product 'A : B' for two (arrays of) symmetric 2nd-order tensors (no assertion).",
        py::arg("A"),
        py::arg("B"));
}

template <class R, class T, class M>
void add3d_A2s_ddot_B2s_ret(M& module)
{
    module.def(
        "A2s_ddot_B2s",
        py::overload_cast<const T&, const T&, R&>(&GMatTensor::Cartesian3d::A2s_ddot_B2s<T, R>),
        "Product 'A : B' for two (arrays of) symmetric 2nd-order tensors (no assertion).",
        py::arg("A"),
        py::arg("B"),
        py::arg("ret"));
}

template <class R, class T, class M>
void add3d_Norm_deviatoric(M& module)
{
    module.def(
        "Norm_deviatoric",
        py::overload_cast<const T&>(&GMatTensor::Cartesian3d::Norm_deviatoric<T>),
        "Norm of the deviatoric part of a(n) (array of) tensor(s).",
        py::arg("A"));
}

template <class R, class T, class M>
void add3d_norm_deviatoric(M& module)
{
    module.def(
        "norm_deviatoric",
        py::overload_cast<const T&, R&>(&GMatTensor::Cartesian3d::norm_deviatoric<T, R>),
        "Norm of the deviatoric part of a(n) (array of) tensor(s).",
        py::arg("A"),
        py::arg("ret"));
}

template <class R, class T, class M>
void add3d_Deviatoric(M& module)
{
    module.def(
        "Deviatoric",
        py::overload_cast<const T&>(&GMatTensor::Cartesian3d::Deviatoric<T>),
        "Deviatoric part of a (array of) tensor(s).",
        py::arg("A"));
}

template <class R, class T, class M>
void add3d_deviatoric(M& module)
{
    module.def(
        "deviatoric",
        py::overload_cast<const T&, R&>(&GMatTensor::Cartesian3d::deviatoric<T, R>),
        "Deviatoric part of a (array of) tensor(s).",
        py::arg("A"),
        py::arg("ret"));
}

template <class R, class T, class M>
void add3d_Sym(M& module)
{
    module.def(
        "Sym",
        py::overload_cast<const T&>(&GMatTensor::Cartesian3d::Sym<T>),
        "Symmetric part of a (array of) tensor(s).",
        py::arg("A"));
}

template <class R, class T, class M>
void add3d_sym(M& module)
{
    module.def(
        "sym",
        py::overload_cast<const T&, R&>(&GMatTensor::Cartesian3d::sym<T, R>),
        "Symmetric part of a (array of) tensor(s).",
        py::arg("A"),
        py::arg("ret"));
}

template <class R, class T, class M>
void add3d_Inv(M& module)
{
    module.def(
        "Inv",
        py::overload_cast<const T&>(&GMatTensor::Cartesian3d::Inv<T>),
        "Inverse of a (array of) tensor(s).",
        py::arg("A"));
}

template <class R, class T, class M>
void add3d_inv(M& module)
{
    module.def(
        "inv",
        py::overload_cast<const T&, R&>(&GMatTensor::Cartesian3d::inv<T, R>),
        "Inverse of a (array of) tensor(s).",
        py::arg("A"),
        py::arg("ret"));
}

template <class R, class T, class M>
void add3d_Logs(M& module)
{
    module.def(
        "Logs",
        py::overload_cast<const T&>(&GMatTensor::Cartesian3d::Logs<T>),
        "Log of a (array of) symmetric 2nd-order tensor(s) (no assertion).",
        py::arg("A"));
}

template <class R, class T, class M>
void add3d_logs(M& module)
{
    module.def(
        "logs",
        py::overload_cast<const T&, R&>(&GMatTensor::Cartesian3d::logs<T, R>),
        "Log of a (array of) symmetric 2nd-order tensor(s) (no assertion).",
        py::arg("A"),
        py::arg("ret"));
}

template <class R, class T, class M>
void add3d_A2_dot_A2T(M& module)
{
    module.def(
        "A2_dot_A2T",
        py::overload_cast<const T&>(&GMatTensor::Cartesian3d::A2_dot_A2T<T>),
        "Product 'A . A^T' of a (array of) tensor(s).",
        py::arg("A"));
}

template <class R, class T, class M>
void add3d_A2_dot_A2T_ret(M& module)
{
    module.def(
        "A2_dot_A2T",
        py::overload_cast<const T&, R&>(&GMatTensor::Cartesian3d::A2_dot_A2T<T, R>),
        "Product 'A . A^T' of a (array of) tensor(s).",
        py::arg("A"),
        py::arg("ret"));
}

template <class R, class T, class M>
void add3d_A2_dot_B2(M& module)
{
    module.def(
        "A2_dot_B2",
        py::overload_cast<const T&, const T&>(&GMatTensor::Cartesian3d::A2_dot_B2<T>),
        "Product 'A . B' of a (array of) tensor(s).",
        py::arg("A"),
        py::arg("B"));
}

template <class R, class T, class M>
void add3d_A2_dot_B2_ret(M& module)
{
    module.def(
        "A2_dot_B2",
        py::overload_cast<const T&, const T&, R&>(&GMatTensor::Cartesian3d::A2_dot_B2<T, R>),
        "Product 'A . B' of a (array of) tensor(s).",
        py::arg("A"),
        py::arg("B"),
        py::arg("ret"));
}

template <class R, class T, class M>
void add3d_A2_dyadic_B2(M& module)
{
    module.def(
        "A2_dyadic_B2",
        py::overload_cast<const T&, const T&>(&GMatTensor::Cartesian3d::A2_dyadic_B2<T>),
        "Product 'A * B' for two (arrays of) 2nd-order tensors.",
        py::arg("A"),
        py::arg("B"));
}

template <class R, class T, class M>
void add3d_A2_dyadic_B2_ret(M& module)
{
    module.def(
        "A2_dyadic_B2",
        py::overload_cast<const T&, const T&, R&>(&GMatTensor::Cartesian3d::A2_dyadic_B2<T, R>),
        "Product 'A * B' for two (arrays of) 2nd-order tensors.",
        py::arg("A"),
        py::arg("B"),
        py::arg("ret"));
}

template <class R, class T, class U, class M>
void add3d_A4_ddot_B2(M& module)
{
    module.def(
        "A4_ddot_B2",
        py::overload_cast<const T&, const U&>(&GMatTensor::Cartesian3d::A4_ddot_B2<T, U>),
        "Product 'A : B' for two (arrays of) 4th and 2nd-order tensors.",
        py::arg("A"),
        py::arg("B"));
}

template <class R, class T, class U, class M>
void add3d_A4_ddot_B2_ret(M& module)
{
    module.def(
        "A4_ddot_B2",
        py::overload_cast<const T&, const U&, R&>(&GMatTensor::Cartesian3d::A4_ddot_B2<T, U, R>),
        "Product 'A : B' for two (arrays of) 4th and 2nd-order tensors.",
        py::arg("A"),
        py::arg("B"),
        py::arg("ret"));
}

template <class R, class T, class U, class M>
void add3d_A4_dot_B2(M& module)
{
    module.def(
        "A4_dot_B2",
        py::overload_cast<const T&, const U&>(&GMatTensor::Cartesian3d::A4_dot_B2<T, U>),
        "Product 'A . B' for two (arrays of) 4th and 2nd-order tensors.",
        py::arg("A"),
        py::arg("B"));
}

template <class R, class T, class U, class M>
void add3d_A4_dot_B2_ret(M& module)
{
    module.def(
        "A4_dot_B2",
        py::overload_cast<const T&, const U&, R&>(&GMatTensor::Cartesian3d::A4_dot_B2<T, U, R>),
        "Product 'A . B' for two (arrays of) 4th and 2nd-order tensors.",
        py::arg("A"),
        py::arg("B"),
        py::arg("ret"));
}

void init_Cartesian3d(py::module& m)
{
    namespace M = GMatTensor::Cartesian3d;

    // Unit tensors

    m.def("O2", &M::O2, "Second order null tensor.");
    m.def("O4", &M::O4, "Fourth order null tensor.");
    m.def("I2", &M::I2, "Second order unit tensor.");
    m.def("II", &M::II, "Fourth order tensor with the result of the dyadic product II.");
    m.def("I4", &M::I4, "Fourth order unit tensor.");
    m.def("I4rt", &M::I4rt, "Fourth right-transposed order unit tensor.");
    m.def("I4s", &M::I4s, "Fourth order symmetric projection tensor.");
    m.def("I4d", &M::I4d, "Fourth order deviatoric projection tensor.");

    // Tensor algebra

    add3d_Trace<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(m);
    add3d_Trace<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(m);
    add3d_Trace<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(m);

    add3d_trace<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(m);
    add3d_trace<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(m);
    add3d_trace<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(m);

    add3d_Hydrostatic<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(m);
    add3d_Hydrostatic<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(m);
    add3d_Hydrostatic<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(m);

    add3d_hydrostatic<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(m);
    add3d_hydrostatic<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(m);
    add3d_hydrostatic<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(m);

    add3d_Det<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(m);
    add3d_Det<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(m);
    add3d_Det<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(m);

    add3d_det<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(m);
    add3d_det<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(m);
    add3d_det<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(m);

    add3d_A2_ddot_B2<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(m);
    add3d_A2_ddot_B2<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(m);
    add3d_A2_ddot_B2<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(m);

    add3d_A2_ddot_B2_ret<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(m);
    add3d_A2_ddot_B2_ret<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(m);
    add3d_A2_ddot_B2_ret<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(m);

    add3d_A2s_ddot_B2s<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(m);
    add3d_A2s_ddot_B2s<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(m);
    add3d_A2s_ddot_B2s<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(m);

    add3d_A2s_ddot_B2s_ret<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(m);
    add3d_A2s_ddot_B2s_ret<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(m);
    add3d_A2s_ddot_B2s_ret<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(m);

    add3d_Norm_deviatoric<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(m);
    add3d_Norm_deviatoric<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(m);
    add3d_Norm_deviatoric<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(m);

    add3d_norm_deviatoric<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(m);
    add3d_norm_deviatoric<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(m);
    add3d_norm_deviatoric<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(m);

    add3d_Deviatoric<xt::pytensor<double, 4>, xt::pytensor<double, 4>>(m);
    add3d_Deviatoric<xt::pytensor<double, 3>, xt::pytensor<double, 3>>(m);
    add3d_Deviatoric<xt::pytensor<double, 2>, xt::pytensor<double, 2>>(m);

    add3d_deviatoric<xt::pytensor<double, 4>, xt::pytensor<double, 4>>(m);
    add3d_deviatoric<xt::pytensor<double, 3>, xt::pytensor<double, 3>>(m);
    add3d_deviatoric<xt::pytensor<double, 2>, xt::pytensor<double, 2>>(m);

    add3d_Sym<xt::pytensor<double, 4>, xt::pytensor<double, 4>>(m);
    add3d_Sym<xt::pytensor<double, 3>, xt::pytensor<double, 3>>(m);
    add3d_Sym<xt::pytensor<double, 2>, xt::pytensor<double, 2>>(m);

    add3d_sym<xt::pytensor<double, 4>, xt::pytensor<double, 4>>(m);
    add3d_sym<xt::pytensor<double, 3>, xt::pytensor<double, 3>>(m);
    add3d_sym<xt::pytensor<double, 2>, xt::pytensor<double, 2>>(m);

    add3d_Inv<xt::pytensor<double, 4>, xt::pytensor<double, 4>>(m);
    add3d_Inv<xt::pytensor<double, 3>, xt::pytensor<double, 3>>(m);
    add3d_Inv<xt::pytensor<double, 2>, xt::pytensor<double, 2>>(m);

    add3d_inv<xt::pytensor<double, 4>, xt::pytensor<double, 4>>(m);
    add3d_inv<xt::pytensor<double, 3>, xt::pytensor<double, 3>>(m);
    add3d_inv<xt::pytensor<double, 2>, xt::pytensor<double, 2>>(m);

    add3d_Logs<xt::pytensor<double, 4>, xt::pytensor<double, 4>>(m);
    add3d_Logs<xt::pytensor<double, 3>, xt::pytensor<double, 3>>(m);
    add3d_Logs<xt::pytensor<double, 2>, xt::pytensor<double, 2>>(m);

    add3d_logs<xt::pytensor<double, 4>, xt::pytensor<double, 4>>(m);
    add3d_logs<xt::pytensor<double, 3>, xt::pytensor<double, 3>>(m);
    add3d_logs<xt::pytensor<double, 2>, xt::pytensor<double, 2>>(m);

    add3d_A2_dot_A2T<xt::pytensor<double, 4>, xt::pytensor<double, 4>>(m);
    add3d_A2_dot_A2T<xt::pytensor<double, 3>, xt::pytensor<double, 3>>(m);
    add3d_A2_dot_A2T<xt::pytensor<double, 2>, xt::pytensor<double, 2>>(m);

    add3d_A2_dot_A2T_ret<xt::pytensor<double, 4>, xt::pytensor<double, 4>>(m);
    add3d_A2_dot_A2T_ret<xt::pytensor<double, 3>, xt::pytensor<double, 3>>(m);
    add3d_A2_dot_A2T_ret<xt::pytensor<double, 2>, xt::pytensor<double, 2>>(m);

    add3d_A2_dot_B2<xt::pytensor<double, 4>, xt::pytensor<double, 4>>(m);
    add3d_A2_dot_B2<xt::pytensor<double, 3>, xt::pytensor<double, 3>>(m);
    add3d_A2_dot_B2<xt::pytensor<double, 2>, xt::pytensor<double, 2>>(m);

    add3d_A2_dot_B2_ret<xt::pytensor<double, 4>, xt::pytensor<double, 4>>(m);
    add3d_A2_dot_B2_ret<xt::pytensor<double, 3>, xt::pytensor<double, 3>>(m);
    add3d_A2_dot_B2_ret<xt::pytensor<double, 2>, xt::pytensor<double, 2>>(m);

    add3d_A2_dyadic_B2<xt::pytensor<double, 6>, xt::pytensor<double, 4>>(m);
    add3d_A2_dyadic_B2<xt::pytensor<double, 5>, xt::pytensor<double, 3>>(m);
    add3d_A2_dyadic_B2<xt::pytensor<double, 4>, xt::pytensor<double, 2>>(m);

    add3d_A2_dyadic_B2_ret<xt::pytensor<double, 6>, xt::pytensor<double, 4>>(m);
    add3d_A2_dyadic_B2_ret<xt::pytensor<double, 5>, xt::pytensor<double, 3>>(m);
    add3d_A2_dyadic_B2_ret<xt::pytensor<double, 4>, xt::pytensor<double, 2>>(m);

    add3d_A4_ddot_B2<xt::pytensor<double, 4>, xt::pytensor<double, 6>, xt::pytensor<double, 4>>(m);
    add3d_A4_ddot_B2<xt::pytensor<double, 3>, xt::pytensor<double, 5>, xt::pytensor<double, 3>>(m);
    add3d_A4_ddot_B2<xt::pytensor<double, 2>, xt::pytensor<double, 4>, xt::pytensor<double, 2>>(m);

    add3d_A4_ddot_B2_ret<xt::pytensor<double, 4>, xt::pytensor<double, 6>, xt::pytensor<double, 4>>(m);
    add3d_A4_ddot_B2_ret<xt::pytensor<double, 3>, xt::pytensor<double, 5>, xt::pytensor<double, 3>>(m);
    add3d_A4_ddot_B2_ret<xt::pytensor<double, 2>, xt::pytensor<double, 4>, xt::pytensor<double, 2>>(m);

    add3d_A4_dot_B2<xt::pytensor<double, 6>, xt::pytensor<double, 6>, xt::pytensor<double, 4>>(m);
    add3d_A4_dot_B2<xt::pytensor<double, 5>, xt::pytensor<double, 5>, xt::pytensor<double, 3>>(m);
    add3d_A4_dot_B2<xt::pytensor<double, 4>, xt::pytensor<double, 4>, xt::pytensor<double, 2>>(m);

    add3d_A4_dot_B2_ret<xt::pytensor<double, 6>, xt::pytensor<double, 6>, xt::pytensor<double, 4>>(m);
    add3d_A4_dot_B2_ret<xt::pytensor<double, 5>, xt::pytensor<double, 5>, xt::pytensor<double, 3>>(m);
    add3d_A4_dot_B2_ret<xt::pytensor<double, 4>, xt::pytensor<double, 4>, xt::pytensor<double, 2>>(m);

    // Array

    py::class_<M::Array<1>> array1d(m, "Array1d");
    py::class_<M::Array<2>> array2d(m, "Array2d");
    py::class_<M::Array<3>> array3d(m, "Array3d");

    construct_Array_3d<M::Array<1>>(array1d);
    construct_Array_3d<M::Array<2>>(array2d);
    construct_Array_3d<M::Array<3>>(array3d);
}
