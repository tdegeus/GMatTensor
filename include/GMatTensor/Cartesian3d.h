/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatTensor

*/

#ifndef GMATTENSOR_CARTESIAN3D_H
#define GMATTENSOR_CARTESIAN3D_H

#include <xtensor/xtensor.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xview.hpp>

#include "config.h"

namespace GMatTensor {
namespace Cartesian3d {

// Alias

#if defined(_WIN32) || defined(_WIN64)
    using Tensor2 = xt::xtensor<double, 2>;
    using Tensor4 = xt::xtensor<double, 4>;
#else
    #include <xtensor/xfixed.hpp>
    using Tensor2 = xt::xtensor_fixed<double, xt::xshape<3, 3>>;
    using Tensor4 = xt::xtensor_fixed<double, xt::xshape<3, 3, 3, 3>>;
#endif

// Unit tensors

inline Tensor2 I2();
inline Tensor4 II();
inline Tensor4 I4();
inline Tensor4 I4rt();
inline Tensor4 I4s();
inline Tensor4 I4d();

// Tensor decomposition

template <class T, class U>
inline void hydrostatic(const T& A, U& B);

template <class T>
inline auto Hydrostatic(const T& A);

template <class T, class U>
inline void deviatoric(const T& A, U& B);

template <class T>
inline auto Deviatoric(const T& A);

// Equivalent value of the deviatoric part of the tensor

template <class T, class U, class V>
inline void equivalent_deviatoric(const T& A, U& B, V factor);

template <class T, class V>
inline auto Equivalent_deviatoric(const T& A, V factor);

// Array of tensors

template <size_t rank>
class Array
{
public:
    // Constructors

    Array() = default;
    Array(const std::array<size_t, rank>& shape);

    // Shape

    std::array<size_t, rank> shape() const;

    // Array of unit tensors

    xt::xtensor<double, rank + 2> I2() const;
    xt::xtensor<double, rank + 4> II() const;
    xt::xtensor<double, rank + 4> I4() const;
    xt::xtensor<double, rank + 4> I4rt() const;
    xt::xtensor<double, rank + 4> I4s() const;
    xt::xtensor<double, rank + 4> I4d() const;

protected:
    static const size_t m_ndim = 3;
    size_t m_size;
    std::array<size_t, rank> m_shape;
    std::array<size_t, rank + 2> m_shape_tensor2;
    std::array<size_t, rank + 4> m_shape_tensor4;
};

} // namespace Cartesian3d
} // namespace GMatTensor

#include "Cartesian3d.hpp"
#include "Cartesian3d_Array.hpp"

#endif
