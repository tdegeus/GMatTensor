/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatTensor

*/

#ifndef GMATTENSOR_CARTESIAN2D_H
#define GMATTENSOR_CARTESIAN2D_H

#include <xtensor/xtensor.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xview.hpp>

#include "config.h"

namespace GMatTensor {
namespace Cartesian2d {

// Unit tensors
inline xt::xtensor<double, 2> I2();
inline xt::xtensor<double, 4> II();
inline xt::xtensor<double, 4> I4();
inline xt::xtensor<double, 4> I4rt();
inline xt::xtensor<double, 4> I4s();
inline xt::xtensor<double, 4> I4d();

// Tensor decomposition
template <class T, class U>
inline void hydrostatic(const T& A, U& B);

template <class T>
inline auto Hydrostatic(const T& A);

template <class T, class U>
inline void deviatoric(const T& A, U& B);

template <class T>
inline auto Deviatoric(const T& A);

// Equivalent value of the tensor's deviator
template <class T, class U>
inline void equivalent_deviatoric(const T& A, U& B);

template <class T>
inline auto Equivalent_deviatoric(const T& A);

// Array of tensors
template <size_t N>
class Array
{
public:
    constexpr static std::size_t rank = N;

    // Constructors
    Array() = default;
    Array(const std::array<size_t, N>& shape);

    // Shape
    std::array<size_t, N> shape() const;

    // Array of unit tensors
    xt::xtensor<double, N + 2> I2() const;
    xt::xtensor<double, N + 4> II() const;
    xt::xtensor<double, N + 4> I4() const;
    xt::xtensor<double, N + 4> I4rt() const;
    xt::xtensor<double, N + 4> I4s() const;
    xt::xtensor<double, N + 4> I4d() const;

protected:
    void init();

    static const size_t m_ndim = 2;
    size_t m_size;
    std::array<size_t, N> m_shape;
    std::array<size_t, N + 2> m_shape_tensor2;
    std::array<size_t, N + 4> m_shape_tensor4;
};

} // namespace Cartesian2d
} // namespace GMatTensor

#include "Cartesian2d.hpp"
#include "Cartesian2d_Array.hpp"

#endif
