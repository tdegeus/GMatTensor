/**
Implementation of Cartesian2d.h

\file Cartesian3d_Array.hpp
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the MIT License.
*/

#ifndef GMATTENSOR_CARTESIAN2D_ARRAY_HPP
#define GMATTENSOR_CARTESIAN2D_ARRAY_HPP

#include "Cartesian2d.h"

namespace GMatTensor {
namespace Cartesian2d {

template <size_t N>
inline Array<N>::Array(const std::array<size_t, N>& shape)
{
    this->init(shape);
}

template <size_t N>
inline void Array<N>::init(const std::array<size_t, N>& shape)
{
    m_shape = shape;
    size_t nd = m_ndim;
    std::copy(m_shape.begin(), m_shape.end(), m_shape_tensor2.begin());
    std::copy(m_shape.begin(), m_shape.end(), m_shape_tensor4.begin());
    std::fill(m_shape_tensor2.begin() + N, m_shape_tensor2.end(), nd);
    std::fill(m_shape_tensor4.begin() + N, m_shape_tensor4.end(), nd);
    m_size = std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<size_t>());
}

template <size_t N>
inline std::array<size_t, N> Array<N>::shape() const
{
    return m_shape;
}

template <size_t N>
inline xt::xtensor<double, N + 2> Array<N>::O2() const
{
    xt::xtensor<double, N + 2> ret = xt::empty<double>(m_shape_tensor2);

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        Cartesian2d::pointer::O2(&ret.data()[i * m_stride_tensor2]);
    }

    return ret;
}

template <size_t N>
inline xt::xtensor<double, N + 4> Array<N>::O4() const
{
    xt::xtensor<double, N + 4> ret = xt::empty<double>(m_shape_tensor4);

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        Cartesian2d::pointer::O4(&ret.data()[i * m_stride_tensor4]);
    }

    return ret;
}

template <size_t N>
inline xt::xtensor<double, N + 2> Array<N>::I2() const
{
    xt::xtensor<double, N + 2> ret = xt::empty<double>(m_shape_tensor2);

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        Cartesian2d::pointer::I2(&ret.data()[i * m_stride_tensor2]);
    }

    return ret;
}

template <size_t N>
inline xt::xtensor<double, N + 4> Array<N>::II() const
{
    xt::xtensor<double, N + 4> ret = xt::empty<double>(m_shape_tensor4);

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        Cartesian2d::pointer::II(&ret.data()[i * m_stride_tensor4]);
    }

    return ret;
}

template <size_t N>
inline xt::xtensor<double, N + 4> Array<N>::I4() const
{
    xt::xtensor<double, N + 4> ret = xt::empty<double>(m_shape_tensor4);

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        Cartesian2d::pointer::I4(&ret.data()[i * m_stride_tensor4]);
    }

    return ret;
}

template <size_t N>
inline xt::xtensor<double, N + 4> Array<N>::I4rt() const
{
    xt::xtensor<double, N + 4> ret = xt::empty<double>(m_shape_tensor4);

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        Cartesian2d::pointer::I4rt(&ret.data()[i * m_stride_tensor4]);
    }

    return ret;
}

template <size_t N>
inline xt::xtensor<double, N + 4> Array<N>::I4s() const
{
    xt::xtensor<double, N + 4> ret = xt::empty<double>(m_shape_tensor4);

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        Cartesian2d::pointer::I4s(&ret.data()[i * m_stride_tensor4]);
    }

    return ret;
}

template <size_t N>
inline xt::xtensor<double, N + 4> Array<N>::I4d() const
{
    xt::xtensor<double, N + 4> ret = xt::empty<double>(m_shape_tensor4);

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        Cartesian2d::pointer::I4d(&ret.data()[i * m_stride_tensor4]);
    }

    return ret;
}

} // namespace Cartesian2d
} // namespace GMatTensor

#endif
