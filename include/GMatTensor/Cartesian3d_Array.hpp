/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatTensor

*/

#ifndef GMATTENSOR_CARTESIAN3D_ARRAY_HPP
#define GMATTENSOR_CARTESIAN3D_ARRAY_HPP

#include "Cartesian3d.h"

namespace GMatTensor {
namespace Cartesian3d {

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
inline xt::xtensor<double, N + 2> Array<N>::I2() const
{
    xt::xtensor<double, N + 2> ret = xt::empty<double>(m_shape_tensor2);

    #pragma omp parallel
    {
        auto unit = Cartesian3d::I2();
        auto shape = xt::xshape<m_ndim, m_ndim>();
        size_t stride = m_stride_tensor2;

        #pragma omp for
        for (size_t i = 0; i < m_size; ++i) {
            auto view = xt::adapt(&ret.data()[i * stride], shape);
            xt::noalias(view) = unit;
        }
    }

    return ret;
}

template <size_t N>
inline xt::xtensor<double, N + 4> Array<N>::II() const
{
    xt::xtensor<double, N + 4> ret = xt::empty<double>(m_shape_tensor4);

    #pragma omp parallel
    {
        auto unit = Cartesian3d::II();
        auto shape = xt::xshape<m_ndim, m_ndim, m_ndim, m_ndim>();
        size_t stride = m_stride_tensor4;

        #pragma omp for
        for (size_t i = 0; i < m_size; ++i) {
            auto view = xt::adapt(&ret.data()[i * stride], shape);
            xt::noalias(view) = unit;
        }
    }

    return ret;
}

template <size_t N>
inline xt::xtensor<double, N + 4> Array<N>::I4() const
{
    xt::xtensor<double, N + 4> ret = xt::empty<double>(m_shape_tensor4);

    #pragma omp parallel
    {
        auto unit = Cartesian3d::I4();
        auto shape = xt::xshape<m_ndim, m_ndim, m_ndim, m_ndim>();
        size_t stride = m_stride_tensor4;

        #pragma omp for
        for (size_t i = 0; i < m_size; ++i) {
            auto view = xt::adapt(&ret.data()[i * stride], shape);
            xt::noalias(view) = unit;
        }
    }

    return ret;
}

template <size_t N>
inline xt::xtensor<double, N + 4> Array<N>::I4rt() const
{
    xt::xtensor<double, N + 4> ret = xt::empty<double>(m_shape_tensor4);

    #pragma omp parallel
    {
        auto unit = Cartesian3d::I4rt();
        auto shape = xt::xshape<m_ndim, m_ndim, m_ndim, m_ndim>();
        size_t stride = m_stride_tensor4;

        #pragma omp for
        for (size_t i = 0; i < m_size; ++i) {
            auto view = xt::adapt(&ret.data()[i * stride], shape);
            xt::noalias(view) = unit;
        }
    }

    return ret;
}

template <size_t N>
inline xt::xtensor<double, N + 4> Array<N>::I4s() const
{
    xt::xtensor<double, N + 4> ret = xt::empty<double>(m_shape_tensor4);

    #pragma omp parallel
    {
        auto unit = Cartesian3d::I4s();
        auto shape = xt::xshape<m_ndim, m_ndim, m_ndim, m_ndim>();
        size_t stride = m_stride_tensor4;

        #pragma omp for
        for (size_t i = 0; i < m_size; ++i) {
            auto view = xt::adapt(&ret.data()[i * stride], shape);
            xt::noalias(view) = unit;
        }
    }

    return ret;
}

template <size_t N>
inline xt::xtensor<double, N + 4> Array<N>::I4d() const
{
    xt::xtensor<double, N + 4> ret = xt::empty<double>(m_shape_tensor4);

    #pragma omp parallel
    {
        auto unit = Cartesian3d::I4d();
        auto shape = xt::xshape<m_ndim, m_ndim, m_ndim, m_ndim>();
        size_t stride = m_stride_tensor4;

        #pragma omp for
        for (size_t i = 0; i < m_size; ++i) {
            auto view = xt::adapt(&ret.data()[i * stride], shape);
            xt::noalias(view) = unit;
        }
    }

    return ret;
}

} // namespace Cartesian3d
} // namespace GMatTensor

#endif
