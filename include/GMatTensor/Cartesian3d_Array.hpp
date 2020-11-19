/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatTensor

*/

#ifndef GMATTENSOR_CARTESIAN3D_ARRAY_HPP
#define GMATTENSOR_CARTESIAN3D_ARRAY_HPP

#include "Cartesian3d.h"

namespace GMatTensor {
namespace Cartesian3d {

template <size_t rank>
inline Array<rank>::Array(const std::array<size_t, rank>& shape) : m_shape(shape)
{
    size_t nd = m_ndim;
    std::copy(shape.begin(), shape.end(), m_shape_tensor2.begin());
    std::copy(shape.begin(), shape.end(), m_shape_tensor4.begin());
    std::fill(m_shape_tensor2.begin() + rank, m_shape_tensor2.end(), nd);
    std::fill(m_shape_tensor4.begin() + rank, m_shape_tensor4.end(), nd);
    m_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
}

template <size_t rank>
inline std::array<size_t, rank> Array<rank>::shape() const
{
    return m_shape;
}

template <size_t rank>
inline xt::xtensor<double, rank + 2> Array<rank>::I2() const
{
    xt::xtensor<double, rank + 2> ret = xt::empty<double>(m_shape_tensor2);

    #pragma omp parallel
    {
        Tensor2 unit = Cartesian3d::I2();
        size_t stride = m_ndim * m_ndim;

        #pragma omp for
        for (size_t i = 0; i < m_size; ++i) {
            auto view = xt::adapt(&ret.data()[i * stride], xt::xshape<m_ndim, m_ndim>());
            xt::noalias(view) = unit;
        }
    }

    return ret;
}

template <size_t rank>
inline xt::xtensor<double, rank + 4> Array<rank>::II() const
{
    xt::xtensor<double, rank + 4> ret = xt::empty<double>(m_shape_tensor4);

    #pragma omp parallel
    {
        Tensor4 unit = Cartesian3d::II();
        size_t stride = m_ndim * m_ndim * m_ndim * m_ndim;

        #pragma omp for
        for (size_t i = 0; i < m_size; ++i) {
            auto view = xt::adapt(&ret.data()[i * stride], xt::xshape<m_ndim, m_ndim, m_ndim, m_ndim>());
            xt::noalias(view) = unit;
        }
    }

    return ret;
}

template <size_t rank>
inline xt::xtensor<double, rank + 4> Array<rank>::I4() const
{
    xt::xtensor<double, rank + 4> ret = xt::empty<double>(m_shape_tensor4);

    #pragma omp parallel
    {
        Tensor4 unit = Cartesian3d::I4();
        size_t stride = m_ndim * m_ndim * m_ndim * m_ndim;

        #pragma omp for
        for (size_t i = 0; i < m_size; ++i) {
            auto view = xt::adapt(&ret.data()[i * stride], xt::xshape<m_ndim, m_ndim, m_ndim, m_ndim>());
            xt::noalias(view) = unit;
        }
    }

    return ret;
}

template <size_t rank>
inline xt::xtensor<double, rank + 4> Array<rank>::I4rt() const
{
    xt::xtensor<double, rank + 4> ret = xt::empty<double>(m_shape_tensor4);

    #pragma omp parallel
    {
        Tensor4 unit = Cartesian3d::I4rt();
        size_t stride = m_ndim * m_ndim * m_ndim * m_ndim;

        #pragma omp for
        for (size_t i = 0; i < m_size; ++i) {
            auto view = xt::adapt(&ret.data()[i * stride], xt::xshape<m_ndim, m_ndim, m_ndim, m_ndim>());
            xt::noalias(view) = unit;
        }
    }

    return ret;
}

template <size_t rank>
inline xt::xtensor<double, rank + 4> Array<rank>::I4s() const
{
    xt::xtensor<double, rank + 4> ret = xt::empty<double>(m_shape_tensor4);

    #pragma omp parallel
    {
        Tensor4 unit = Cartesian3d::I4s();
        size_t stride = m_ndim * m_ndim * m_ndim * m_ndim;

        #pragma omp for
        for (size_t i = 0; i < m_size; ++i) {
            auto view = xt::adapt(&ret.data()[i * stride], xt::xshape<m_ndim, m_ndim, m_ndim, m_ndim>());
            xt::noalias(view) = unit;
        }
    }

    return ret;
}

template <size_t rank>
inline xt::xtensor<double, rank + 4> Array<rank>::I4d() const
{
    xt::xtensor<double, rank + 4> ret = xt::empty<double>(m_shape_tensor4);

    #pragma omp parallel
    {
        Tensor4 unit = Cartesian3d::I4d();
        size_t stride = m_ndim * m_ndim * m_ndim * m_ndim;

        #pragma omp for
        for (size_t i = 0; i < m_size; ++i) {
            auto view = xt::adapt(&ret.data()[i * stride], xt::xshape<m_ndim, m_ndim, m_ndim, m_ndim>());
            xt::noalias(view) = unit;
        }
    }

    return ret;
}

} // namespace Cartesian3d
} // namespace GMatTensor

#endif
