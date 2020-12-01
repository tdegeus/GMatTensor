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

// Unit tensors
inline xt::xtensor<double, 2> I2();
inline xt::xtensor<double, 4> II();
inline xt::xtensor<double, 4> I4();
inline xt::xtensor<double, 4> I4rt();
inline xt::xtensor<double, 4> I4s();
inline xt::xtensor<double, 4> I4d();

// Hydrostatic part of a tensor (== trace(A) / 3)
template <class T, class U>
inline void hydrostatic(const T& A, U& B);

template <class T>
inline auto Hydrostatic(const T& A);

// Deviatoric part of a tensor (== A - Hydrostatic(A) * I2)
template <class T, class U>
inline void deviatoric(const T& A, U& B);

template <class T>
inline auto Deviatoric(const T& A);

// Equivalent value of the tensor's deviator: (dev(A))_ij (dev(A))_ji
template <class T, class U>
inline void equivalent_deviatoric(const T& A, U& B);

template <class T>
inline auto Equivalent_deviatoric(const T& A);

// Array of tensors: shape (..., d, d), e.g. (R, S, T, d, d)
template <size_t N>
class Array
{
public:
    constexpr static std::size_t rank = N;

    // Constructors
    Array() = default;
    Array(const std::array<size_t, N>& shape);

    // Shape of the array (excluding the two additional tensor ranks), e,g. (R, S, T)
    std::array<size_t, N> shape() const;

    // Array of unit tensors, shape (..., d, d), e.g. (R, S, T, d, d)
    xt::xtensor<double, N + 2> I2() const;
    xt::xtensor<double, N + 4> II() const;
    xt::xtensor<double, N + 4> I4() const;
    xt::xtensor<double, N + 4> I4rt() const;
    xt::xtensor<double, N + 4> I4s() const;
    xt::xtensor<double, N + 4> I4d() const;

protected:
    void init(const std::array<size_t, N>& shape);

    static constexpr size_t m_ndim = 3;
    static constexpr size_t m_stride_tensor2 = 9;
    static constexpr size_t m_stride_tensor4 = 81;
    size_t m_size;
    std::array<size_t, N> m_shape;
    std::array<size_t, N + 2> m_shape_tensor2;
    std::array<size_t, N + 4> m_shape_tensor4;
};

// API for pure-tensor with pointer-only input
// Storage convention:
// - 2nd-order tensor: (xx, xy, xz, yx, yy, yz, zx, zy, zz)
namespace pointer {

    template <class T>
    inline void I2(T ret);

    template <class T>
    inline void II(T ret);

    template <class T>
    inline void I4(T ret);

    template <class T>
    inline void I4rt(T ret);

    template <class T>
    inline void I4s(T ret);

    template <class T>
    inline void I4d(T ret);

    template <class T>
    inline auto trace(const T A);

    template <class T, class U>
    inline auto hydrostatic_deviatoric(const T A, U ret);

    template <class T>
    inline auto deviatoric_ddot_deviatoric(const T A);

    template <class T, class U>
    inline auto A2_ddot_B2(const T A, const U B);

    template <class U, class V, class W>
    inline void A2_dyadic_B2(const U A, const V B, W C);

} // namespace pointer

} // namespace Cartesian3d
} // namespace GMatTensor

#include "Cartesian3d.hpp"
#include "Cartesian3d_Array.hpp"

#endif
