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

// Hydrostatic part of a tensor (== trace(A) / 2)
template <class T, class U>
inline void hydrostatic(const T& A, U& ret);

template <class T>
inline auto Hydrostatic(const T& A);

// Deviatoric part of a tensor (== A - Hydrostatic(A) * I2)
template <class T, class U>
inline void deviatoric(const T& A, U& ret);

template <class T>
inline auto Deviatoric(const T& A);

// Equivalent value of the tensor's deviator: (dev(A))_ij (dev(A))_ji
template <class T, class U>
inline void equivalent_deviatoric(const T& A, U& ret);

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

    static constexpr size_t m_ndim = 2;
    static constexpr size_t m_stride_tensor2 = 4;
    static constexpr size_t m_stride_tensor4 = 16;
    size_t m_size;
    std::array<size_t, N> m_shape;
    std::array<size_t, N + 2> m_shape_tensor2;
    std::array<size_t, N + 4> m_shape_tensor4;
};

// API for pure-tensor with pointer-only input
// Storage convention:
// - Second order tensor: (xx, xy, yx, yy)
namespace pointer {

    // Zero second order tensor
    template <class T>
    inline void O2(T* ret);

    // Zero fourth order tensor
    template <class T>
    inline void O4(T* ret);

    // Second order unit tensor
    template <class T>
    inline void I2(T* ret);

    // Dyadic product I2 * I2
    template <class T>
    inline void II(T* ret);

    // Fourth order unite tensor
    template <class T>
    inline void I4(T* ret);

    // Fourth order right-transposed tensor
    template <class T>
    inline void I4rt(T* ret);

    // Symmetric projection
    template <class T>
    inline void I4s(T* ret);

    // Deviatoric projection
    template <class T>
    inline void I4d(T* ret);

    // Trace of second order tensor
    template <class T>
    inline auto trace(const T* A);

    // Hydrostatic and deviatoric decomposition of second order tensor
    template <class S, class T>
    inline auto hydrostatic_deviatoric(const S* A, T* ret);

    // dev(A) : dev(A)
    template <class T>
    inline auto deviatoric_ddot_deviatoric(const T* A);

    // A : B
    template <class S, class T>
    inline auto A2_ddot_B2(const S* A, const T* B);

    // A : B (both assumed symmetric)
    template <class S, class T>
    inline auto A2s_ddot_B2s(const S* A, const T* B);

    // A * B
    template <class R, class S, class T>
    inline void A2_dyadic_B2(const R* A, const S* B, T* C);

} // namespace pointer

} // namespace Cartesian2d
} // namespace GMatTensor

#include "Cartesian2d.hpp"
#include "Cartesian2d_Array.hpp"

#endif
