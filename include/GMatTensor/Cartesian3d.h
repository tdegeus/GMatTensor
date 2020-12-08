/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatTensor

*/

#ifndef GMATTENSOR_CARTESIAN3D_H
#define GMATTENSOR_CARTESIAN3D_H

#include <xtensor/xtensor.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>

#include "config.h"

namespace GMatTensor {
namespace Cartesian3d {

// Unit tensors
inline xt::xtensor<double, 2> Random2(); // random tensor
inline xt::xtensor<double, 4> Random4(); // random tensor
inline xt::xtensor<double, 2> O2(); // null tensor
inline xt::xtensor<double, 4> O4(); // null tensor
inline xt::xtensor<double, 2> I2();
inline xt::xtensor<double, 4> II();
inline xt::xtensor<double, 4> I4();
inline xt::xtensor<double, 4> I4rt();
inline xt::xtensor<double, 4> I4s();
inline xt::xtensor<double, 4> I4d();

// Trace
template <class T, class R>
inline void trace(const T& A, R& ret);

template <class T>
inline auto Trace(const T& A);

// Hydrostatic part of a tensor (== trace(A) / 3)
template <class T, class R>
inline void hydrostatic(const T& A, R& ret);

template <class T>
inline auto Hydrostatic(const T& A);

// Determinant
template <class T, class R>
inline void det(const T& A, R& ret);

template <class T>
inline auto Det(const T& A);

// A : B
template <class T, class R>
inline void A2_ddot_B2(const T& A, const T& B, R& ret);

template <class T>
inline auto A2_ddot_B2(const T& A, const T& B);

// A : B
template <class T, class R>
inline void A2s_ddot_B2s(const T& A, const T& B, R& ret);

template <class T>
inline auto A2s_ddot_B2s(const T& A, const T& B);

// Norm of the tensor's deviator: sqrt((dev(A))_ij (dev(A))_ji)
template <class T, class R>
inline void norm_deviatoric(const T& A, R& ret);

template <class T>
inline auto Norm_deviatoric(const T& A);

// Deviatoric part of a tensor (== A - Hydrostatic(A) * I2)
template <class T, class R>
inline void deviatoric(const T& A, R& ret);

template <class T>
inline auto Deviatoric(const T& A);

// inv(A)
template <class T, class R>
inline void inv(const T& A, R& ret);

template <class T>
inline auto Inv(const T& A);

// A . A^T
template <class T, class R>
inline void A2_dot_A2T(const T& A, R& ret);

template <class T>
inline auto A2_dot_A2T(const T& A);

// A . B
template <class T, class R>
inline void A2_dot_B2(const T& A, R& ret);

template <class T>
inline auto A2_dot_B2(const T& A);

// A * B
template <class T, class R>
inline void A2_dyadic_B2(const T& A, const T& B, R& ret);

template <class T>
inline auto A2_dyadic_B2(const T& A, const T& B);

// A : B
template <class T, class U, class R>
inline void A4_ddot_B2(const T& A, const U& B, R& ret);

template <class T, class U>
inline auto A4_ddot_B2(const T& A, const U& B);

// A . B
template <class T, class U, class R>
inline void A4_dot_B2(const T& A, const U& B, R& ret);

template <class T, class U>
inline auto A4_dot_B2(const T& A, const U& B);

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
    xt::xtensor<double, N + 2> O2() const;
    xt::xtensor<double, N + 4> O4() const;
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
// - Second order tensor: (xx, xy, xz, yx, yy, yz, zx, zy, zz)
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
    inline T trace(const T* A);

    // trace(A) / 3
    template <class T>
    inline T hydrostatic(const T* A);

    // Determinant
    template <class T>
    inline T det(const T* A);

    // inv(A)
    // Returns determinant
    template <class T>
    inline T inv(const T* A, T* ret);

    // Deviatoric decomposition of second order tensor
    // Returns hydrostatic part
    // "ret" may be the same as "A"
    template <class T>
    inline T hydrostatic_deviatoric(const T* A, T* ret);

    // dev(A) : dev(A)
    template <class T>
    inline T deviatoric_ddot_deviatoric(const T* A);

    // sqrt(dev(A) : dev(A))
    template <class T>
    inline T norm_deviatoric(const T* A);

    // A : B
    template <class T>
    inline T A2_ddot_B2(const T* A, const T* B);

    // A : B
    // Symmetric tensors only, no assertion
    template <class T>
    inline T A2s_ddot_B2s(const T* A, const T* B);

    // A * B
    template <class T>
    inline void A2_dyadic_B2(const T* A, const T* B, T* ret);

    // A . B
    template <class T>
    inline void A4_dot_B2(const T* A, const T* B, T* ret);

    // A . B
    template <class T>
    inline void A2_dot_B2(const T* A, const T* B, T* ret);

    // A . A^T
    template <class T>
    inline void A2_dot_A2T(const T* A, T* ret);

    // A : B
    template <class T>
    inline void A4_ddot_B2(const T* A, const T* B, T* ret);

    // A : B : C
    template <class T>
    inline void A4_ddot_B4_ddot_C4(const T* A, const T* B, const T* C, T* ret);

    // A . B . C^T
    template <class T>
    inline void A2_dot_B2_dot_C2T(const T* A, const T* B, const T* C, T* ret);

    // Get eigenvalues/-vectors such that "A_ij = lambda^a v^a_i v^a_j"
    // Symmetric tensors only, no assertion
    // Storage:
    // - lambda^a = val[a]
    // - v^a = vec[:, a]
    template <class T>
    void eigs(const T* A, T* vec, T* val);

    // Reconstruct tensor from eigenvalues/-vectors (reverse operation of "eigs")
    // Symmetric tensors only, no assertion
    template <class T>
    void from_eigs(const T* vec, const T* val, T* ret);

} // namespace pointer

} // namespace Cartesian3d
} // namespace GMatTensor

#include "Cartesian3d.hpp"
#include "Cartesian3d_Array.hpp"

#endif
