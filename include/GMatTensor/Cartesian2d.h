/**
 * @file
 * @copyright Copyright 2020. Tom de Geus. All rights reserved.
 * @license This project is released under the MIT License.
 */

#ifndef GMATTENSOR_CARTESIAN2D_H
#define GMATTENSOR_CARTESIAN2D_H

#include <xtensor/xadapt.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "config.h"
#include "detail.hpp"
#include "version.h"

namespace GMatTensor {

/**
 * Tensors and tensor operations for a(n) array of 2d tensors of different rank,
 * defined in a Cartesian coordinate system.
 */
namespace Cartesian2d {

/**
 * API for individual tensors with pointer-only input.
 * No arrays of tensors are allowed, hence the input is fixed to:
 *
 * -   Second order tensors, ``size = 2 * 2 = 4``.
 *     Storage convention ``(xx, xy, yx, yy)``.
 *
 * -   Fourth order tensors, ``size = 2 * 2 * 2 * 2 = 16``.
 */
namespace pointer {

/**
 * See Cartesian2d::O2()
 *
 * @param ret output 2nd order tensor
 */
template <class T>
inline void O2(T* ret)
{
    std::fill(ret, ret + 4, T(0));
}

/**
 * See Cartesian2d::O4()
 *
 * @param ret output 2nd order tensor
 */
template <class T>
inline void O4(T* ret)
{
    std::fill(ret, ret + 16, T(0));
}

/**
 * See Cartesian2d::I2()
 *
 * @param ret output 2nd order tensor
 */
template <class T>
inline void I2(T* ret)
{
    ret[0] = 1.0;
    ret[1] = 0.0;
    ret[2] = 0.0;
    ret[3] = 1.0;
}

/**
 * See Cartesian2d::II()
 *
 * @param ret output 2nd order tensor
 */
template <class T>
inline void II(T* ret)
{
    std::fill(ret, ret + 16, T(0));

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                for (size_t l = 0; l < 2; ++l) {
                    if (i == j && k == l) {
                        ret[i * 8 + j * 4 + k * 2 + l] = 1.0;
                    }
                }
            }
        }
    }
}

/**
 * See Cartesian2d::I4()
 *
 * @param ret output 2nd order tensor
 */
template <class T>
inline void I4(T* ret)
{
    std::fill(ret, ret + 16, T(0));

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                for (size_t l = 0; l < 2; ++l) {
                    if (i == l && j == k) {
                        ret[i * 8 + j * 4 + k * 2 + l] = 1.0;
                    }
                }
            }
        }
    }
}

/**
 * See Cartesian2d::I4rt()
 *
 * @param ret output 2nd order tensor
 */
template <class T>
inline void I4rt(T* ret)
{
    std::fill(ret, ret + 16, T(0));

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                for (size_t l = 0; l < 2; ++l) {
                    if (i == k && j == l) {
                        ret[i * 8 + j * 4 + k * 2 + l] = 1.0;
                    }
                }
            }
        }
    }
}

/**
 * See Cartesian2d::I4s()
 *
 * @param ret output 2nd order tensor
 */
template <class T>
inline void I4s(T* ret)
{
    I4(ret);

    std::array<double, 16> i4rt;
    I4rt(&i4rt[0]);

    std::transform(ret, ret + 16, &i4rt[0], ret, std::plus<T>());

    std::transform(ret, ret + 16, ret, std::bind(std::multiplies<T>(), std::placeholders::_1, 0.5));
}

/**
 * See Cartesian2d::I4d()
 *
 * @param ret output 2nd order tensor
 */
template <class T>
inline void I4d(T* ret)
{
    I4s(ret);

    std::array<double, 16> ii;
    II(&ii[0]);

    std::transform(
        &ii[0], &ii[0] + 16, &ii[0], std::bind(std::multiplies<T>(), std::placeholders::_1, 0.5));

    std::transform(ret, ret + 16, &ii[0], ret, std::minus<T>());
}

/**
 * See Cartesian2d::Trace()
 *
 * @param A 2nd order tensor
 * @return scalar
 */
template <class T>
inline T Trace(const T* A)
{
    return A[0] + A[3];
}

/**
 * See Cartesian2d::Hydrostatic()
 *
 * @param A 2nd order tensor
 * @return scalar
 */
template <class T>
inline T Hydrostatic(const T* A)
{
    return T(0.5) * Trace(A);
}

/**
 * See Cartesian2d::Sym()
 *
 * @param A 2nd order tensor
 * @param ret 2nd order tensor, may be the same pointer as ``A``
 */
template <class T>
inline void sym(const T* A, T* ret)
{
    ret[0] = A[0];
    ret[1] = 0.5 * (A[1] + A[2]);
    ret[2] = ret[1];
    ret[3] = A[3];
}

/**
 * Returns Cartesian2d::Hydrostatic() and computes Cartesian2d::Deviatoric()
 *
 * @param A 2nd order tensor
 * @param ret 2nd order tensor, may be the same pointer as ``A``
 * @return scalar
 */
template <class T>
inline T Hydrostatic_deviatoric(const T* A, T* ret)
{
    T m = Hydrostatic(A);
    ret[0] = A[0] - m;
    ret[1] = A[1];
    ret[2] = A[2];
    ret[3] = A[3] - m;
    return m;
}

/**
 * Double tensor contraction of the tensor's deviator
 *
 * \f$ (dev(A))_{ij} (dev(A))_{ji} \f$
 *
 * @param A 2nd order tensor
 * @return scalar
 */
template <class T>
inline T Deviatoric_ddot_deviatoric(const T* A)
{
    T m = Hydrostatic(A);
    return (A[0] - m) * (A[0] - m) + (A[3] - m) * (A[3] - m) + T(2) * A[1] * A[2];
}

/**
 * See Cartesian2d::Norm_deviatoric()
 *
 * @param A 2nd order tensor
 * @return scalar
 */
template <class T>
inline T Norm_deviatoric(const T* A)
{
    return std::sqrt(Deviatoric_ddot_deviatoric(A));
}

/**
 * See Cartesian2d::A2_ddot_B2()
 *
 * @param A 2nd order tensor
 * @param B 2nd order tensor
 * @return scalar
 */
template <class T>
inline T A2_ddot_B2(const T* A, const T* B)
{
    return A[0] * B[0] + A[3] * B[3] + A[1] * B[2] + A[2] * B[1];
}

/**
 * See Cartesian2d::A2s_ddot_B2s()
 *
 * @param A 2nd order tensor
 * @param B 2nd order tensor
 * @return scalar
 */
template <class T>
inline T A2s_ddot_B2s(const T* A, const T* B)
{
    return A[0] * B[0] + A[3] * B[3] + T(2) * A[1] * B[1];
}

/**
 * See Cartesian2d::A2_dyadic_B2()
 *
 * @param A 2nd order tensor
 * @param B 2nd order tensor
 * @param ret output 4th order tensor
 */
template <class T>
inline void A2_dyadic_B2(const T* A, const T* B, T* ret)
{
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                for (size_t l = 0; l < 2; ++l) {
                    ret[i * 8 + j * 4 + k * 2 + l] = A[i * 2 + j] * B[k * 2 + l];
                }
            }
        }
    }
}

/**
 * See Cartesian2d::A2_dot_B2()
 *
 * @param A 2nd order tensor
 * @param B 2nd order tensor
 * @param ret output 2th order tensor
 */
template <class T>
inline void A2_dot_B2(const T* A, const T* B, T* ret)
{
    ret[0] = A[1] * B[2] + A[0] * B[0];
    ret[1] = A[0] * B[1] + A[1] * B[3];
    ret[2] = A[2] * B[0] + A[3] * B[2];
    ret[3] = A[2] * B[1] + A[3] * B[3];
}

/**
 * See Cartesian2d::A4_ddot_B2()
 *
 * @param A 4th order tensor
 * @param B 2nd order tensor
 * @param ret output 2th order tensor
 */
template <class T>
inline void A4_ddot_B2(const T* A, const T* B, T* ret)
{
    std::fill(ret, ret + 4, T(0));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            for (size_t k = 0; k < 2; k++) {
                for (size_t l = 0; l < 2; l++) {
                    ret[i * 2 + j] += A[i * 8 + j * 4 + k * 2 + l] * B[l * 2 + k];
                }
            }
        }
    }
}

} // namespace pointer

/**
 * Random 2nd-order tensor (for example for use in testing).
 *
 * @return [2, 2] array.
 */
inline array_type::tensor<double, 2> Random2()
{
    array_type::tensor<double, 2> ret = xt::random::randn<double>({2, 2});
    return ret;
}

/**
 * Random 4th-order tensor (for example for use in testing).
 *
 * @return [2, 2, 2, 2] array.
 */
inline array_type::tensor<double, 4> Random4()
{
    array_type::tensor<double, 4> ret = xt::random::randn<double>({2, 2, 2, 2});
    return ret;
}

/**
 * 2nd-order null tensor (all components equal to zero).
 *
 * @return [2, 2] array.
 */
inline array_type::tensor<double, 2> O2()
{
    return xt::zeros<double>({2, 2});
}

/**
 * 4th-order null tensor (all components equal to zero).
 *
 * @return [2, 2, 2, 2] array.
 */
inline array_type::tensor<double, 4> O4()
{
    return xt::zeros<double>({2, 2, 2, 2});
}

/**
 * 2nd-order identity tensor.
 * By definition
 *
 * \f$ I_{ij} = \delta_{ij} \f$
 *
 * such that
 *
 * \f$ I \cdot A = A \f$
 *
 * or in index notation
 *
 * \f$ I_{ij} A_{jk} = A_{ik} \f$
 *
 * See A2_dot_B2().
 *
 * @return [2, 2] array.
 */
inline array_type::tensor<double, 2> I2()
{
    array_type::tensor<double, 2> ret = xt::empty<double>({2, 2});
    pointer::I2(ret.data());
    return ret;
}

/**
 * Result of the dyadic product of two 2nd-order identity tensors (see I2()).
 * By definition
 *
 * \f$ (II)_{ijkl} = \delta_{ij} \delta_{kl} \f$
 *
 * such that
 *
 * \f$ II : A = tr(A) I \f$
 *
 * or in index notation
 *
 * \f$ (II)_{ijkl} A_{lk} = tr(A) I_{ij} \f$
 *
 * See A4_ddot_B2(), Trace(), I2().
 *
 * @return [2, 2, 2, 2] array.
 */
inline array_type::tensor<double, 4> II()
{
    array_type::tensor<double, 4> ret = xt::empty<double>({2, 2, 2, 2});
    pointer::II(ret.data());
    return ret;
}

/**
 * Fourth order unit tensor.
 * By definition
 *
 * \f$ I_{ijkl} = \delta_{il} \delta_{jk} \f$
 *
 * such that
 *
 * \f$ I : A = A \f$
 *
 * or in index notation
 *
 * \f$ I_{ijkl} A_{lk} = A_{ij} \f$
 *
 * See A4_ddot_B2().
 *
 * @return [2, 2, 2, 2] array.
 */
inline array_type::tensor<double, 4> I4()
{
    array_type::tensor<double, 4> ret = xt::empty<double>({2, 2, 2, 2});
    pointer::I4(ret.data());
    return ret;
}

/**
 * Right-transposed fourth order unit tensor.
 * By definition
 *
 * \f$ I_{ijkl} = \delta_{ik} \delta_{jl} \f$
 *
 * such that
 *
 * \f$ I : A = A^T \f$
 *
 * or in index notation
 *
 * \f$ I_{ijkl} A_{lk} = A_{ji} \f$
 *
 * See A4_ddot_B2().
 *
 * @return [2, 2, 2, 2] array.
 */
inline array_type::tensor<double, 4> I4rt()
{
    array_type::tensor<double, 4> ret = xt::empty<double>({2, 2, 2, 2});
    pointer::I4rt(ret.data());
    return ret;
}

/**
 * Fourth order symmetric projection.
 * By definition
 *
 *     I = 0.5 * (I4() + I4rt())
 *
 * such that
 *
 *  \f$ I : A = sym(A) \f$
 *
 *  or in index notation
 *
 *  \f$ I_{ijkl} A_{lk} = (A_{ij} + A_{ji}) / 2 \f$
 *
 * See A4_ddot_B2(), Sym().
 *
 * @return [2, 2, 2, 2] array.
 */
inline array_type::tensor<double, 4> I4s()
{
    array_type::tensor<double, 4> ret = xt::empty<double>({2, 2, 2, 2});
    pointer::I4s(ret.data());
    return ret;
}

/**
 * Fourth order deviatoric projection.
 * By definition
 *
 *     I = I4s() - 0.5 * II()
 *
 * such that
 *
 * \f$ I : A = sym(A) - tr(A) / 2 \f$
 *
 * See A4_ddot_B2(), Deviatoric().
 *
 * @return [2, 2, 2, 2] array.
 */
inline array_type::tensor<double, 4> I4d()
{
    array_type::tensor<double, 4> ret = xt::empty<double>({2, 2, 2, 2});
    pointer::I4d(ret.data());
    return ret;
}

/**
 * Trace or 2nd-order tensor.
 *
 * \f$ tr(A) = A_{ii} \f$
 *
 * To write to allocated data use trace().
 *
 * @param A [..., 2, 2] array.
 * @return [...] array.
 */
template <class T>
inline auto Trace(const T& A)
{
    return detail::impl_A2<T, 2>::ret0(A, [](const auto& a) { return pointer::Trace(a); });
}

/**
 * Same as Trace() but writes to externally allocated output.
 *
 * @param A [..., 2, 2] array.
 * @param ret output [...] array.
 */
template <class T, class R>
inline void trace(const T& A, R& ret)
{
    detail::impl_A2<T, 2>::ret0(A, ret, [](const auto& a) { return pointer::Trace(a); });
}

/**
 * Hydrostatic part of a tensor
 *
 *     == trace(A) / 2 == trace(A) / d
 *
 * where ``d = 2``.
 * To write to allocated output use hydrostatic().
 *
 * @param A [..., 2, 2] array.
 * @return [...] array.
 */
template <class T>
inline auto Hydrostatic(const T& A)
{
    return detail::impl_A2<T, 2>::ret0(A, [](const auto& a) { return pointer::Hydrostatic(a); });
}

/**
 * Same as Hydrostatic() but writes to externally allocated output.
 *
 * @param A [..., 2, 2] array.
 * @param ret output [...] array.
 */
template <class T, class R>
inline void hydrostatic(const T& A, R& ret)
{
    detail::impl_A2<T, 2>::ret0(A, ret, [](const auto& a) { return pointer::Hydrostatic(a); });
}

/**
 * Double tensor contraction
 *
 * \f$ c = A : B \f$
 *
 * or in index notation
 *
 * \f$ c = A_{ij} A_{ji} \f$
 *
 * To write to allocated data use A2_ddot_B2(const T& A, const T& B, R& ret).
 *
 * @param A [..., 2, 2] array.
 * @param B [..., 2, 2] array.
 * @return [...] array.
 */
template <class T>
inline auto A2_ddot_B2(const T& A, const T& B)
{
    return detail::impl_A2<T, 2>::B2_ret0(
        A, B, [](const auto& a, const auto& b) { return pointer::A2_ddot_B2(a, b); });
}

/**
 * Same as A2_ddot_B2(const T& A, const T& B) but writes to externally allocated output.
 *
 * @param A [..., 2, 2] array.
 * @param B [..., 2, 2] array.
 * @param ret output [...] array.
 */
template <class T, class R>
inline void A2_ddot_B2(const T& A, const T& B, R& ret)
{
    detail::impl_A2<T, 2>::B2_ret0(
        A, B, ret, [](const auto& a, const auto& b) { return pointer::A2_ddot_B2(a, b); });
}

/**
 * Same as A2_ddot_B2(const T& A, const T& B, R& ret) but for symmetric tensors.
 * This function is slightly faster.
 * There is no assertion to check the symmetry.
 * To write to allocated data use A2s_ddot_B2s(const T& A, const T& B, R& ret).
 *
 * @param A [..., 2, 2] array.
 * @param B [..., 2, 2] array.
 * @return [...] array.
 */
template <class T>
inline auto A2s_ddot_B2s(const T& A, const T& B)
{
    return detail::impl_A2<T, 2>::B2_ret0(
        A, B, [](const auto& a, const auto& b) { return pointer::A2s_ddot_B2s(a, b); });
}

/**
 * Same as A2s_ddot_B2s(const T& A, const T& B) but writes to externally allocated output.
 *
 * @param A [..., 2, 2] array.
 * @param B [..., 2, 2] array.
 * @param ret output [...] array.
 */
template <class T, class R>
inline void A2s_ddot_B2s(const T& A, const T& B, R& ret)
{
    detail::impl_A2<T, 2>::B2_ret0(
        A, B, ret, [](const auto& a, const auto& b) { return pointer::A2s_ddot_B2s(a, b); });
}

/**
 * Norm of the tensor's deviator:
 *
 * \f$ \sqrt{(dev(A))_{ij} (dev(A))_{ji}} \f$
 *
 * To write to allocated data use norm_deviatoric().
 *
 * @param A [..., 2, 2] array.
 * @return [...] array.
 */
template <class T>
inline auto Norm_deviatoric(const T& A)
{
    return detail::impl_A2<T, 2>::ret0(
        A, [](const auto& a) { return pointer::Norm_deviatoric(a); });
}

/**
 * Same as Norm_deviatoric() but writes to externally allocated output.
 *
 * @param A [..., 2, 2] array.
 * @param ret output [...] array
 */
template <class T, class R>
inline void norm_deviatoric(const T& A, R& ret)
{
    detail::impl_A2<T, 2>::ret0(A, ret, [](const auto& a) { return pointer::Norm_deviatoric(a); });
}

/**
 * Deviatoric part of a tensor:
 *
 *     A - Hydrostatic(A) * I2
 *
 * See Hydrostatic().
 * To write to allocated data use deviatoric().
 *
 * @param A [..., 2, 2] array.
 * @return [..., 2, 2] array.
 */
template <class T>
inline auto Deviatoric(const T& A)
{
    return detail::impl_A2<T, 2>::ret2(
        A, [](const auto& a, const auto& r) { return pointer::Hydrostatic_deviatoric(a, r); });
}

/**
 * Same as Deviatoric() but writes to externally allocated output.
 *
 * @param A [..., 2, 2] array.
 * @param ret output [..., 2, 2] array.
 */
template <class T, class R>
inline void deviatoric(const T& A, R& ret)
{
    detail::impl_A2<T, 2>::ret2(
        A, ret, [](const auto& a, const auto& r) { return pointer::Hydrostatic_deviatoric(a, r); });
}

/**
 * Symmetric part of a tensor:
 *
 * \f$ (A + A^T) / 2 \f$
 *
 * of in index notation
 *
 * \f$ (A_{ij} + A_{ji}) / 2 \f$
 *
 * To write to allocated data use sym().
 *
 * @param A [..., 2, 2] array.
 * @return [..., 2, 2] array.
 */
template <class T>
inline auto Sym(const T& A)
{
    return detail::impl_A2<T, 2>::ret2(
        A, [](const auto& a, const auto& r) { return pointer::sym(a, r); });
}

/**
 * Same as Sym() but writes to externally allocated output.
 *
 * @param A [..., 2, 2] array.
 * @param ret output [..., 2, 2] array, may be the same reference as ``A``.
 */
template <class T, class R>
inline void sym(const T& A, R& ret)
{
    detail::impl_A2<T, 2>::ret2(
        A, ret, [](const auto& a, const auto& r) { return pointer::sym(a, r); });
}

/**
 * Dot-product (single tensor contraction)
 *
 * \f$ C = A \cdot B \f$
 *
 * or in index notation
 *
 * \f$ C_{ik} = A_{ij} B_{jk} \f$
 *
 * To write to allocated data use A2_dot_B2(const T& A, const T& B, R& ret).
 *
 * @param A [..., 2, 2] array.
 * @param B [..., 2, 2] array.
 * @return [..., 2, 2] array.
 */
template <class T>
inline auto A2_dot_B2(const T& A, const T& B)
{
    return detail::impl_A2<T, 2>::B2_ret2(A, B, [](const auto& a, const auto& b, const auto& r) {
        return pointer::A2_dot_B2(a, b, r);
    });
}

/**
 * Same as A2_dot_B2(const T& A, const T& B) but writes to externally allocated output.
 *
 * @param A [..., 2, 2] array.
 * @param B [..., 2, 2] array.
 * @param ret output [..., 2, 2] array.
 */
template <class T, class R>
inline void A2_dot_B2(const T& A, const T& B, R& ret)
{
    detail::impl_A2<T, 2>::B2_ret2(A, B, ret, [](const auto& a, const auto& b, const auto& r) {
        return pointer::A2_dot_B2(a, b, r);
    });
}

/**
 * Dyadic product
 *
 * \f$ C = A \otimes B \f$
 *
 * or in index notation
 *
 * \f$ C_{ijkl} = A_{ij} B_{kl} \f$
 *
 * To write to allocated data use A2_dyadic_B2(const T& A, const T& B, R& ret).
 *
 * @param A [..., 2, 2] array.
 * @param B [..., 2, 2] array.
 * @return [..., 2, 2, 2, 2] array.
 */
template <class T>
inline auto A2_dyadic_B2(const T& A, const T& B)
{
    return detail::impl_A2<T, 2>::B2_ret4(A, B, [](const auto& a, const auto& b, const auto& r) {
        return pointer::A2_dyadic_B2(a, b, r);
    });
}

/**
 * Same as A2_dyadic_B2(const T& A, const T& B) but writes to externally allocated output.
 *
 * @param A [..., 2, 2] array.
 * @param B [..., 2, 2] array.
 * @param ret output [..., 2, 2, 2, 2] array.
 */
template <class T, class R>
inline void A2_dyadic_B2(const T& A, const T& B, R& ret)
{
    detail::impl_A2<T, 2>::B2_ret4(A, B, ret, [](const auto& a, const auto& b, const auto& r) {
        return pointer::A2_dyadic_B2(a, b, r);
    });
}

/**
 * Double tensor contraction
 *
 * \f$ C = A : B \f$
 *
 * or in index notation
 *
 * \f$ C_{ij} = A_{ijkl} A_{lk} \f$
 *
 * To write to allocated data use A4_ddot_B2(const T& A, const U& B, R& ret).
 *
 * @param A [..., 2, 2, 2, 2] array.
 * @param B [..., 2, 2] array.
 * @return [..., 2, 2] array.
 */
template <class T, class U>
inline auto A4_ddot_B2(const T& A, const U& B)
{
    return detail::impl_A4<T, 2>::B2_ret2(A, B, [](const auto& a, const auto& b, const auto& r) {
        return pointer::A4_ddot_B2(a, b, r);
    });
}

/**
 * Same as A4_ddot_B2(const T& A, const U& B) but writes to externally allocated output.
 *
 * @param A [..., 2, 2, 2, 2] array.
 * @param B [..., 2, 2] array.
 * @param ret output [..., 2, 2] array.
 */
template <class T, class U, class R>
inline void A4_ddot_B2(const T& A, const U& B, R& ret)
{
    detail::impl_A4<T, 2>::B2_ret2(A, B, ret, [](const auto& a, const auto& b, const auto& r) {
        return pointer::A4_ddot_B2(a, b, r);
    });
}

/**
 * Size of the underlying array.
 *
 * @param A [..., 2, 2] array.
 * @return `prod([...])`.
 */
template <class T>
inline size_t underlying_size_A2(const T& A)
{
    return detail::impl_A2<T, 2>::toSizeT0(A.shape());
}

/**
 * Size of the underlying array.
 *
 * @param A [..., 2, 2] array.
 * @return `prod([...])`.
 */
template <class T>
inline size_t underlying_size_A4(const T& A)
{
    return detail::impl_A4<T, 2>::toSizeT0(A.shape());
}

/**
 * Shape of the underlying array.
 *
 * @param A [..., 2, 2] array.
 * @return `[...]`.
 */
template <class T>
inline auto underlying_shape_A2(const T& A) -> std::array<size_t, detail::impl_A2<T, 2>::rank>
{
    return detail::impl_A2<T, 2>::toShapeT0(A.shape());
}

/**
 * Shape of the underlying array.
 *
 * @param A [..., 2, 2] array.
 * @return `[...]`.
 */
template <class T>
inline auto underlying_shape_A4(const T& A) -> std::array<size_t, detail::impl_A4<T, 2>::rank>
{
    return detail::impl_A4<T, 2>::toShapeT0(A.shape());
}

/**
 * Array of tensors:
 * -   scalars: shape ``[...]``.
 * -   2nd-order tensors: shape ``[..., 2, 2]``.
 * -   4nd-order tensors: shape ``[..., 2, 2, 2, 2]``.
 *
 * @tparam N The rank of the array (the actual rank is increased with the tensor-rank).
 */
template <size_t N>
class Array {
public:
    /**
     * Rank of the array (the actual rank is increased with the tensor-rank).
     */
    constexpr static std::size_t rank = N;

    Array() = default;

    virtual ~Array() = default;

    /**
     * Constructor.
     *
     * @param shape The shape of the array (or scalars).
     */
    Array(const std::array<size_t, N>& shape)
    {
        this->init(shape);
    }

    /**
     * Shape of the array (of scalars).
     *
     * @return List of size #rank.
     */
    const std::array<size_t, N>& shape() const
    {
        return m_shape;
    }

    /**
     * Shape of the array of second-order tensors.
     *
     * @return List of size #rank + 2.
     */
    const std::array<size_t, N + 2>& shape_tensor2() const
    {
        return m_shape_tensor2;
    }

    /**
     * Shape of the array of fourth-order tensors.
     *
     * @return List of size #rank + 4.
     */
    const std::array<size_t, N + 4>& shape_tensor4() const
    {
        return m_shape_tensor4;
    }

    /**
     * Array of Cartesian2d::O2()
     *
     * @return [shape(), 2, 2]
     */
    array_type::tensor<double, N + 2> O2() const
    {
        return xt::zeros<double>(m_shape_tensor2);
    }

    /**
     * Array of Cartesian2d::O4()
     *
     * @return [shape(), 2, 2, 2, 2]
     */
    array_type::tensor<double, N + 4> O4() const
    {
        return xt::zeros<double>(m_shape_tensor4);
    }

    /**
     * Array of Cartesian2d::I2()
     *
     * @return [shape(), 2, 2]
     */
    array_type::tensor<double, N + 2> I2() const
    {
        array_type::tensor<double, N + 2> ret = xt::empty<double>(m_shape_tensor2);

#pragma omp parallel for
        for (size_t i = 0; i < m_size; ++i) {
            Cartesian2d::pointer::I2(&ret.flat(i * m_stride_tensor2));
        }

        return ret;
    }

    /**
     * Array of Cartesian2d::II()
     *
     * @return [shape(), 2, 2, 2, 2]
     */
    array_type::tensor<double, N + 4> II() const
    {
        array_type::tensor<double, N + 4> ret = xt::empty<double>(m_shape_tensor4);

#pragma omp parallel for
        for (size_t i = 0; i < m_size; ++i) {
            Cartesian2d::pointer::II(&ret.flat(i * m_stride_tensor4));
        }

        return ret;
    }

    /**
     * Array of Cartesian2d::I4()
     *
     * @return [shape(), 2, 2, 2, 2]
     */
    array_type::tensor<double, N + 4> I4() const
    {
        array_type::tensor<double, N + 4> ret = xt::empty<double>(m_shape_tensor4);

#pragma omp parallel for
        for (size_t i = 0; i < m_size; ++i) {
            Cartesian2d::pointer::I4(&ret.flat(i * m_stride_tensor4));
        }

        return ret;
    }

    /**
     * Array of Cartesian2d::I4rt()
     *
     * @return [shape(), 2, 2, 2, 2]
     */
    array_type::tensor<double, N + 4> I4rt() const
    {
        array_type::tensor<double, N + 4> ret = xt::empty<double>(m_shape_tensor4);

#pragma omp parallel for
        for (size_t i = 0; i < m_size; ++i) {
            Cartesian2d::pointer::I4rt(&ret.flat(i * m_stride_tensor4));
        }

        return ret;
    }

    /**
     * Array of Cartesian2d::I4s()
     *
     * @return [shape(), 2, 2, 2, 2]
     */
    array_type::tensor<double, N + 4> I4s() const
    {
        array_type::tensor<double, N + 4> ret = xt::empty<double>(m_shape_tensor4);

#pragma omp parallel for
        for (size_t i = 0; i < m_size; ++i) {
            Cartesian2d::pointer::I4s(&ret.flat(i * m_stride_tensor4));
        }

        return ret;
    }

    /**
     * Array of Cartesian2d::I4d()
     *
     * @return [shape(), 2, 2, 2, 2]
     */
    array_type::tensor<double, N + 4> I4d() const
    {
        array_type::tensor<double, N + 4> ret = xt::empty<double>(m_shape_tensor4);

#pragma omp parallel for
        for (size_t i = 0; i < m_size; ++i) {
            Cartesian2d::pointer::I4d(&ret.flat(i * m_stride_tensor4));
        }

        return ret;
    }

protected:
    /**
     * Constructor 'alias'. Can be used by constructor of derived classes.
     *
     * @param shape The shape of the array (or scalars).
     */
    void init(const std::array<size_t, N>& shape)
    {
        m_shape = shape;
        size_t nd = m_ndim;
        std::copy(m_shape.begin(), m_shape.end(), m_shape_tensor2.begin());
        std::copy(m_shape.begin(), m_shape.end(), m_shape_tensor4.begin());
        std::fill(m_shape_tensor2.begin() + N, m_shape_tensor2.end(), nd);
        std::fill(m_shape_tensor4.begin() + N, m_shape_tensor4.end(), nd);
        m_size = std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<size_t>());
    }

    /** Number of dimensions of tensors. */
    static constexpr size_t m_ndim = 2;

    /** Storage stride for 2nd-order tensors (\f$ 2^2 \f$). */
    static constexpr size_t m_stride_tensor2 = 4;

    /** Storage stride for 4th-order tensors (\f$ 2^4 \f$). */
    static constexpr size_t m_stride_tensor4 = 16;

    /** Size of the array (of scalars) == prod(#m_shape). */
    size_t m_size;

    /** Shape of the array (of scalars). */
    std::array<size_t, N> m_shape;

    /** Shape of an array of 2nd-order tensors == [#m_shape, 2, 2]. */
    std::array<size_t, N + 2> m_shape_tensor2;

    /** Shape of an array of 4th-order tensors == [#m_shape, 2, 2, 2, 2]. */
    std::array<size_t, N + 4> m_shape_tensor4;
};

} // namespace Cartesian2d
} // namespace GMatTensor

#endif
