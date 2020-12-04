/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatTensor

*/

#ifndef GMATTENSOR_CARTESIAN3D_HPP
#define GMATTENSOR_CARTESIAN3D_HPP

#include "Cartesian3d.h"

namespace GMatTensor {
namespace Cartesian3d {

inline xt::xtensor<double, 2> O2()
{
    xt::xtensor<double, 2> ret = xt::zeros<double>({3, 3});
    return ret;
}

inline xt::xtensor<double, 4> O4()
{
    xt::xtensor<double, 4> ret = xt::zeros<double>({3, 3, 3, 3});
    return ret;
}

inline xt::xtensor<double, 2> I2()
{
    return xt::xtensor<double, 2>({{1.0, 0.0, 0.0},
                                   {0.0, 1.0, 0.0},
                                   {0.0, 0.0, 1.0}});
}

inline xt::xtensor<double, 4> II()
{
    xt::xtensor<double, 4> ret = xt::zeros<double>({3, 3, 3, 3});

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 3; ++k) {
                for (size_t l = 0; l < 3; ++l) {
                    if (i == j && k == l) {
                        ret(i, j, k, l) = 1.0;
                    }
                }
            }
        }
    }

    return ret;
}

inline xt::xtensor<double, 4> I4()
{
    xt::xtensor<double, 4> ret = xt::zeros<double>({3, 3, 3, 3});

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 3; ++k) {
                for (size_t l = 0; l < 3; ++l) {
                    if (i == l && j == k) {
                        ret(i, j, k, l) = 1.0;
                    }
                }
            }
        }
    }

    return ret;
}

inline xt::xtensor<double, 4> I4rt()
{
    xt::xtensor<double, 4> ret = xt::zeros<double>({3, 3, 3, 3});

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 3; ++k) {
                for (size_t l = 0; l < 3; ++l) {
                    if (i == k && j == l) {
                        ret(i, j, k, l) = 1.0;
                    }
                }
            }
        }
    }

    return ret;
}

inline xt::xtensor<double, 4> I4s()
{
    return 0.5 * (I4() + I4rt());
}

inline xt::xtensor<double, 4> I4d()
{
    return I4s() - II() / 3.0;
}

template <class T>
inline auto trace(const T& A)
{
    GMATTENSOR_ASSERT(xt::has_shape(A, {3, 3}));
    return pointer::trace(A.data());
}

template <class T>
inline auto det(const T& A)
{
    GMATTENSOR_ASSERT(xt::has_shape(A, {3, 3}));
    return pointer::det(A.data());
}

template <class T>
inline auto inv(const T& A)
{
    GMATTENSOR_ASSERT(xt::has_shape(A, {3, 3}));
    xt::xtensor<double, 2> ret = xt::zeros<double>({3, 3});
    pointer::inv(A.data(), ret.data());
    return ret;
}

template <class S, class T>
inline auto A2_ddot_B2(const S& A, const T& B)
{
    GMATTENSOR_ASSERT(xt::has_shape(A, {3, 3}));
    GMATTENSOR_ASSERT(xt::has_shape(B, {3, 3}));
    return pointer::A2_ddot_B2(A.data(), B.data());
}

template <class S, class T>
inline auto A2s_ddot_B2s(const S& A, const T& B)
{
    GMATTENSOR_ASSERT(xt::has_shape(A, {3, 3}));
    GMATTENSOR_ASSERT(xt::has_shape(B, {3, 3}));
    return pointer::A2s_ddot_B2s(A.data(), B.data());
}

template <class S, class T>
inline auto A2_dyadic_B2(const S& A, const T& B)
{
    GMATTENSOR_ASSERT(xt::has_shape(A, {3, 3}));
    GMATTENSOR_ASSERT(xt::has_shape(B, {3, 3}));
    xt::xtensor<double, 4> ret = xt::zeros<double>({3, 3, 3, 3});
    pointer::A2_dyadic_B2(A.data(), B.data(), ret.data());
    return ret;
}

template <class S, class T>
inline auto A4_dot_B2(const S& A, const T& B)
{
    GMATTENSOR_ASSERT(xt::has_shape(A, {3, 3, 3, 3}));
    GMATTENSOR_ASSERT(xt::has_shape(B, {3, 3}));
    xt::xtensor<double, 4> ret = xt::zeros<double>({3, 3, 3, 3});
    pointer::A4_dot_B2(A.data(), B.data(), ret.data());
    return ret;
}

template <class S, class T>
inline auto A2_dot_B2(const S& A, const T& B)
{
    GMATTENSOR_ASSERT(xt::has_shape(A, {3, 3}));
    GMATTENSOR_ASSERT(xt::has_shape(B, {3, 3}));
    xt::xtensor<double, 2> ret = xt::zeros<double>({3, 3});
    pointer::A2_dot_B2(A.data(), B.data(), ret.data());
    return ret;
}

template <class T>
inline auto A2_dot_A2T(const T& A)
{
    GMATTENSOR_ASSERT(xt::has_shape(A, {3, 3}));
    xt::xtensor<double, 2> ret = xt::zeros<double>({3, 3});
    pointer::A2_dot_A2T(A.data(), ret.data());
    return ret;
}

template <class S, class T>
inline auto A4_ddot_B2(const S& A, const T& B)
{
    GMATTENSOR_ASSERT(xt::has_shape(A, {3, 3, 3, 3}));
    GMATTENSOR_ASSERT(xt::has_shape(B, {3, 3}));
    xt::xtensor<double, 2> ret = xt::zeros<double>({3, 3});
    pointer::A4_ddot_B2(A.data(), B.data(), ret.data());
    return ret;
}

namespace detail {

    template <class T>
    struct equiv_impl
    {
        using value_type = typename T::value_type;
        using shape_type = typename T::shape_type;
        static_assert(xt::has_fixed_rank_t<T>::value, "Only fixed rank allowed.");
        static_assert(xt::get_rank<T>::value >= 2, "Rank too low.");
        constexpr static size_t rank = xt::get_rank<T>::value;

        template <class S>
        static size_t toMatrixSize(const S& shape)
        {
            using ST = typename S::value_type;
            return std::accumulate(shape.cbegin(), shape.cend() - 2, ST(1), std::multiplies<ST>());
        }

        template <class S>
        static std::array<size_t, rank - 2> toMatrixShape(const S& shape)
        {
            std::array<size_t, rank - 2> ret;
            std::copy(shape.cbegin(), shape.cend() - 2, ret.begin());
            return ret;
        }

        template <class S>
        static std::array<size_t, rank> toShape(const S& shape)
        {
            std::array<size_t, rank> ret;
            std::copy(shape.cbegin(), shape.cend() - 2, ret.begin());
            ret[rank - 2] = 3;
            ret[rank - 1] = 3;
            return ret;
        }

        static void hydrostatic_no_alloc(const T& A, xt::xtensor<value_type, rank - 2>& B)
        {
            GMATTENSOR_ASSERT(xt::has_shape(A, toShape(A.shape())));
            GMATTENSOR_ASSERT(xt::has_shape(B, toMatrixShape(A.shape())));
            #pragma omp parallel for
            for (size_t i = 0; i < toMatrixSize(A.shape()); ++i) {
                B.data()[i] = pointer::trace(&A.data()[i * 9]) / 3.0;
            }
        }

        static void deviatoric_no_alloc(const T& A, xt::xtensor<value_type, rank>& B)
        {
            GMATTENSOR_ASSERT(xt::has_shape(A, toShape(A.shape())));
            GMATTENSOR_ASSERT(xt::has_shape(A, B.shape()));
            #pragma omp parallel for
            for (size_t i = 0; i < toMatrixSize(A.shape()); ++i) {
                pointer::hydrostatic_deviatoric(&A.data()[i * 9], &B.data()[i * 9]);
            }
        }

        static void equivalent_deviatoric_no_alloc(const T& A, xt::xtensor<value_type, rank - 2>& B)
        {
            GMATTENSOR_ASSERT(xt::has_shape(A, toShape(A.shape())));
            GMATTENSOR_ASSERT(xt::has_shape(B, toMatrixShape(A.shape())));
            #pragma omp parallel for
            for (size_t i = 0; i < toMatrixSize(A.shape()); ++i) {
                auto b = pointer::deviatoric_ddot_deviatoric(&A.data()[i * 9]);
                B.data()[i] = std::sqrt( b);
            }
        }

        static auto hydrostatic_alloc(const T& A)
        {
            xt::xtensor<value_type, rank - 2> B = xt::empty<value_type>(toMatrixShape(A.shape()));
            hydrostatic_no_alloc(A, B);
            return B;
        }

        static auto deviatoric_alloc(const T& A)
        {
            xt::xtensor<value_type, rank> B = xt::empty<value_type>(A.shape());
            deviatoric_no_alloc(A, B);
            return B;
        }

        static auto equivalent_deviatoric_alloc(const T& A)
        {
            xt::xtensor<value_type, rank - 2> B = xt::empty<value_type>(toMatrixShape(A.shape()));
            equivalent_deviatoric_no_alloc(A, B);
            return B;
        }
    };

} // namespace detail

template <class T, class U>
inline void hydrostatic(const T& A, U& B)
{
    return detail::equiv_impl<T>::hydrostatic_no_alloc(A, B);
}

template <class T>
inline auto Hydrostatic(const T& A)
{
    return detail::equiv_impl<T>::hydrostatic_alloc(A);
}

template <class T, class U>
inline void deviatoric(const T& A, U& B)
{
    return detail::equiv_impl<T>::deviatoric_no_alloc(A, B);
}

template <class T>
inline auto Deviatoric(const T& A)
{
    return detail::equiv_impl<T>::deviatoric_alloc(A);
}

template <class T, class U>
inline void equivalent_deviatoric(const T& A, U& B)
{
    return detail::equiv_impl<T>::equivalent_deviatoric_no_alloc(A, B);
}

template <class T>
inline auto Equivalent_deviatoric(const T& A)
{
    return detail::equiv_impl<T>::equivalent_deviatoric_alloc(A);
}

namespace pointer {

    namespace detail {

        // ----------------------------------------------------------------------------
        // Numerical diagonalization of 3x3 matrices
        // Copyright (C) 2006  Joachim Kopp
        // ----------------------------------------------------------------------------
        // This library is free software; you can redistribute it and/or
        // modify it under the terms of the GNU Lesser General Public
        // License as published by the Free Software Foundation; either
        // version 2.1 of the License, or (at your option) any later version.
        //
        // This library is distributed in the hope that it will be useful,
        // but WITHOUT ANY WARRANTY; without even the implied warranty of
        // MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
        // Lesser General Public License for more details.
        //
        // You should have received a copy of the GNU Lesser General Public
        // License along with this library; if not, write to the Free Software
        // Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
        // ----------------------------------------------------------------------------

        // ----------------------------------------------------------------------------
        inline int dsyevj3(double A[3][3], double Q[3][3], double w[3])
        // ----------------------------------------------------------------------------
        // Calculates the eigenvalues and normalized eigenvectors of a symmetric 3x3
        // matrix A using the Jacobi algorithm.
        // The upper triangular part of A is destroyed during the calculation,
        // the diagonal elements are read but not destroyed, and the lower
        // triangular elements are not referenced at all.
        // ----------------------------------------------------------------------------
        // Parameters:
        //   A: The symmetric input matrix
        //   Q: Storage buffer for eigenvectors
        //   w: Storage buffer for eigenvalues
        // ----------------------------------------------------------------------------
        // Return value:
        //   0: Success
        //  -1: Error (no convergence)
        // ----------------------------------------------------------------------------
        {
            const int n = 3;
            double sd, so;          // Sums of diagonal resp. off-diagonal elements
            double s, c, t;         // sin(phi), cos(phi), tan(phi) and temporary storage
            double g, h, z, theta;  // More temporary storage
            double thresh;

            // Initialize Q to the identity matrix
            for (int i = 0; i < n; i++) {
                Q[i][i] = 1.0;
                for (int j = 0; j < i; j++)
                    Q[i][j] = Q[j][i] = 0.0;
            }

            // Initialize w to diag(A)
            for (int i = 0; i < n; i++)
                w[i] = A[i][i];

            // Calculate SQR(tr(A))
            sd = 0.0;
            for (int i = 0; i < n; i++)
                sd += fabs(w[i]);
            sd = sd * sd;

            // Main iteration loop
            for (int nIter = 0; nIter < 50; nIter++) {
                // Test for convergence
                so = 0.0;
                for (int p = 0; p < n; p++)
                    for (int q = p + 1; q < n; q++)
                        so += fabs(A[p][q]);
                if (so == 0.0)
                  return 0;

                if (nIter < 4)
                    thresh = 0.2 * so / (n * n);
                else
                    thresh = 0.0;

                // Do sweep
                for (int p = 0; p < n; p++)
                    for (int q = p + 1; q < n; q++) {
                        g = 100.0 * fabs(A[p][q]);
                        if (nIter > 4 && fabs(w[p]) + g == fabs(w[p]) && fabs(w[q]) + g == fabs(w[q])) {
                            A[p][q] = 0.0;
                        }
                        else if (fabs(A[p][q]) > thresh) {
                            // Calculate Jacobi transformation
                            h = w[q] - w[p];
                            if (fabs(h) + g == fabs(h)) {
                                t = A[p][q] / h;
                            }
                            else {
                                theta = 0.5 * h / A[p][q];
                                if (theta < 0.0)
                                    t = -1.0 / (sqrt(1.0 + theta * theta) - theta);
                                else
                                    t = 1.0 / (sqrt(1.0 + theta * theta) + theta);
                            }
                            c = 1.0 / sqrt(1.0 + t * t);
                            s = t * c;
                            z = t * A[p][q];

                            // Apply Jacobi transformation
                            A[p][q] = 0.0;
                            w[p] -= z;
                            w[q] += z;
                            for (int r = 0; r < p; r++) {
                                t = A[r][p];
                                A[r][p] = c * t - s * A[r][q];
                                A[r][q] = s * t + c * A[r][q];
                            }
                            for (int r = p + 1; r < q; r++) {
                                t = A[p][r];
                                A[p][r] = c * t - s * A[r][q];
                                A[r][q] = s * t + c * A[r][q];
                            }
                            for (int r = q + 1; r < n; r++) {
                                t = A[p][r];
                                A[p][r] = c * t - s * A[q][r];
                                A[q][r] = s * t + c * A[q][r];
                            }

                            // Update eigenvectors
                            for (int r = 0; r < n; r++) {
                                t = Q[r][p];
                                Q[r][p] = c * t - s * Q[r][q];
                                Q[r][q] = s * t + c * Q[r][q];
                            }
                        }
                    }
            }

            return -1;
        }
        // ----------------------------------------------------------------------------

    } // namespace detail

    template <class T>
    inline void O2(T* ret)
    {
        std::fill(ret, ret + 9, T(0));
    }

    template <class T>
    inline void O4(T* ret)
    {
        std::fill(ret, ret + 81, T(0));
    }

    template <class T>
    inline void I2(T* ret)
    {
        ret[0] = 1.0;
        ret[1] = 0.0;
        ret[2] = 0.0;
        ret[3] = 0.0;
        ret[4] = 1.0;
        ret[5] = 0.0;
        ret[6] = 0.0;
        ret[7] = 0.0;
        ret[8] = 1.0;
    }

    template <class T>
    inline void II(T* ret)
    {
        std::fill(ret, ret + 81, T(0));

        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                for (size_t k = 0; k < 3; ++k) {
                    for (size_t l = 0; l < 3; ++l) {
                        if (i == j && k == l) {
                            ret[i * 27 + j * 9 + k * 3 + l] = 1.0;
                        }
                    }
                }
            }
        }
    }

    template <class T>
    inline void I4(T* ret)
    {
        std::fill(ret, ret + 81, T(0));

        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                for (size_t k = 0; k < 3; ++k) {
                    for (size_t l = 0; l < 3; ++l) {
                        if (i == l && j == k) {
                            ret[i * 27 + j * 9 + k * 3 + l] = 1.0;
                        }
                    }
                }
            }
        }
    }

    template <class T>
    inline void I4rt(T* ret)
    {
        std::fill(ret, ret + 81, T(0));

        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                for (size_t k = 0; k < 3; ++k) {
                    for (size_t l = 0; l < 3; ++l) {
                        if (i == k && j == l) {
                            ret[i * 27 + j * 9 + k * 3 + l] = 1.0;
                        }
                    }
                }
            }
        }
    }

    template <class T>
    inline void I4s(T* ret)
    {
        I4(ret);

        std::array<double, 81> i4rt;
        I4rt(&i4rt[0]);

        std::transform(ret, ret + 81, &i4rt[0], ret, std::plus<T>());

        std::transform(ret, ret + 81, ret,
            std::bind(std::multiplies<T>(), std::placeholders::_1, 0.5));
    }

    template <class T>
    inline void I4d(T* ret)
    {
        I4s(ret);

        std::array<double, 81> ii;
        II(&ii[0]);

        std::transform(&ii[0], &ii[0] + 81, &ii[0],
            std::bind(std::divides<T>(), std::placeholders::_1, 3.0));

        std::transform(ret, ret + 81, &ii[0], ret, std::minus<T>());
    }

    template <class T>
    inline auto trace(const T* A)
    {
        return A[0] + A[4] + A[8];
    }

    template <class T>
    inline auto det(const T* A)
    {
        return (A[0] * A[4] * A[8] + A[1] * A[5] * A[6] + A[2] * A[3] * A[7]) -
               (A[2] * A[4] * A[6] + A[1] * A[3] * A[8] + A[0] * A[5] * A[7]);
    }

    template <class S, class T>
    inline auto inv(const S* A, T* ret)
    {
        auto D = det(A);
        ret[0] = (A[4] * A[8] - A[5] * A[7]) / D;
        ret[1] = (A[2] * A[7] - A[1] * A[8]) / D;
        ret[2] = (A[1] * A[5] - A[2] * A[4]) / D;
        ret[3] = (A[5] * A[6] - A[3] * A[8]) / D;
        ret[4] = (A[0] * A[8] - A[2] * A[6]) / D;
        ret[5] = (A[2] * A[3] - A[0] * A[5]) / D;
        ret[6] = (A[3] * A[7] - A[4] * A[6]) / D;
        ret[7] = (A[1] * A[6] - A[0] * A[7]) / D;
        ret[8] = (A[0] * A[4] - A[1] * A[3]) / D;
        return D;
    }

    template <class S, class T>
    inline auto hydrostatic_deviatoric(const S* A, T* ret)
    {
        auto m = trace(A) / T(3);
        ret[0] = A[0] - m;
        ret[1] = A[1];
        ret[2] = A[2];
        ret[3] = A[3];
        ret[4] = A[4] - m;
        ret[5] = A[5];
        ret[6] = A[6];
        ret[7] = A[7];
        ret[8] = A[8] - m;
        return m;
    }

    template <class T>
    inline auto deviatoric_ddot_deviatoric(const T* A)
    {
        auto m = trace(A) / T(3);
        return (A[0] - m) * (A[0] - m)
             + (A[4] - m) * (A[4] - m)
             + (A[8] - m) * (A[8] - m)
             + T(2) * A[1] * A[3]
             + T(2) * A[2] * A[6]
             + T(2) * A[5] * A[7];
    }

    template <class S, class T>
    inline auto A2_ddot_B2(const S* A, const T* B)
    {
        return A[0] * B[0]
             + A[4] * B[4]
             + A[8] * B[8]
             + A[1] * B[3]
             + A[2] * B[6]
             + A[3] * B[1]
             + A[5] * B[7]
             + A[6] * B[2]
             + A[7] * B[5];
    }

    template <class S, class T>
    inline auto A2s_ddot_B2s(const S* A, const T* B)
    {
        return A[0] * B[0]
             + A[4] * B[4]
             + A[8] * B[8]
             + T(2) * A[1] * B[1]
             + T(2) * A[2] * B[2]
             + T(2) * A[5] * B[5];
    }

    template <class R, class S, class T>
    inline void A2_dyadic_B2(const R* A, const S* B, T* ret)
    {
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                for (size_t k = 0; k < 3; ++k) {
                    for (size_t l = 0; l < 3; ++l) {
                        ret[i * 27 + j * 9 + k * 3 + l] = A[i * 3 + j] * B[k * 3 + l];
                    }
                }
            }
        }
    }

    template <class R, class S, class T>
    inline void A4_dot_B2(const R* A, const S* B, T* ret)
    {
        std::fill(ret, ret + 81, T(0));

        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                for (size_t k = 0; k < 3; ++k) {
                    for (size_t l = 0; l < 3; ++l) {
                        for (size_t m = 0; m < 3; ++m) {
                            ret[i * 27 + j * 9 + k * 3 + m]
                                += A[i * 27 + j * 9 + k * 3 + l]
                                * B[l * 3 + m];
                        }
                    }
                }
            }
        }
    }

    template <class R, class S, class T>
    inline void A2_dot_B2(const R* A, const S* B, T* ret)
    {
        std::fill(ret, ret + 9, T(0));

        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                for (size_t k = 0; k < 3; ++k) {
                    ret[i * 3 + k] += A[i * 3 + j] * B[j * 3 + k];
                }
            }
        }
    }

    template <class S, class T>
    inline void A2_dot_A2T(const S* A, T* ret)
    {
        ret[0] = A[0] * A[0] + A[1] * A[1] + A[2] * A[2];
        ret[1] = A[0] * A[3] + A[1] * A[4] + A[2] * A[5];
        ret[2] = A[0] * A[6] + A[1] * A[7] + A[2] * A[8];
        ret[4] = A[3] * A[3] + A[4] * A[4] + A[5] * A[5];
        ret[5] = A[3] * A[6] + A[4] * A[7] + A[5] * A[8];
        ret[8] = A[6] * A[6] + A[7] * A[7] + A[8] * A[8];
        ret[3] = ret[1];
        ret[6] = ret[2];
        ret[7] = ret[5];
    }

    template <class R, class S, class T>
    inline void A4_ddot_B2(const R* A, const S* B, T* ret)
    {
        std::fill(ret, ret + 9, T(0));

        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 3; k++) {
                    for (size_t l = 0; l < 3; l++) {
                        ret[i * 3 + j] += A[i * 27 + j * 9 + k * 3 + l] * B[l * 3 + k];
                    }
                }
            }
        }
    }

    template <class R, class S, class T, class U>
    inline void A4_ddot_B4_ddot_C4(const R* A, const S* B, const T* C, U* ret)
    {
        std::fill(ret, ret + 81, T(0));

        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                for (size_t k = 0; k < 3; ++k) {
                    for (size_t l = 0; l < 3; ++l) {
                        for (size_t m = 0; m < 3; ++m) {
                            for (size_t n = 0; n < 3; ++n) {
                                for (size_t o = 0; o < 3; ++o) {
                                    for (size_t p = 0; p < 3; ++p) {
                                        ret[i * 27 + j * 9 + o * 3 + p]
                                            += A[i * 27 + j * 9 + k * 3 + l]
                                            * B[l * 27 + k * 9 + m * 3 + n]
                                            * C[n * 27 + m * 9 + o * 3 + p];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    template <class R, class S, class T, class U>
    inline void A2_dot_B2_dot_C2T(const R* A, const S* B, const T* C, U* ret)
    {
        std::fill(ret, ret + 9, T(0));

        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                for (size_t h = 0; h < 3; ++h) {
                    for (size_t l = 0; l < 3; ++l) {
                        ret[i * 3 + l] += A[i * 3 + j] * B[j * 3 + h] * C[l * 3 + h];
                    }
                }
            }
        }
    }

    template <class U, class V, class W>
    void eigs(const U* A, V* vec, W* val)
    {
        double a[3][3];
        double Q[3][3];
        double w[3];

        std::copy(&A[0], &A[0] + 9, &a[0][0]);

        // use the 'Jacobi' algorithm, which is accurate but not very fast
        // (in practice the faster 'hybrid' "dsyevh3" is too inaccurate for finite elements)
        auto succes = detail::dsyevj3(a, Q, w);
        (void)(succes);
        GMATTENSOR_ASSERT(succes == 0);

        std::copy(&Q[0][0], &Q[0][0] + 3 * 3, vec);
        std::copy(&w[0], &w[0] + 3, val);
    }

    template <class U, class V, class W>
    void from_eigs(const U* vec, const V* val, W* ret)
    {
        ret[0] = val[0] * vec[0] * vec[0] + val[1] * vec[1] * vec[1] + val[2] * vec[2] * vec[2];
        ret[1] = val[0] * vec[0] * vec[3] + val[1] * vec[1] * vec[4] + val[2] * vec[2] * vec[5];
        ret[2] = val[0] * vec[0] * vec[6] + val[1] * vec[1] * vec[7] + val[2] * vec[2] * vec[8];
        ret[4] = val[0] * vec[3] * vec[3] + val[1] * vec[4] * vec[4] + val[2] * vec[5] * vec[5];
        ret[5] = val[0] * vec[3] * vec[6] + val[1] * vec[4] * vec[7] + val[2] * vec[5] * vec[8];
        ret[8] = val[0] * vec[6] * vec[6] + val[1] * vec[7] * vec[7] + val[2] * vec[8] * vec[8];
        ret[3] = ret[1];
        ret[6] = ret[2];
        ret[7] = ret[5];
    }

} // namespace pointer

} // namespace Cartesian3d
} // namespace GMatTensor

#endif
