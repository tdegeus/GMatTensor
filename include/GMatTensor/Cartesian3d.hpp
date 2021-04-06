/**
Implementation of Cartesian3d.h

\file GMatTensor/Cartesian3d.hpp
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the MIT License.
*/

#ifndef GMATTENSOR_CARTESIAN3D_HPP
#define GMATTENSOR_CARTESIAN3D_HPP

#include "Cartesian3d.h"
#include "detail.hpp"

namespace GMatTensor {
namespace Cartesian3d {

namespace detail {
    using GMatTensor::detail::impl_A2;
    using GMatTensor::detail::impl_A4;
} // namespace detail

inline xt::xtensor<double, 2> Random2()
{
    xt::xtensor<double, 2> ret = xt::random::randn<double>({3, 3});
    return ret;
}

inline xt::xtensor<double, 4> Random4()
{
    xt::xtensor<double, 4> ret = xt::random::randn<double>({3, 3, 3, 3});
    return ret;
}

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
    xt::xtensor<double, 2> ret = xt::empty<double>({3, 3});
    pointer::I2(ret.data());
    return ret;
}

inline xt::xtensor<double, 4> II()
{
    xt::xtensor<double, 4> ret = xt::empty<double>({3, 3, 3, 3});
    pointer::II(ret.data());
    return ret;
}

inline xt::xtensor<double, 4> I4()
{
    xt::xtensor<double, 4> ret = xt::empty<double>({3, 3, 3, 3});
    pointer::I4(ret.data());
    return ret;
}

inline xt::xtensor<double, 4> I4rt()
{
    xt::xtensor<double, 4> ret = xt::empty<double>({3, 3, 3, 3});
    pointer::I4rt(ret.data());
    return ret;
}

inline xt::xtensor<double, 4> I4s()
{
    xt::xtensor<double, 4> ret = xt::empty<double>({3, 3, 3, 3});
    pointer::I4s(ret.data());
    return ret;
}

inline xt::xtensor<double, 4> I4d()
{
    xt::xtensor<double, 4> ret = xt::empty<double>({3, 3, 3, 3});
    pointer::I4d(ret.data());
    return ret;
}

template <class T, class R>
inline void trace(const T& A, R& ret)
{
    return detail::impl_A2<T, 3>::ret0(A, ret,
        [](const auto& a){ return pointer::Trace(a); });
}

template <class T>
inline auto Trace(const T& A)
{
    return detail::impl_A2<T, 3>::ret0(A,
        [](const auto& a){ return pointer::Trace(a); });
}

template <class T, class R>
inline void hydrostatic(const T& A, R& ret)
{
    return detail::impl_A2<T, 3>::ret0(A, ret,
        [](const auto& a){ return pointer::Hydrostatic(a); });
}

template <class T>
inline auto Hydrostatic(const T& A)
{
    return detail::impl_A2<T, 3>::ret0(A,
        [](const auto& a){ return pointer::Hydrostatic(a); });
}

template <class T, class R>
inline void det(const T& A, R& ret)
{
    return detail::impl_A2<T, 3>::ret0(A, ret,
        [](const auto& a){ return pointer::Det(a); });
}

template <class T>
inline auto Det(const T& A)
{
    return detail::impl_A2<T, 3>::ret0(A,
        [](const auto& a){ return pointer::Det(a); });
}

template <class T, class R>
inline void A2_ddot_B2(const T& A, const T& B, R& ret)
{
    return detail::impl_A2<T, 3>::B2_ret0(A, B, ret,
        [](const auto& a, const auto& b){ return pointer::A2_ddot_B2(a, b); });
}

template <class T>
inline auto A2_ddot_B2(const T& A, const T& B)
{
    return detail::impl_A2<T, 3>::B2_ret0(A, B,
        [](const auto& a, const auto& b){ return pointer::A2_ddot_B2(a, b); });
}

template <class T, class R>
inline void A2s_ddot_B2s(const T& A, const T& B, R& ret)
{
    return detail::impl_A2<T, 3>::B2_ret0(A, B, ret,
        [](const auto& a, const auto& b){ return pointer::A2s_ddot_B2s(a, b); });
}

template <class T>
inline auto A2s_ddot_B2s(const T& A, const T& B)
{
    return detail::impl_A2<T, 3>::B2_ret0(A, B,
        [](const auto& a, const auto& b){ return pointer::A2s_ddot_B2s(a, b); });
}

template <class T, class R>
inline void norm_deviatoric(const T& A, R& ret)
{
    return detail::impl_A2<T, 3>::ret0(A, ret,
        [](const auto& a){ return pointer::Norm_deviatoric(a); });
}

template <class T>
inline auto Norm_deviatoric(const T& A)
{
    return detail::impl_A2<T, 3>::ret0(A,
        [](const auto& a){ return pointer::Norm_deviatoric(a); });
}

template <class T, class R>
inline void deviatoric(const T& A, R& ret)
{
    return detail::impl_A2<T, 3>::ret2(A, ret,
        [](const auto& a, const auto& r){ return pointer::Hydrostatic_deviatoric(a, r); });
}

template <class T>
inline auto Deviatoric(const T& A)
{
    return detail::impl_A2<T, 3>::ret2(A,
        [](const auto& a, const auto& r){ return pointer::Hydrostatic_deviatoric(a, r); });
}

template <class T, class R>
inline void sym(const T& A, R& ret)
{
    return detail::impl_A2<T, 3>::ret2(A, ret,
        [](const auto& a, const auto& r){ return pointer::sym(a, r); });
}

template <class T>
inline auto Sym(const T& A)
{
    return detail::impl_A2<T, 3>::ret2(A,
        [](const auto& a, const auto& r){ return pointer::sym(a, r); });
}

template <class T, class R>
inline void inv(const T& A, R& ret)
{
    return detail::impl_A2<T, 3>::ret2(A, ret,
        [](const auto& a, const auto& r){ return pointer::Inv(a, r); });
}

template <class T>
inline auto Inv(const T& A)
{
    return detail::impl_A2<T, 3>::ret2(A,
        [](const auto& a, const auto& r){ return pointer::Inv(a, r); });
}

template <class T, class R>
inline void logs(const T& A, R& ret)
{
    return detail::impl_A2<T, 3>::ret2(A, ret,
        [](const auto& a, const auto& r){ return pointer::logs(a, r); });
}

template <class T>
inline auto Logs(const T& A)
{
    return detail::impl_A2<T, 3>::ret2(A,
        [](const auto& a, const auto& r){ return pointer::logs(a, r); });
}

template <class T, class R>
inline void A2_dot_A2T(const T& A, R& ret)
{
    return detail::impl_A2<T, 3>::ret2(A, ret,
        [](const auto& a, const auto& r){ return pointer::A2_dot_A2T(a, r); });
}

template <class T>
inline auto A2_dot_A2T(const T& A)
{
    return detail::impl_A2<T, 3>::ret2(A,
        [](const auto& a, const auto& r){ return pointer::A2_dot_A2T(a, r); });
}

template <class T, class R>
inline void A2_dot_B2(const T& A, const T& B, R& ret)
{
    return detail::impl_A2<T, 3>::B2_ret2(A, B, ret,
        [](const auto& a, const auto& b, const auto& r){ return pointer::A2_dot_B2(a, b, r); });
}

template <class T>
inline auto A2_dot_B2(const T& A, const T& B)
{
    return detail::impl_A2<T, 3>::B2_ret2(A, B,
        [](const auto& a, const auto& b, const auto& r){ return pointer::A2_dot_B2(a, b, r); });
}

template <class T, class R>
inline void A2_dyadic_B2(const T& A, const T& B, R& ret)
{
    return detail::impl_A2<T, 3>::B2_ret4(A, B, ret,
        [](const auto& a, const auto& b, const auto& r){ return pointer::A2_dyadic_B2(a, b, r); });
}

template <class T>
inline auto A2_dyadic_B2(const T& A, const T& B)
{
    return detail::impl_A2<T, 3>::B2_ret4(A, B,
        [](const auto& a, const auto& b, const auto& r){ return pointer::A2_dyadic_B2(a, b, r); });
}

template <class T, class U, class R>
inline void A4_ddot_B2(const T& A, const U& B, R& ret)
{
    return detail::impl_A4<T, 3>::B2_ret2(A, B, ret,
        [](const auto& a, const auto& b, const auto& r){ return pointer::A4_ddot_B2(a, b, r); });
}

template <class T, class U>
inline auto A4_ddot_B2(const T& A, const U& B)
{
    return detail::impl_A4<T, 3>::B2_ret2(A, B,
        [](const auto& a, const auto& b, const auto& r){ return pointer::A4_ddot_B2(a, b, r); });
}

template <class T, class U, class R>
inline auto A4_dot_B2(const T& A, const U& B, R& ret)
{
    return detail::impl_A4<T, 3>::B2_ret4(A, B, ret,
        [](const auto& a, const auto& b, const auto& r){ return pointer::A4_dot_B2(a, b, r); });
}

template <class T, class U>
inline auto A4_dot_B2(const T& A, const U& B)
{
    return detail::impl_A4<T, 3>::B2_ret4(A, B,
        [](const auto& a, const auto& b, const auto& r){ return pointer::A4_dot_B2(a, b, r); });
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
    inline T Trace(const T* A)
    {
        return A[0] + A[4] + A[8];
    }

    template <class T>
    inline T Hydrostatic(const T* A)
    {
        return Trace(A) / T(3);
    }

    template <class T>
    inline T Det(const T* A)
    {
        return (A[0] * A[4] * A[8] + A[1] * A[5] * A[6] + A[2] * A[3] * A[7]) -
               (A[2] * A[4] * A[6] + A[1] * A[3] * A[8] + A[0] * A[5] * A[7]);
    }

    template <class T>
    inline void sym(const T* A, T* ret)
    {
        ret[0] = A[0];
        ret[1] = 0.5 * (A[1] + A[3]);
        ret[2] = 0.5 * (A[2] + A[6]);
        ret[3] = ret[1];
        ret[4] = A[4];
        ret[5] = 0.5 * (A[5] + A[7]);
        ret[6] = ret[2];
        ret[7] = ret[5];
        ret[8] = A[8];
    }

    template <class T>
    inline T Inv(const T* A, T* ret)
    {
        T D = Det(A);
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

    template <class T>
    inline T Hydrostatic_deviatoric(const T* A, T* ret)
    {
        T m = Hydrostatic(A);
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
    inline T Deviatoric_ddot_deviatoric(const T* A)
    {
        T m = Hydrostatic(A);
        return (A[0] - m) * (A[0] - m)
             + (A[4] - m) * (A[4] - m)
             + (A[8] - m) * (A[8] - m)
             + T(2) * A[1] * A[3]
             + T(2) * A[2] * A[6]
             + T(2) * A[5] * A[7];
    }

    template <class T>
    inline T Norm_deviatoric(const T* A)
    {
        return std::sqrt(Deviatoric_ddot_deviatoric(A));
    }

    template <class T>
    inline T A2_ddot_B2(const T* A, const T* B)
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

    template <class T>
    inline T A2s_ddot_B2s(const T* A, const T* B)
    {
        return A[0] * B[0]
             + A[4] * B[4]
             + A[8] * B[8]
             + T(2) * A[1] * B[1]
             + T(2) * A[2] * B[2]
             + T(2) * A[5] * B[5];
    }

    template <class T>
    inline void A2_dyadic_B2(const T* A, const T* B, T* ret)
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

    template <class T>
    inline void A4_dot_B2(const T* A, const T* B, T* ret)
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

    template <class T>
    inline void A2_dot_B2(const T* A, const T* B, T* ret)
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

    template <class T>
    inline void A2_dot_A2T(const T* A, T* ret)
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

    template <class T>
    inline void A4_ddot_B2(const T* A, const T* B, T* ret)
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

    template <class T>
    inline void A4_ddot_B4_ddot_C4(const T* A, const T* B, const T* C, T* ret)
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

    template <class T>
    inline void A2_dot_B2_dot_C2T(const T* A, const T* B, const T* C, T* ret)
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

    template <class T>
    void eigs(const T* A, T* vec, T* val)
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

    template <class T>
    void from_eigs(const T* vec, const T* val, T* ret)
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

    template <class T>
    inline void logs(const T* A, T* ret)
    {
        std::array<double, 3> val;
        std::array<double, 9> vec;
        eigs(&A[0], &vec[0], &val[0]);
        for (size_t j = 0; j < 3; ++j) {
            val[j] = std::log(val[j]);
        }
        from_eigs(&vec[0], &val[0], &ret[0]);
    }

} // namespace pointer

} // namespace Cartesian3d
} // namespace GMatTensor

#endif
