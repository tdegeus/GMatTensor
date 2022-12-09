/**
 * @file
 * @copyright Copyright 2020. Tom de Geus. All rights reserved.
 * @license This project is released under the MIT License.
 */

#ifndef GMATTENSOR_CARTESIAN3D_H
#define GMATTENSOR_CARTESIAN3D_H

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
 * Tensors and tensor operations for a(n) array of 3d tensors of different rank,
 * defined in a Cartesian coordinate system.
 */
namespace Cartesian3d {

/**
 * API for individual tensors with pointer-only input.
 * No arrays of tensors are allowed, hence the input is fixed to:
 *
 * -   Second order tensors, ``size = 3 * 3 = 9``.
 *     Storage convention ``(xx, xy, xz, yx, yy, yz, zx, zy, zz)``.
 *
 * -   Fourth order tensors, ``size = 3 * 3 * 3 * 3 = 81``.
 */
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
    double sd, so; // Sums of diagonal resp. off-diagonal elements
    double s, c, t; // sin(phi), cos(phi), tan(phi) and temporary storage
    double g, h, z, theta; // More temporary storage
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

/**
 * See Cartesian3d::O2()
 *
 * @param ret output 2nd order tensor
 */
template <class T>
inline void O2(T* ret)
{
    std::fill(ret, ret + 9, T(0));
}

/**
 * See Cartesian3d::O4()
 *
 * @param ret output 2nd order tensor
 */
template <class T>
inline void O4(T* ret)
{
    std::fill(ret, ret + 81, T(0));
}

/**
 * See Cartesian3d::I2()
 *
 * @param ret output 2nd order tensor
 */
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

/**
 * See Cartesian3d::II()
 *
 * @param ret output 2nd order tensor
 */
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

/**
 * See Cartesian3d::I4()
 *
 * @param ret output 2nd order tensor
 */
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

/**
 * See Cartesian3d::I4rt()
 *
 * @param ret output 2nd order tensor
 */
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

/**
 * See Cartesian3d::I4s()
 *
 * @param ret output 2nd order tensor
 */
template <class T>
inline void I4s(T* ret)
{
    I4(ret);

    std::array<double, 81> i4rt;
    I4rt(&i4rt[0]);

    std::transform(ret, ret + 81, &i4rt[0], ret, std::plus<T>());

    std::transform(ret, ret + 81, ret, std::bind(std::multiplies<T>(), std::placeholders::_1, 0.5));
}

/**
 * See Cartesian3d::I4d()
 *
 * @param ret output 2nd order tensor
 */
template <class T>
inline void I4d(T* ret)
{
    I4s(ret);

    std::array<double, 81> ii;
    II(&ii[0]);

    std::transform(
        &ii[0], &ii[0] + 81, &ii[0], std::bind(std::divides<T>(), std::placeholders::_1, 3.0));

    std::transform(ret, ret + 81, &ii[0], ret, std::minus<T>());
}

/**
 * See Cartesian3d::Trace()
 *
 * @param A 2nd order tensor
 * @return scalar
 */
template <class T>
inline T Trace(const T* A)
{
    return A[0] + A[4] + A[8];
}

/**
 * See Cartesian3d::Hydrostatic()
 *
 * @param A 2nd order tensor
 * @return scalar
 */
template <class T>
inline T Hydrostatic(const T* A)
{
    return Trace(A) / T(3);
}

/**
 * See Cartesian3d::Det()
 *
 * @param A 2nd order tensor
 * @return scalar
 */
template <class T>
inline T Det(const T* A)
{
    return (A[0] * A[4] * A[8] + A[1] * A[5] * A[6] + A[2] * A[3] * A[7]) -
           (A[2] * A[4] * A[6] + A[1] * A[3] * A[8] + A[0] * A[5] * A[7]);
}

/**
 * See Cartesian3d::Sym()
 *
 * @param A 2nd order tensor
 * @param ret 2nd order tensor, may be the same pointer as ``A``
 */
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

/**
 * See Cartesian3d::Inv(), returns Cartesian3d::Det()
 *
 * @param A 2nd order tensor
 * @param ret 2nd order tensor
 * @return scalar
 */
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

/**
 * Returns Cartesian3d::Hydrostatic() and computes Cartesian3d::Deviatoric()
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
    ret[3] = A[3];
    ret[4] = A[4] - m;
    ret[5] = A[5];
    ret[6] = A[6];
    ret[7] = A[7];
    ret[8] = A[8] - m;
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
    return (A[0] - m) * (A[0] - m) + (A[4] - m) * (A[4] - m) + (A[8] - m) * (A[8] - m) +
           T(2) * A[1] * A[3] + T(2) * A[2] * A[6] + T(2) * A[5] * A[7];
}

/**
 * See Cartesian3d::Norm_deviatoric()
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
 * See Cartesian3d::A2_ddot_B2()
 *
 * @param A 2nd order tensor
 * @param B 2nd order tensor
 * @return scalar
 */
template <class T>
inline T A2_ddot_B2(const T* A, const T* B)
{
    return A[0] * B[0] + A[4] * B[4] + A[8] * B[8] + A[1] * B[3] + A[2] * B[6] + A[3] * B[1] +
           A[5] * B[7] + A[6] * B[2] + A[7] * B[5];
}

/**
 * See Cartesian3d::A2s_ddot_B2s()
 *
 * @param A 2nd order tensor
 * @param B 2nd order tensor
 * @return scalar
 */
template <class T>
inline T A2s_ddot_B2s(const T* A, const T* B)
{
    return A[0] * B[0] + A[4] * B[4] + A[8] * B[8] + T(2) * A[1] * B[1] + T(2) * A[2] * B[2] +
           T(2) * A[5] * B[5];
}

/**
 * See Cartesian3d::A2_dyadic_B2()
 *
 * @param A 2nd order tensor
 * @param B 2nd order tensor
 * @param ret output 4th order tensor
 */
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

/**
 * See Cartesian3d::A4_dot_B2()
 *
 * @param A 4th order tensor
 * @param B 2nd order tensor
 * @param ret output 4th order tensor
 */
template <class T>
inline void A4_dot_B2(const T* A, const T* B, T* ret)
{
    std::fill(ret, ret + 81, T(0));

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 3; ++k) {
                for (size_t l = 0; l < 3; ++l) {
                    for (size_t m = 0; m < 3; ++m) {
                        ret[i * 27 + j * 9 + k * 3 + m] +=
                            A[i * 27 + j * 9 + k * 3 + l] * B[l * 3 + m];
                    }
                }
            }
        }
    }
}

/**
 * See Cartesian3d::A2_dot_B2()
 *
 * @param A 2nd order tensor
 * @param B 2nd order tensor
 * @param ret output 2nd order tensor
 */
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

/**
 * See Cartesian3d::A2_dot_A2T()
 *
 * @param A 2nd order tensor
 * @param ret output 2nd order tensor
 */
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

/**
 * See Cartesian3d::A4_ddot_B2()
 *
 * @param A 4th order tensor
 * @param B 2nd order tensor
 * @param ret output 2nd order tensor
 */
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

/**
 * Product
 *
 * \f$ A : B : C \f$
 *
 * or in index notation
 *
 * \f$ D_{ijop} = A_{ijkl} B_{lkmn} C_{nmop} \f$
 *
 * @param A 4th order tensor
 * @param B 4th order tensor
 * @param C 4th order tensor
 * @param ret output 4th order tensor
 */
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
                                    ret[i * 27 + j * 9 + o * 3 + p] +=
                                        A[i * 27 + j * 9 + k * 3 + l] *
                                        B[l * 27 + k * 9 + m * 3 + n] *
                                        C[n * 27 + m * 9 + o * 3 + p];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/**
 * Product
 *
 * \f$ A \cdot B \cdot C^T \f$
 *
 * or in index notation
 *
 * \f$ D_{il} = A_{ij} B_{jk} C_{lk} \f$
 *
 * @param A 2nd order tensor
 * @param B 2nd order tensor
 * @param C 2nd order tensor
 * @param ret output 2nd order tensor
 */
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

/**
 * Get eigenvalues/-vectors such that
 *
 * \f$ A_{ij} = \lambda^a v^a_i v^a_j \f$
 *
 * Symmetric tensors only, no assertion.
 *
 * @param A 2nd order tensor
 * @param vec eigenvectors (storage as 2nd order tensor), \f$ v^a_i \f$ = ``vec[i, a]``
 * @param val eigenvalues (storage as vector), \f$ \lambda^a \f$ = ``val[a]``
 *
 */
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

/**
 * Reconstruct tensor from eigenvalues/-vectors (reverse operation of eigs())
 * Symmetric tensors only, no assertion.
 *
 * @param vec eigenvectors (storage as 2nd order tensor), \f$ v^a_i \f$ = ``vec[i, a]``
 * @param val eigenvalues (storage as vector), \f$ \lambda^a \f$ = ``val[a]``
 * @param ret 2nd order tensor
 */
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

/**
 * See Cartesian3d::Logs()
 *
 * @param A 2nd order tensor
 * @param ret output 2nd order tensor
 */
template <class T>
void logs(const T* A, T* ret)
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

/**
 * Random 2nd-order tensor (for example for use in testing).
 *
 * @return [3, 3] array.
 */
inline array_type::tensor<double, 2> Random2()
{
    array_type::tensor<double, 2> ret = xt::random::randn<double>({3, 3});
    return ret;
}

/**
 * Random 4th-order tensor (for example for use in testing).
 *
 * @return [3, 3, 3, 3] array.
 */
inline array_type::tensor<double, 4> Random4()
{
    array_type::tensor<double, 4> ret = xt::random::randn<double>({3, 3, 3, 3});
    return ret;
}

/**
 * 2nd-order null tensor (all components equal to zero).
 *
 * @return [3, 3] array.
 */
inline array_type::tensor<double, 2> O2()
{
    return xt::zeros<double>({3, 3});
}

/**
 * 4th-order null tensor (all components equal to zero).
 *
 * @return [3, 3, 3, 3] array.
 */
inline array_type::tensor<double, 4> O4()
{
    return xt::zeros<double>({3, 3, 3, 3});
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
 * @return [3, 3] array.
 */
inline array_type::tensor<double, 2> I2()
{
    array_type::tensor<double, 2> ret = xt::empty<double>({3, 3});
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
 * @return [3, 3, 3, 3] array.
 */
inline array_type::tensor<double, 4> II()
{
    array_type::tensor<double, 4> ret = xt::empty<double>({3, 3, 3, 3});
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
 * @return [3, 3, 3, 3] array.
 */
inline array_type::tensor<double, 4> I4()
{
    array_type::tensor<double, 4> ret = xt::empty<double>({3, 3, 3, 3});
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
 * @return [3, 3, 3, 3] array.
 */
inline array_type::tensor<double, 4> I4rt()
{
    array_type::tensor<double, 4> ret = xt::empty<double>({3, 3, 3, 3});
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
 * @return [3, 3, 3, 3] array.
 */
inline array_type::tensor<double, 4> I4s()
{
    array_type::tensor<double, 4> ret = xt::empty<double>({3, 3, 3, 3});
    pointer::I4s(ret.data());
    return ret;
}

/**
 * Fourth order deviatoric projection.
 * By definition
 *
 *     I = I4s() - 1.0 / 3.0 * II()
 *
 * such that
 *
 * \f$ I : A = sym(A) - tr(A) / 3 \f$
 *
 * See A4_ddot_B2(), Deviatoric().
 *
 * @return [3, 3, 3, 3] array.
 */
inline array_type::tensor<double, 4> I4d()
{
    array_type::tensor<double, 4> ret = xt::empty<double>({3, 3, 3, 3});
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
 * @param A [..., 3, 3] array.
 * @return [...] array.
 */
template <class T>
inline auto Trace(const T& A)
{
    return detail::impl_A2<T, 3>::ret0(A, [](const auto& a) { return pointer::Trace(a); });
}

/**
 * Same as Trace() but writes to externally allocated output.
 *
 * @param A [..., 3, 3] array.
 * @param ret output [...] array.
 */
template <class T, class R>
inline void trace(const T& A, R& ret)
{
    detail::impl_A2<T, 3>::ret0(A, ret, [](const auto& a) { return pointer::Trace(a); });
}

/**
 * Hydrostatic part of a tensor
 *
 *     == trace(A) / 3 == trace(A) / d
 *
 * where ``d = 3``.
 * To write to allocated output use hydrostatic().
 *
 * @param A [..., 3, 3] array.
 * @return [...] array.
 */
template <class T>
inline auto Hydrostatic(const T& A)
{
    return detail::impl_A2<T, 3>::ret0(A, [](const auto& a) { return pointer::Hydrostatic(a); });
}

/**
 * Same as Hydrostatic() but writes to externally allocated output.
 *
 * @param A [..., 3, 3] array.
 * @param ret output [...] array.
 */
template <class T, class R>
inline void hydrostatic(const T& A, R& ret)
{
    detail::impl_A2<T, 3>::ret0(A, ret, [](const auto& a) { return pointer::Hydrostatic(a); });
}

/**
 * Determinant.
 * To write to allocated output use det().
 *
 * @param A [..., 3, 3] array.
 * @return [...] array.
 */
template <class T>
inline auto Det(const T& A)
{
    return detail::impl_A2<T, 3>::ret0(A, [](const auto& a) { return pointer::Det(a); });
}

/**
 * Same as Det() but writes to externally allocated output.
 *
 * @param A [..., 3, 3] array.
 * @param ret output [...] array.
 */
template <class T, class R>
inline void det(const T& A, R& ret)
{
    detail::impl_A2<T, 3>::ret0(A, ret, [](const auto& a) { return pointer::Det(a); });
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
 * @param A [..., 3, 3] array.
 * @param B [..., 3, 3] array.
 * @return [...] array.
 */
template <class T>
inline auto A2_ddot_B2(const T& A, const T& B)
{
    return detail::impl_A2<T, 3>::B2_ret0(
        A, B, [](const auto& a, const auto& b) { return pointer::A2_ddot_B2(a, b); });
}

/**
 * Same as A2_ddot_B2(const T& A, const T& B) but writes to externally allocated output.
 *
 * @param A [..., 3, 3] array.
 * @param B [..., 3, 3] array.
 * @param ret output [...] array.
 */
template <class T, class R>
inline void A2_ddot_B2(const T& A, const T& B, R& ret)
{
    detail::impl_A2<T, 3>::B2_ret0(
        A, B, ret, [](const auto& a, const auto& b) { return pointer::A2_ddot_B2(a, b); });
}

/**
 * Same as A2_ddot_B2(const T& A, const T& B, R& ret) but for symmetric tensors.
 * This function is slightly faster.
 * There is no assertion to check the symmetry.
 * To write to allocated data use A2s_ddot_B2s(const T& A, const T& B, R& ret).
 *
 * @param A [..., 3, 3] array.
 * @param B [..., 3, 3] array.
 * @return [...] array.
 */
template <class T>
inline auto A2s_ddot_B2s(const T& A, const T& B)
{
    return detail::impl_A2<T, 3>::B2_ret0(
        A, B, [](const auto& a, const auto& b) { return pointer::A2s_ddot_B2s(a, b); });
}

/**
 * Same as A2s_ddot_B2s(const T& A, const T& B) but writes to externally allocated output.
 *
 * @param A [..., 3, 3] array.
 * @param B [..., 3, 3] array.
 * @param ret output [...] array.
 */
template <class T, class R>
inline void A2s_ddot_B2s(const T& A, const T& B, R& ret)
{
    detail::impl_A2<T, 3>::B2_ret0(
        A, B, ret, [](const auto& a, const auto& b) { return pointer::A2s_ddot_B2s(a, b); });
}

/**
 * Norm of the tensor's deviator:
 *
 * \f$ \sqrt{(dev(A))_{ij} (dev(A))_{ji}} \f$
 *
 * To write to allocated data use norm_deviatoric().
 *
 * @param A [..., 3, 3] array.
 * @return [...] array.
 */
template <class T>
inline auto Norm_deviatoric(const T& A)
{
    return detail::impl_A2<T, 3>::ret0(
        A, [](const auto& a) { return pointer::Norm_deviatoric(a); });
}

/**
 * Same as Norm_deviatoric()  but writes to externally allocated output.
 *
 * @param A [..., 3, 3] array.
 * @param ret output [...] array
 */
template <class T, class R>
inline void norm_deviatoric(const T& A, R& ret)
{
    detail::impl_A2<T, 3>::ret0(A, ret, [](const auto& a) { return pointer::Norm_deviatoric(a); });
}

/**
 * Deviatoric part of a tensor:
 *
 *     A - Hydrostatic(A) * I2
 *
 * See Hydrostatic().
 * To write to allocated data use deviatoric().
 *
 * @param A [..., 3, 3] array.
 * @return [..., 3, 3] array.
 */
template <class T>
inline auto Deviatoric(const T& A)
{
    return detail::impl_A2<T, 3>::ret2(
        A, [](const auto& a, const auto& r) { return pointer::Hydrostatic_deviatoric(a, r); });
}

/**
 * Same as Deviatoric() but writes to externally allocated output.
 *
 * @param A [..., 3, 3] array.
 * @param ret output [..., 3, 3] array.
 */
template <class T, class R>
inline void deviatoric(const T& A, R& ret)
{
    detail::impl_A2<T, 3>::ret2(
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
 * @param A [..., 3, 3] array.
 * @return [..., 3, 3] array.
 */
template <class T>
inline auto Sym(const T& A)
{
    return detail::impl_A2<T, 3>::ret2(
        A, [](const auto& a, const auto& r) { return pointer::sym(a, r); });
}

/**
 * Same as Sym() but writes to externally allocated output.
 *
 * @param A [..., 3, 3] array.
 * @param ret output [..., 3, 3] array, may be the same reference as ``A``.
 */
template <class T, class R>
inline void sym(const T& A, R& ret)
{
    detail::impl_A2<T, 3>::ret2(
        A, ret, [](const auto& a, const auto& r) { return pointer::sym(a, r); });
}

/**
 * Inverse.
 * To write to allocated output use inv().
 *
 * @param A [..., 3, 3] array.
 * @return [..., 3, 3] array.
 */
template <class T>
inline auto Inv(const T& A)
{
    return detail::impl_A2<T, 3>::ret2(
        A, [](const auto& a, const auto& r) { return pointer::Inv(a, r); });
}

/**
 * Same as Inv() but writes to externally allocated output.
 *
 * @param A [..., 3, 3] array.
 * @param ret output [..., 3, 3] array.
 */
template <class T, class R>
inline void inv(const T& A, R& ret)
{
    detail::impl_A2<T, 3>::ret2(
        A, ret, [](const auto& a, const auto& r) { return pointer::Inv(a, r); });
}

/**
 * Logarithm.
 * Symmetric tensors only, no assertion.
 * To write to allocated output use logs().
 *
 * @param A [..., 3, 3] array.
 * @return [..., 3, 3] array.
 */
template <class T>
inline auto Logs(const T& A)
{
    return detail::impl_A2<T, 3>::ret2(
        A, [](const auto& a, const auto& r) { return pointer::logs(a, r); });
}

/**
 * Same as Logs() but writes to externally allocated output.
 *
 * @param A [..., 3, 3] array.
 * @param ret output [..., 3, 3] array, may be the same reference as ``A``.
 */
template <class T, class R>
inline void logs(const T& A, R& ret)
{
    detail::impl_A2<T, 3>::ret2(
        A, ret, [](const auto& a, const auto& r) { return pointer::logs(a, r); });
}

/**
 * Dot-product (single tensor contraction)
 *
 * \f$ C = A \cdot A^T \f$
 *
 * or in index notation
 *
 * \f$ C_{ik} = A_{ij} A_{kj} \f$
 *
 * To write to allocated data use A2_dot_A2T(const T& A, R& ret).
 *
 * @param A [..., 3, 3] array.
 * @return [..., 3, 3] array.
 */
template <class T>
inline auto A2_dot_A2T(const T& A)
{
    return detail::impl_A2<T, 3>::ret2(
        A, [](const auto& a, const auto& r) { return pointer::A2_dot_A2T(a, r); });
}

/**
 * Same as A2_dot_A2T(const T& A) but writes to externally allocated output.
 *
 * @param A [..., 3, 3] array.
 * @param ret output [..., 3, 3] array.
 */
template <class T, class R>
inline void A2_dot_A2T(const T& A, R& ret)
{
    detail::impl_A2<T, 3>::ret2(
        A, ret, [](const auto& a, const auto& r) { return pointer::A2_dot_A2T(a, r); });
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
 * @param A [..., 3, 3] array.
 * @param B [..., 3, 3] array.
 * @return [..., 3, 3] array.
 */
template <class T>
inline auto A2_dot_B2(const T& A, const T& B)
{
    return detail::impl_A2<T, 3>::B2_ret2(A, B, [](const auto& a, const auto& b, const auto& r) {
        return pointer::A2_dot_B2(a, b, r);
    });
}

/**
 * Same as A2_dot_B2(const T& A, const T& B) but writes to externally allocated output.
 *
 * @param A [..., 3, 3] array.
 * @param B [..., 3, 3] array.
 * @param ret output [..., 3, 3] array.
 */
template <class T, class R>
inline void A2_dot_B2(const T& A, const T& B, R& ret)
{
    detail::impl_A2<T, 3>::B2_ret2(A, B, ret, [](const auto& a, const auto& b, const auto& r) {
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
 * @param A [..., 3, 3] array.
 * @param B [..., 3, 3] array.
 * @return [..., 3, 3, 3, 3] array.
 */
template <class T>
inline auto A2_dyadic_B2(const T& A, const T& B)
{
    return detail::impl_A2<T, 3>::B2_ret4(A, B, [](const auto& a, const auto& b, const auto& r) {
        return pointer::A2_dyadic_B2(a, b, r);
    });
}

/**
 * Same as A2_dyadic_B2(const T& A, const T& B) but writes to externally allocated output.
 *
 * @param A [..., 3, 3] array.
 * @param B [..., 3, 3] array.
 * @param ret output [..., 3, 3, 3, 3] array.
 */
template <class T, class R>
inline void A2_dyadic_B2(const T& A, const T& B, R& ret)
{
    detail::impl_A2<T, 3>::B2_ret4(A, B, ret, [](const auto& a, const auto& b, const auto& r) {
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
 * @param A [..., 3, 3, 3, 3] array.
 * @param B [..., 3, 3] array.
 * @return [..., 3, 3] array.
 */
template <class T, class U>
inline auto A4_ddot_B2(const T& A, const U& B)
{
    return detail::impl_A4<T, 3>::B2_ret2(A, B, [](const auto& a, const auto& b, const auto& r) {
        return pointer::A4_ddot_B2(a, b, r);
    });
}

/**
 * Same as A4_ddot_B2(const T& A, const U& B) but writes to externally allocated output.
 *
 * @param A [..., 3, 3, 3, 3] array.
 * @param B [..., 3, 3] array.
 * @param ret output [..., 3, 3] array.
 */
template <class T, class U, class R>
inline void A4_ddot_B2(const T& A, const U& B, R& ret)
{
    detail::impl_A4<T, 3>::B2_ret2(A, B, ret, [](const auto& a, const auto& b, const auto& r) {
        return pointer::A4_ddot_B2(a, b, r);
    });
}

/**
 * Tensor contraction
 *
 * \f$ C = A \cdot B \f$
 *
 * or in index notation
 *
 * \f$ C_{ijkm} = A_{ijkl} A_{lm} \f$
 *
 * To write to allocated data use A4_dot_B2(const T& A, const U& B, R& ret).
 *
 * @param A [..., 3, 3, 3, 3] array.
 * @param B [..., 3, 3] array.
 * @return [..., 3, 3, 3, 3] array.
 */
template <class T, class U>
inline auto A4_dot_B2(const T& A, const U& B)
{
    return detail::impl_A4<T, 3>::B2_ret4(A, B, [](const auto& a, const auto& b, const auto& r) {
        return pointer::A4_dot_B2(a, b, r);
    });
}

/**
 * Same as A4_dot_B2(const T& A, const U& B) but writes to externally allocated output.
 *
 * @param A [..., 3, 3, 3, 3] array.
 * @param B [..., 3, 3] array.
 * @param ret output [..., 3, 3, 3, 3] array.
 */
template <class T, class U, class R>
inline void A4_dot_B2(const T& A, const U& B, R& ret)
{
    detail::impl_A4<T, 3>::B2_ret4(A, B, ret, [](const auto& a, const auto& b, const auto& r) {
        return pointer::A4_dot_B2(a, b, r);
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
    return detail::impl_A2<T, 3>::toSizeT0(A.shape());
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
    return detail::impl_A4<T, 3>::toSizeT0(A.shape());
}

/**
 * Shape of the underlying array.
 *
 * @param A [..., 2, 2] array.
 * @return `[...]`.
 */
template <class T>
inline auto underlying_shape_A2(const T& A) -> std::array<size_t, detail::impl_A2<T, 3>::rank>
{
    return detail::impl_A2<T, 3>::toShapeT0(A.shape());
}

/**
 * Shape of the underlying array.
 *
 * @param A [..., 2, 2] array.
 * @return `[...]`.
 */
template <class T>
inline auto underlying_shape_A4(const T& A) -> std::array<size_t, detail::impl_A4<T, 3>::rank>
{
    return detail::impl_A4<T, 3>::toShapeT0(A.shape());
}

/**
 * Array of tensors:
 * -   scalars: shape ``[...]``.
 * -   2nd-order tensors: shape ``[..., 3, 3]``.
 * -   4nd-order tensors: shape ``[..., 3, 3, 3, 3]``.
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
     * Array of Cartesian3d::O2()
     *
     * @return [shape(), 3, 3]
     */
    array_type::tensor<double, N + 2> O2() const
    {
        return xt::zeros<double>(m_shape_tensor2);
    }

    /**
     * Array of Cartesian3d::O4()
     *
     * @return [shape(), 3, 3, 3, 3]
     */
    array_type::tensor<double, N + 4> O4() const
    {
        return xt::zeros<double>(m_shape_tensor4);
    }

    /**
     * Array of Cartesian3d::I2()
     *
     * @return [shape(), 3, 3]
     */
    array_type::tensor<double, N + 2> I2() const
    {
        array_type::tensor<double, N + 2> ret = xt::empty<double>(m_shape_tensor2);

#pragma omp parallel for
        for (size_t i = 0; i < m_size; ++i) {
            Cartesian3d::pointer::I2(&ret.flat(i * m_stride_tensor2));
        }

        return ret;
    }

    /**
     * Array of Cartesian3d::II()
     *
     * @return [shape(), 3, 3, 3, 3]
     */
    array_type::tensor<double, N + 4> II() const
    {
        array_type::tensor<double, N + 4> ret = xt::empty<double>(m_shape_tensor4);

#pragma omp parallel for
        for (size_t i = 0; i < m_size; ++i) {
            Cartesian3d::pointer::II(&ret.flat(i * m_stride_tensor4));
        }

        return ret;
    }

    /**
     * Array of Cartesian3d::I4()
     *
     * @return [shape(), 3, 3, 3, 3]
     */
    array_type::tensor<double, N + 4> I4() const
    {
        array_type::tensor<double, N + 4> ret = xt::empty<double>(m_shape_tensor4);

#pragma omp parallel for
        for (size_t i = 0; i < m_size; ++i) {
            Cartesian3d::pointer::I4(&ret.flat(i * m_stride_tensor4));
        }

        return ret;
    }

    /**
     * Array of Cartesian3d::I4rt()
     *
     * @return [shape(), 3, 3, 3, 3]
     */
    array_type::tensor<double, N + 4> I4rt() const
    {
        array_type::tensor<double, N + 4> ret = xt::empty<double>(m_shape_tensor4);

#pragma omp parallel for
        for (size_t i = 0; i < m_size; ++i) {
            Cartesian3d::pointer::I4rt(&ret.flat(i * m_stride_tensor4));
        }

        return ret;
    }

    /**
     * Array of Cartesian3d::I4s()
     *
     * @return [shape(), 3, 3, 3, 3]
     */
    array_type::tensor<double, N + 4> I4s() const
    {
        array_type::tensor<double, N + 4> ret = xt::empty<double>(m_shape_tensor4);

#pragma omp parallel for
        for (size_t i = 0; i < m_size; ++i) {
            Cartesian3d::pointer::I4s(&ret.flat(i * m_stride_tensor4));
        }

        return ret;
    }

    /**
     * Array of Cartesian3d::I4d()
     *
     * @return [shape(), 3, 3, 3, 3]
     */
    array_type::tensor<double, N + 4> I4d() const
    {
        array_type::tensor<double, N + 4> ret = xt::empty<double>(m_shape_tensor4);

#pragma omp parallel for
        for (size_t i = 0; i < m_size; ++i) {
            Cartesian3d::pointer::I4d(&ret.flat(i * m_stride_tensor4));
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
    static constexpr size_t m_ndim = 3;

    /** Storage stride for 2nd-order tensors (\f$ 3^2 \f$). */
    static constexpr size_t m_stride_tensor2 = 9;

    /** Storage stride for 4th-order tensors (\f$ 3^4 \f$). */
    static constexpr size_t m_stride_tensor4 = 81;

    /** Size of the array (of scalars) == prod(#m_shape). */
    size_t m_size;

    /** Shape of the array (of scalars). */
    std::array<size_t, N> m_shape;

    /** Shape of an array of 2nd-order tensors == [#m_shape, 3, 3]. */
    std::array<size_t, N + 2> m_shape_tensor2;

    /** Shape of an array of 4th-order tensors == [#m_shape, 3, 3, 3, 3]. */
    std::array<size_t, N + 4> m_shape_tensor4;
};

} // namespace Cartesian3d
} // namespace GMatTensor

#endif
