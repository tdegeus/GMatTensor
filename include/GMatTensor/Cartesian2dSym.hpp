/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatTensor

*/

#ifndef GMATTENSOR_CARTESIAN2DSYM_HPP
#define GMATTENSOR_CARTESIAN2DSYM_HPP

#include "Cartesian2d.h"

namespace GMatTensor {
namespace Cartesian2dSym {

namespace detail {

    template <class T>
    struct equiv_impl : GMatTensor::Cartesian2d::detail::equiv_impl<T>
    {
        using value_type = typename T::value_type;
        using GMatTensor::Cartesian2d::detail::equiv_impl<T>::rank;
        using GMatTensor::Cartesian2d::detail::equiv_impl<T>::toMatrixSize;
        using GMatTensor::Cartesian2d::detail::equiv_impl<T>::toMatrixShape;
        using GMatTensor::Cartesian2d::detail::equiv_impl<T>::toShape;

        static void equivalent_deviatoric_no_alloc(const T& A, xt::xtensor<value_type, rank - 2>& B)
        {
            GMATTENSOR_ASSERT(xt::has_shape(A, toShape(A.shape())));
            GMATTENSOR_ASSERT(xt::has_shape(B, toMatrixShape(A.shape())));
            #pragma omp parallel for
            for (size_t i = 0; i < toMatrixSize(A.shape()); ++i) {
                auto b = pointer::deviatoric_ddot_deviatoric(&A.data()[i * 4]);
                B.data()[i] = std::sqrt(b);
            }
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

    template <class T>
    inline auto deviatoric_ddot_deviatoric(const T* A)
    {
        auto m = 0.5 * (A[0] + A[3]);
        return (A[0] - m) * (A[0] - m)
             + (A[3] - m) * (A[3] - m)
             + 2.0 * A[1] * A[1];
    }

    template <class S, class T>
    inline auto A2_ddot_B2(const S* A, const T* B)
    {
        return A[0] * B[0]
             + A[3] * B[3]
             + 2.0 * A[1] * B[1];
    }

} // namespace pointer

} // namespace Cartesian2dSym
} // namespace GMatTensor

#endif
