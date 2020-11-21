/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatTensor

*/

#ifndef GMATTENSOR_CARTESIAN3D_HPP
#define GMATTENSOR_CARTESIAN3D_HPP

#include "Cartesian3d.h"

namespace GMatTensor {
namespace Cartesian3d {

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

    template <class T>
    inline auto trace(const T A)
    {
        return A[0] + A[4] + A[8];
    }

    template <class T, class U>
    inline auto hydrostatic_deviatoric(const T A, U ret)
    {
        auto m = (A[0] + A[4] + A[8]) / 3.0;
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
    inline auto deviatoric_ddot_deviatoric(const T A)
    {
        auto m = (A[0] + A[4] + A[8]) / 3.0;
        return (A[0] - m) * (A[0] - m)
             + (A[4] - m) * (A[4] - m)
             + (A[8] - m) * (A[8] - m)
             + 2.0 * A[1] * A[1]
             + 2.0 * A[2] * A[2]
             + 2.0 * A[5] * A[5];
    }

    template <class T, class U>
    inline auto A2_ddot_B2(const T A, const U B)
    {
        return A[0] * B[0]
             + A[4] * B[4]
             + A[8] * B[8]
             + 2.0 * A[1] * B[1]
             + 2.0 * A[2] * B[2]
             + 2.0 * A[5] * B[5];
    }

} // namespace pointer

} // namespace Cartesian3d
} // namespace GMatTensor

#endif
