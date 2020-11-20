/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatTensor

*/

#ifndef GMATTENSOR_CARTESIAN2D_HPP
#define GMATTENSOR_CARTESIAN2D_HPP

#include "Cartesian2d.h"

namespace GMatTensor {
namespace Cartesian2d {

inline xt::xtensor<double, 2> I2()
{
    return xt::xtensor<double, 2>({{1.0, 0.0},
                                   {0.0, 1.0}});
}

inline xt::xtensor<double, 4> II()
{
    xt::xtensor<double, 4> ret = xt::zeros<double>({2, 2, 2, 2});

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                for (size_t l = 0; l < 2; ++l) {
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
    xt::xtensor<double, 4> ret = xt::zeros<double>({2, 2, 2, 2});

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                for (size_t l = 0; l < 2; ++l) {
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
    xt::xtensor<double, 4> ret = xt::zeros<double>({2, 2, 2, 2});

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                for (size_t l = 0; l < 2; ++l) {
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
    return I4s() - 0.5 * II();
}

namespace detail {

    template <class T>
    inline auto trace(const T A)
    {
        return A[0] + A[3];
    }

    template <class T, class U>
    inline void deviatoric(const T A, U ret)
    {
        auto m = 0.5 * (A[0] + A[3]);
        ret[0] = A[0] - m;
        ret[1] = A[1];
        ret[2] = A[2];
        ret[3] = A[3] - m;
    }

    template <class T>
    inline auto deviatoric_ddot_deviatoric(const T A)
    {
        auto m = 0.5 * (A[0] + A[3]);
        return (A[0] - m) * (A[0] - m) + 2.0 * A[1] * A[1] + (A[3] - m) * (A[3] - m);
    }

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
            ret[rank - 2] = 2;
            ret[rank - 1] = 2;
            return ret;
        }

        static void deviatoric_no_alloc(const T& A, xt::xtensor<value_type, rank>& B)
        {
            GMATTENSOR_ASSERT(xt::has_shape(A, toShape(A.shape())));
            GMATTENSOR_ASSERT(xt::has_shape(A, B.shape()));
            for (size_t i = 0; i < toMatrixSize(A.shape()); ++i) {
                detail::deviatoric(&A.data()[i * 4], &B.data()[i * 4]);
            }
        }

        static void hydrostatic_no_alloc(const T& A, xt::xtensor<value_type, rank - 2>& B)
        {
            GMATTENSOR_ASSERT(xt::has_shape(A, toShape(A.shape())));
            GMATTENSOR_ASSERT(xt::has_shape(B, toMatrixShape(A.shape())));
            for (size_t i = 0; i < toMatrixSize(A.shape()); ++i) {
                B.data()[i] = 0.5 * detail::trace(&A.data()[i * 4]);
            }
        }

        static void equivalent_deviatoric_no_alloc(const T& A, xt::xtensor<value_type, rank - 2>& B)
        {
            GMATTENSOR_ASSERT(xt::has_shape(A, toShape(A.shape())));
            GMATTENSOR_ASSERT(xt::has_shape(B, toMatrixShape(A.shape())));
            for (size_t i = 0; i < toMatrixSize(A.shape()); ++i) {
                auto b = detail::deviatoric_ddot_deviatoric(&A.data()[i * 4]);
                B.data()[i] = std::sqrt(b);
            }
        }

        static auto deviatoric_alloc(const T& A)
        {
            xt::xtensor<value_type, rank> B = xt::empty<value_type>(A.shape());
            deviatoric_no_alloc(A, B);
            return B;
        }

        static auto hydrostatic_alloc(const T& A)
        {
            xt::xtensor<value_type, rank - 2> B = xt::empty<value_type>(toMatrixShape(A.shape()));
            hydrostatic_no_alloc(A, B);
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

} // namespace Cartesian2d
} // namespace GMatTensor

#endif
