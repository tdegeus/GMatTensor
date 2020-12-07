/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatTensor

*/

#ifndef GMATTENSOR_CARTESIAN2D_HPP
#define GMATTENSOR_CARTESIAN2D_HPP

#include "Cartesian2d.h"

namespace GMatTensor {
namespace Cartesian2d {

inline xt::xtensor<double, 2> Random2()
{
    xt::xtensor<double, 2> ret = xt::random::randn<double>({2, 2});
    return ret;
}

inline xt::xtensor<double, 4> Random4()
{
    xt::xtensor<double, 4> ret = xt::random::randn<double>({2, 2, 2, 2});
    return ret;
}

inline xt::xtensor<double, 2> O2()
{
    xt::xtensor<double, 2> ret = xt::zeros<double>({2, 2});
    return ret;
}

inline xt::xtensor<double, 4> O4()
{
    xt::xtensor<double, 4> ret = xt::zeros<double>({2, 2, 2, 2});
    return ret;
}

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

template <class T>
inline auto trace(const T& A)
{
    GMATTENSOR_ASSERT(xt::has_shape(A, {2, 2}));
    return pointer::trace(A.data());
}

template <class S, class T>
inline auto A2_ddot_B2(const S& A, const T& B)
{
    GMATTENSOR_ASSERT(xt::has_shape(A, {2, 2}));
    GMATTENSOR_ASSERT(xt::has_shape(B, {2, 2}));
    return pointer::A2_ddot_B2(A.data(), B.data());
}

template <class S, class T>
inline auto A2s_ddot_B2s(const S& A, const T& B)
{
    GMATTENSOR_ASSERT(xt::has_shape(A, {2, 2}));
    GMATTENSOR_ASSERT(xt::has_shape(B, {2, 2}));
    return pointer::A2s_ddot_B2s(A.data(), B.data());
}

template <class S, class T>
inline auto A2_dyadic_B2(const S& A, const T& B)
{
    GMATTENSOR_ASSERT(xt::has_shape(A, {2, 2}));
    GMATTENSOR_ASSERT(xt::has_shape(B, {2, 2}));
    xt::xtensor<double, 4> ret = xt::zeros<double>({2, 2, 2, 2});
    pointer::A2_dyadic_B2(A.data(), B.data(), ret.data());
    return ret;
}

template <class S, class T>
inline auto A4_ddot_B2(const S& A, const T& B)
{
    GMATTENSOR_ASSERT(xt::has_shape(A, {2, 2, 2, 2}));
    GMATTENSOR_ASSERT(xt::has_shape(B, {2, 2}));
    xt::xtensor<double, 2> ret = xt::zeros<double>({2, 2});
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
        constexpr static size_t rank = xt::get_rank<T>::value - 2;

        template <class S>
        static size_t toSizeT0(const S& shape)
        {
            using ST = typename S::value_type;
            return std::accumulate(shape.cbegin(), shape.cend() - 2, ST(1), std::multiplies<ST>());
        }

        template <class S>
        static std::array<size_t, rank> toShapeT0(const S& shape)
        {
            std::array<size_t, rank> ret;
            std::copy(shape.cbegin(), shape.cend() - 2, ret.begin());
            return ret;
        }

        template <class S>
        static std::array<size_t, rank + 2> toShapeT2(const S& shape)
        {
            std::array<size_t, rank + 2> ret;
            std::copy(shape.cbegin(), shape.cend() - 2, ret.begin());
            ret[rank] = 2;
            ret[rank + 1] = 2;
            return ret;
        }

        static void hydrostatic_no_alloc(const T& A, xt::xtensor<value_type, rank>& ret)
        {
            GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT2(A.shape())));
            GMATTENSOR_ASSERT(xt::has_shape(ret, toShapeT0(A.shape())));
            #pragma omp parallel for
            for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
                ret.data()[i] = pointer::hydrostatic(&A.data()[i * 4]);
            }
        }

        static void equivalent_deviatoric_no_alloc(const T& A, xt::xtensor<value_type, rank>& ret)
        {
            GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT2(A.shape())));
            GMATTENSOR_ASSERT(xt::has_shape(ret, toShapeT0(A.shape())));
            #pragma omp parallel for
            for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
                ret.data()[i] = pointer::norm_deviatoric(&A.data()[i * 4]);
            }
        }

        static void deviatoric_no_alloc(const T& A, xt::xtensor<value_type, rank + 2>& ret)
        {
            GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT2(A.shape())));
            GMATTENSOR_ASSERT(xt::has_shape(A, ret.shape()));
            #pragma omp parallel for
            for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
                pointer::hydrostatic_deviatoric(&A.data()[i * 4], &ret.data()[i * 4]);
            }
        }

        static auto hydrostatic_alloc(const T& A)
        {
            xt::xtensor<value_type, rank> ret = xt::empty<value_type>(toShapeT0(A.shape()));
            hydrostatic_no_alloc(A, ret);
            return ret;
        }

        static auto equivalent_deviatoric_alloc(const T& A)
        {
            xt::xtensor<value_type, rank> ret = xt::empty<value_type>(toShapeT0(A.shape()));
            equivalent_deviatoric_no_alloc(A, ret);
            return ret;
        }

        static auto deviatoric_alloc(const T& A)
        {
            xt::xtensor<value_type, rank + 2> ret = xt::empty<value_type>(A.shape());
            deviatoric_no_alloc(A, ret);
            return ret;
        }
    };

} // namespace detail

template <class T, class U>
inline void hydrostatic(const T& A, U& ret)
{
    return detail::equiv_impl<T>::hydrostatic_no_alloc(A, ret);
}

template <class T>
inline auto Hydrostatic(const T& A)
{
    return detail::equiv_impl<T>::hydrostatic_alloc(A);
}

template <class T, class U>
inline void deviatoric(const T& A, U& ret)
{
    return detail::equiv_impl<T>::deviatoric_no_alloc(A, ret);
}

template <class T>
inline auto Deviatoric(const T& A)
{
    return detail::equiv_impl<T>::deviatoric_alloc(A);
}

template <class T, class U>
inline void equivalent_deviatoric(const T& A, U& ret)
{
    return detail::equiv_impl<T>::equivalent_deviatoric_no_alloc(A, ret);
}

template <class T>
inline auto Equivalent_deviatoric(const T& A)
{
    return detail::equiv_impl<T>::equivalent_deviatoric_alloc(A);
}

namespace pointer {

    template <class T>
    inline void O2(T* ret)
    {
        std::fill(ret, ret + 4, T(0));
    }

    template <class T>
    inline void O4(T* ret)
    {
        std::fill(ret, ret + 16, T(0));
    }

    template <class T>
    inline void I2(T* ret)
    {
        ret[0] = 1.0;
        ret[1] = 0.0;
        ret[2] = 0.0;
        ret[3] = 1.0;
    }

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

    template <class T>
    inline void I4s(T* ret)
    {
        I4(ret);

        std::array<double, 16> i4rt;
        I4rt(&i4rt[0]);

        std::transform(ret, ret + 16, &i4rt[0], ret, std::plus<T>());

        std::transform(ret, ret + 16, ret,
            std::bind(std::multiplies<T>(), std::placeholders::_1, 0.5));
    }

    template <class T>
    inline void I4d(T* ret)
    {
        I4s(ret);

        std::array<double, 16> ii;
        II(&ii[0]);

        std::transform(&ii[0], &ii[0] + 16, &ii[0],
            std::bind(std::multiplies<T>(), std::placeholders::_1, 0.5));

        std::transform(ret, ret + 16, &ii[0], ret, std::minus<T>());
    }

    template <class T>
    inline auto trace(const T* A)
    {
        return A[0] + A[3];
    }

    template <class T>
    inline auto hydrostatic(const T* A)
    {
        return T(0.5) * trace(A);
    }

    template <class S, class T>
    inline auto hydrostatic_deviatoric(const S* A, T* ret)
    {
        auto m = hydrostatic(A);
        ret[0] = A[0] - m;
        ret[1] = A[1];
        ret[2] = A[2];
        ret[3] = A[3] - m;
        return m;
    }

    template <class T>
    inline auto deviatoric_ddot_deviatoric(const T* A)
    {
        auto m = hydrostatic(A);
        return (A[0] - m) * (A[0] - m)
             + (A[3] - m) * (A[3] - m)
             + T(2) * A[1] * A[2];
    }

    template <class T>
    inline auto norm_deviatoric(const T* A)
    {
        return std::sqrt(deviatoric_ddot_deviatoric(A));
    }

    template <class S, class T>
    inline auto A2_ddot_B2(const S* A, const T* B)
    {
        return A[0] * B[0]
             + A[3] * B[3]
             + A[1] * B[2]
             + A[2] * B[1];
    }

    template <class S, class T>
    inline auto A2s_ddot_B2s(const S* A, const T* B)
    {
        return A[0] * B[0]
             + A[3] * B[3]
             + T(2) * A[1] * B[1];
    }

    template <class R, class S, class T>
    inline void A2_dyadic_B2(const R* A, const S* B, T* C)
    {
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                for (size_t k = 0; k < 2; ++k) {
                    for (size_t l = 0; l < 2; ++l) {
                        C[i * 8 + j * 4 + k * 2 + l] = A[i * 2 + j] * B[k * 2 + l];
                    }
                }
            }
        }
    }

    template <class R, class S, class T>
    inline void A4_ddot_B2(const R* A, const S* B, T* ret)
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

} // namespace Cartesian2d
} // namespace GMatTensor

#endif
