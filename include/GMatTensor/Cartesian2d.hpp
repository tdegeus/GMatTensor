/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatTensor

*/

#ifndef GMATTENSOR_CARTESIAN2D_HPP
#define GMATTENSOR_CARTESIAN2D_HPP

#include "Cartesian2d.h"
#include "detail.hpp"

namespace GMatTensor {
namespace Cartesian2d {

namespace detail {
    using GMatTensor::detail::impl_A2;
} // namespace detail

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

template <class T, class U>
inline void hydrostatic(const T& A, U& ret)
{
    return detail::impl_A2<T, 2>::ret0(A, ret,
        [](const auto& a){ return pointer::hydrostatic(a); });
}

template <class T>
inline auto Hydrostatic(const T& A)
{
    return detail::impl_A2<T, 2>::ret0(A,
        [](const auto& a){ return pointer::hydrostatic(a); });
}

template <class T, class U>
inline void equivalent_deviatoric(const T& A, U& ret)
{
    return detail::impl_A2<T, 2>::ret0(A, ret,
        [](const auto& a){ return pointer::norm_deviatoric(a); });
}

template <class T>
inline auto Equivalent_deviatoric(const T& A)
{
    return detail::impl_A2<T, 2>::ret0(A,
        [](const auto& a){ return pointer::norm_deviatoric(a); });
}

template <class T, class U>
inline void deviatoric(const T& A, U& ret)
{
    return detail::impl_A2<T, 2>::ret2(A, ret,
        [](const auto& a, const auto& r){ return pointer::hydrostatic_deviatoric(a, r); });
}

template <class T>
inline auto Deviatoric(const T& A)
{
    return detail::impl_A2<T, 2>::ret2(A,
        [](const auto& a, const auto& r){ return pointer::hydrostatic_deviatoric(a, r); });
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

    template <class T>
    inline auto hydrostatic_deviatoric(const T* A, T* ret)
    {
        T m = hydrostatic(A);
        ret[0] = A[0] - m;
        ret[1] = A[1];
        ret[2] = A[2];
        ret[3] = A[3] - m;
        return m;
    }

    template <class T>
    inline auto deviatoric_ddot_deviatoric(const T* A)
    {
        T m = hydrostatic(A);
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
