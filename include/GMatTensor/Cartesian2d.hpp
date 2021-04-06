/**
Implementation of Cartesian2d.h

\file GMatTensor/Cartesian2d.hpp
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the MIT License.
*/

#ifndef GMATTENSOR_CARTESIAN2D_HPP
#define GMATTENSOR_CARTESIAN2D_HPP

#include "Cartesian2d.h"
#include "detail.hpp"

namespace GMatTensor {
namespace Cartesian2d {

namespace detail {
    using GMatTensor::detail::impl_A2;
    using GMatTensor::detail::impl_A4;
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
    xt::xtensor<double, 2> ret = xt::empty<double>({2, 2});
    pointer::I2(ret.data());
    return ret;
}

inline xt::xtensor<double, 4> II()
{
    xt::xtensor<double, 4> ret = xt::empty<double>({2, 2, 2, 2});
    pointer::II(ret.data());
    return ret;
}

inline xt::xtensor<double, 4> I4()
{
    xt::xtensor<double, 4> ret = xt::empty<double>({2, 2, 2, 2});
    pointer::I4(ret.data());
    return ret;
}

inline xt::xtensor<double, 4> I4rt()
{
    xt::xtensor<double, 4> ret = xt::empty<double>({2, 2, 2, 2});
    pointer::I4rt(ret.data());
    return ret;
}

inline xt::xtensor<double, 4> I4s()
{
    xt::xtensor<double, 4> ret = xt::empty<double>({2, 2, 2, 2});
    pointer::I4s(ret.data());
    return ret;
}

inline xt::xtensor<double, 4> I4d()
{
    xt::xtensor<double, 4> ret = xt::empty<double>({2, 2, 2, 2});
    pointer::I4d(ret.data());
    return ret;
}

template <class T, class R>
inline void trace(const T& A, R& ret)
{
    return detail::impl_A2<T, 2>::ret0(A, ret,
        [](const auto& a){ return pointer::Trace(a); });
}

template <class T>
inline auto Trace(const T& A)
{
    return detail::impl_A2<T, 2>::ret0(A,
        [](const auto& a){ return pointer::Trace(a); });
}

template <class T, class R>
inline void hydrostatic(const T& A, R& ret)
{
    return detail::impl_A2<T, 2>::ret0(A, ret,
        [](const auto& a){ return pointer::Hydrostatic(a); });
}

template <class T>
inline auto Hydrostatic(const T& A)
{
    return detail::impl_A2<T, 2>::ret0(A,
        [](const auto& a){ return pointer::Hydrostatic(a); });
}

template <class T, class R>
inline void A2_ddot_B2(const T& A, const T& B, R& ret)
{
    return detail::impl_A2<T, 2>::B2_ret0(A, B, ret,
        [](const auto& a, const auto& b){ return pointer::A2_ddot_B2(a, b); });
}

template <class T>
inline auto A2_ddot_B2(const T& A, const T& B)
{
    return detail::impl_A2<T, 2>::B2_ret0(A, B,
        [](const auto& a, const auto& b){ return pointer::A2_ddot_B2(a, b); });
}

template <class T, class R>
inline void A2s_ddot_B2s(const T& A, const T& B, R& ret)
{
    return detail::impl_A2<T, 2>::B2_ret0(A, B, ret,
        [](const auto& a, const auto& b){ return pointer::A2s_ddot_B2s(a, b); });
}

template <class T>
inline auto A2s_ddot_B2s(const T& A, const T& B)
{
    return detail::impl_A2<T, 2>::B2_ret0(A, B,
        [](const auto& a, const auto& b){ return pointer::A2s_ddot_B2s(a, b); });
}

template <class T, class R>
inline void norm_deviatoric(const T& A, R& ret)
{
    return detail::impl_A2<T, 2>::ret0(A, ret,
        [](const auto& a){ return pointer::Norm_deviatoric(a); });
}

template <class T>
inline auto Norm_deviatoric(const T& A)
{
    return detail::impl_A2<T, 2>::ret0(A,
        [](const auto& a){ return pointer::Norm_deviatoric(a); });
}

template <class T, class R>
inline void deviatoric(const T& A, R& ret)
{
    return detail::impl_A2<T, 2>::ret2(A, ret,
        [](const auto& a, const auto& r){ return pointer::Hydrostatic_deviatoric(a, r); });
}

template <class T>
inline auto Deviatoric(const T& A)
{
    return detail::impl_A2<T, 2>::ret2(A,
        [](const auto& a, const auto& r){ return pointer::Hydrostatic_deviatoric(a, r); });
}

template <class T, class R>
inline void sym(const T& A, R& ret)
{
    return detail::impl_A2<T, 2>::ret2(A, ret,
        [](const auto& a, const auto& r){ return pointer::sym(a, r); });
}

template <class T>
inline auto Sym(const T& A)
{
    return detail::impl_A2<T, 2>::ret2(A,
        [](const auto& a, const auto& r){ return pointer::sym(a, r); });
}

template <class T, class R>
inline void A2_dot_B2(const T& A, const T& B, R& ret)
{
    return detail::impl_A2<T, 2>::B2_ret2(A, B, ret,
        [](const auto& a, const auto& b, const auto& r){ return pointer::A2_dot_B2(a, b, r); });
}

template <class T>
inline auto A2_dot_B2(const T& A, const T& B)
{
    return detail::impl_A2<T, 2>::B2_ret2(A, B,
        [](const auto& a, const auto& b, const auto& r){ return pointer::A2_dot_B2(a, b, r); });
}

template <class T, class R>
inline void A2_dyadic_B2(const T& A, const T& B, R& ret)
{
    return detail::impl_A2<T, 2>::B2_ret4(A, B, ret,
        [](const auto& a, const auto& b, const auto& r){ return pointer::A2_dyadic_B2(a, b, r); });
}

template <class T>
inline auto A2_dyadic_B2(const T& A, const T& B)
{
    return detail::impl_A2<T, 2>::B2_ret4(A, B,
        [](const auto& a, const auto& b, const auto& r){ return pointer::A2_dyadic_B2(a, b, r); });
}

template <class T, class U, class R>
inline void A4_ddot_B2(const T& A, const U& B, R& ret)
{
    return detail::impl_A4<T, 2>::B2_ret2(A, B, ret,
        [](const auto& a, const auto& b, const auto& r){ return pointer::A4_ddot_B2(a, b, r); });
}

template <class T, class U>
inline auto A4_ddot_B2(const T& A, const U& B)
{
    return detail::impl_A4<T, 2>::B2_ret2(A, B,
        [](const auto& a, const auto& b, const auto& r){ return pointer::A4_ddot_B2(a, b, r); });
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
    inline T Trace(const T* A)
    {
        return A[0] + A[3];
    }

    template <class T>
    inline T Hydrostatic(const T* A)
    {
        return T(0.5) * Trace(A);
    }

    template <class T>
    inline void sym(const T* A, T* ret)
    {
        ret[0] = A[0];
        ret[1] = 0.5 * (A[1] + A[2]);
        ret[2] = ret[1];
        ret[3] = A[3];
    }

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

    template <class T>
    inline T Deviatoric_ddot_deviatoric(const T* A)
    {
        T m = Hydrostatic(A);
        return (A[0] - m) * (A[0] - m)
             + (A[3] - m) * (A[3] - m)
             + T(2) * A[1] * A[2];
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
             + A[3] * B[3]
             + A[1] * B[2]
             + A[2] * B[1];
    }

    template <class T>
    inline T A2s_ddot_B2s(const T* A, const T* B)
    {
        return A[0] * B[0]
             + A[3] * B[3]
             + T(2) * A[1] * B[1];
    }

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

    template <class T>
    inline void A2_dot_B2(const T* A, const T* B, T* ret)
    {
        ret[0] = A[1] * B[2] + A[0] * B[0];
        ret[1] = A[0] * B[1] + A[1] * B[3];
        ret[2] = A[2] * B[0] + A[3] * B[2];
        ret[3] = A[2] * B[1] + A[3] * B[3];
    }

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

} // namespace Cartesian2d
} // namespace GMatTensor

#endif
