/**
Implementation details (not part of public API).

\file GMatTensor/detail.hpp
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the MIT License.
*/

#ifndef GMATTENSOR_DETAIL_HPP
#define GMATTENSOR_DETAIL_HPP

namespace GMatTensor {
namespace detail {

// --------------------------
// Array of 2nd-order tensors
// --------------------------

template <class T, size_t nd>
struct impl_A2
{
    using value_type = typename T::value_type;
    using shape_type = typename T::shape_type;
    static_assert(xt::has_fixed_rank_t<T>::value, "Only fixed rank allowed.");
    static_assert(xt::get_rank<T>::value >= 2, "Rank too low.");
    constexpr static size_t rank = xt::get_rank<T>::value - 2;
    constexpr static size_t stride0 = 1;
    constexpr static size_t stride2 = nd * nd;
    constexpr static size_t stride4 = nd * nd * nd * nd;

    // Get the size of the underlying array (strips the last to dimensions)

    template <class S>
    static size_t toSizeT0(const S& shape)
    {
        using ST = typename S::value_type;
        return std::accumulate(shape.cbegin(), shape.cbegin() + rank, ST(1), std::multiplies<ST>());
    }

    // Extract shape of the underlying array (strips the last to dimensions)
    // and optionally add the dimensions

    template <class S>
    static std::array<size_t, rank> toShapeT0(const S& shape)
    {
        std::array<size_t, rank> ret;
        std::copy(shape.cbegin(), shape.cbegin() + rank, ret.begin());
        return ret;
    }

    template <class S>
    static std::array<size_t, rank + 2> toShapeT2(const S& shape)
    {
        std::array<size_t, rank + 2> ret;
        std::copy(shape.cbegin(), shape.cbegin() + rank, ret.begin());
        ret[rank + 0] = nd;
        ret[rank + 1] = nd;
        return ret;
    }

    template <class S>
    static std::array<size_t, rank + 4> toShapeT4(const S& shape)
    {
        std::array<size_t, rank + 4> ret;
        std::copy(shape.cbegin(), shape.cbegin() + rank, ret.begin());
        ret[rank + 0] = nd;
        ret[rank + 1] = nd;
        ret[rank + 2] = nd;
        ret[rank + 3] = nd;
        return ret;
    }

    // Argument: A2
    // Return: scalar

    template <typename F>
    static void ret0(const T& A, xt::xtensor<value_type, rank>& ret, F func)
    {
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT2(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(ret, toShapeT0(A.shape())));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            ret.data()[i] = func(&A.data()[i * stride2]);
        }
    }

    template <typename F>
    static auto ret0(const T& A, F func)
    {
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT2(A.shape())));
        xt::xtensor<value_type, rank> ret = xt::empty<value_type>(toShapeT0(A.shape()));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            ret.data()[i] = func(&A.data()[i * stride2]);
        }
        return ret;
    }

    // Argument: A2, B2
    // Return: scalar

    template <typename F>
    static void B2_ret0(const T& A, const T& B, xt::xtensor<value_type, rank>& ret, F func)
    {
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT2(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(B, toShapeT2(B.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(ret, toShapeT0(A.shape())));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            ret.data()[i] = func(&A.data()[i * stride2], &B.data()[i * stride2]);
        }
    }

    template <typename F>
    static auto B2_ret0(const T& A, const T& B, F func)
    {
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT2(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(B, toShapeT2(A.shape())));
        xt::xtensor<value_type, rank> ret = xt::empty<value_type>(toShapeT0(A.shape()));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            ret.data()[i] = func(&A.data()[i * stride2], &B.data()[i * stride2]);
        }
        return ret;
    }

    // Argument: A2, B2
    // Return: 2nd-order tensor

    template <typename F>
    static void B2_ret2(const T& A, const T& B, xt::xtensor<value_type, rank + 2>& ret, F func)
    {
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT2(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(B, toShapeT2(B.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(ret, toShapeT2(A.shape())));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            func(&A.data()[i * stride2], &B.data()[i * stride2], &ret.data()[i * stride2]);
        }
    }

    template <typename F>
    static auto B2_ret2(const T& A, const T& B, F func)
    {
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT2(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(B, toShapeT2(A.shape())));
        xt::xtensor<value_type, rank + 2> ret = xt::empty<value_type>(toShapeT2(A.shape()));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            func(&A.data()[i * stride2], &B.data()[i * stride2], &ret.data()[i * stride2]);
        }
        return ret;
    }

    // Argument: A2, B2
    // Return: 4th-order tensor

    template <typename F>
    static void B2_ret4(const T& A, const T& B, xt::xtensor<value_type, rank + 4>& ret, F func)
    {
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT2(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(B, toShapeT2(B.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(ret, toShapeT4(A.shape())));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            func(&A.data()[i * stride2], &B.data()[i * stride2], &ret.data()[i * stride4]);
        }
    }

    template <typename F>
    static auto B2_ret4(const T& A, const T& B, F func)
    {
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT2(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(B, toShapeT2(A.shape())));
        xt::xtensor<value_type, rank + 4> ret = xt::empty<value_type>(toShapeT4(A.shape()));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            func(&A.data()[i * stride2], &B.data()[i * stride2], &ret.data()[i * stride4]);
        }
        return ret;
    }

    // Argument: A2
    // Return: 2nd-order tensor

    template <typename F>
    static void ret2(const T& A, xt::xtensor<value_type, rank + 2>& ret, F func)
    {
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT2(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(ret, toShapeT2(A.shape())));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            func(&A.data()[i * stride2], &ret.data()[i * stride2]);
        }
    }

    template <typename F>
    static auto ret2(const T& A, F func)
    {
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT2(A.shape())));
        xt::xtensor<value_type, rank + 2> ret = xt::empty<value_type>(toShapeT2(A.shape()));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            func(&A.data()[i * stride2], &ret.data()[i * stride2]);
        }
        return ret;
    }
};

// --------------------------
// Array of 4th-order tensors
// --------------------------

template <class T, size_t nd>
struct impl_A4
{
    using value_type = typename T::value_type;
    using shape_type = typename T::shape_type;
    static_assert(xt::has_fixed_rank_t<T>::value, "Only fixed rank allowed.");
    static_assert(xt::get_rank<T>::value >= 4, "Rank too low.");
    constexpr static size_t rank = xt::get_rank<T>::value - 4;
    constexpr static size_t stride0 = 1;
    constexpr static size_t stride2 = nd * nd;
    constexpr static size_t stride4 = nd * nd * nd * nd;

    // Get the size of the underlying array (strips the last to dimensions)

    template <class S>
    static size_t toSizeT0(const S& shape)
    {
        using ST = typename S::value_type;
        return std::accumulate(shape.cbegin(), shape.cbegin() + rank, ST(1), std::multiplies<ST>());
    }

    // Extract shape of the underlying array (strips the last to dimensions)
    // and optionally add the dimensions

    template <class S>
    static std::array<size_t, rank> toShapeT0(const S& shape)
    {
        std::array<size_t, rank> ret;
        std::copy(shape.cbegin(), shape.cbegin() + rank, ret.begin());
        return ret;
    }

    template <class S>
    static std::array<size_t, rank + 2> toShapeT2(const S& shape)
    {
        std::array<size_t, rank + 2> ret;
        std::copy(shape.cbegin(), shape.cbegin() + rank, ret.begin());
        ret[rank + 0] = nd;
        ret[rank + 1] = nd;
        return ret;
    }

    template <class S>
    static std::array<size_t, rank + 4> toShapeT4(const S& shape)
    {
        std::array<size_t, rank + 4> ret;
        std::copy(shape.cbegin(), shape.cbegin() + rank, ret.begin());
        ret[rank + 0] = nd;
        ret[rank + 1] = nd;
        ret[rank + 2] = nd;
        ret[rank + 3] = nd;
        return ret;
    }

    // Argument: A4, B2
    // Return: 2nd-order tensor

    template <class U, typename F>
    static void B2_ret2(const T& A, const U& B, xt::xtensor<value_type, rank + 2>& ret, F func)
    {
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT4(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(B, toShapeT2(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(ret, toShapeT2(A.shape())));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            func(&A.data()[i * stride4], &B.data()[i * stride2], &ret.data()[i * stride2]);
        }
    }

    template <class U, typename F>
    static auto B2_ret2(const T& A, const U& B, F func)
    {
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT4(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(B, toShapeT2(A.shape())));
        xt::xtensor<value_type, rank + 2> ret = xt::empty<value_type>(toShapeT2(A.shape()));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            func(&A.data()[i * stride4], &B.data()[i * stride2], &ret.data()[i * stride2]);
        }
        return ret;
    }

    // Argument: A4, B2
    // Return: 4th-order tensor

    template <class U, typename F>
    static void B2_ret4(const T& A, const U& B, xt::xtensor<value_type, rank + 4>& ret, F func)
    {
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT4(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(B, toShapeT2(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(ret, toShapeT4(A.shape())));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            func(&A.data()[i * stride4], &B.data()[i * stride2], &ret.data()[i * stride2]);
        }
    }

    template <class U, typename F>
    static auto B2_ret4(const T& A, const U& B, F func)
    {
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT4(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(B, toShapeT2(A.shape())));
        xt::xtensor<value_type, rank + 4> ret = xt::empty<value_type>(toShapeT4(A.shape()));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            func(&A.data()[i * stride4], &B.data()[i * stride2], &ret.data()[i * stride2]);
        }
        return ret;
    }
};

} // namespace detail
} // namespace GMatTensor

#endif
