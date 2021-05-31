/**
Implementation details (not part of public API).

\file
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the MIT License.
*/

#ifndef GMATTENSOR_DETAIL_HPP
#define GMATTENSOR_DETAIL_HPP

namespace GMatTensor {
namespace detail {

template <size_t RANK, class T>
struct allocate
{
};

template <size_t RANK, class EC, size_t N, xt::layout_type L, class Tag>
struct allocate<RANK, xt::xtensor<EC, N, L, Tag>>
{
    using type = typename xt::xtensor<EC, RANK, L, Tag>;
};

#ifdef XTENSOR_FIXED_HPP
template <size_t RANK, class EC, class S, xt::layout_type L>
struct allocate<RANK, xt::xtensor_fixed<EC, S, L>>
{
    using type = typename xt::xtensor<EC, RANK, L>;
};
#endif

#ifdef XTENSOR_FIXED_HPP
template <size_t RANK, class EC, class S, xt::layout_type L, bool SH, class Tag>
struct allocate<RANK, xt::xfixed_container<EC, S, L, SH, Tag>>
{
    using type = typename xt::xtensor<EC, RANK, L, Tag>;
};
#endif

#ifdef PY_TENSOR_HPP
template <size_t RANK, class EC, size_t N, xt::layout_type L>
struct allocate<RANK, xt::pytensor<EC, N, L>>
{
    using type = typename xt::pytensor<EC, RANK, L>;
};
#endif

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
    constexpr static size_t rank = xt::get_rank<T>::value - 2; // rank of the 'pure' matrix
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

    /**
    \param A 2nd order tensor.
    \param ret scalar (0th order tensor).
    */
    template <class R, typename F>
    static void ret0(const T& A, R& ret, F func)
    {
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT2(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(ret, toShapeT0(A.shape())));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            ret.flat(i) = func(&A.flat(i * stride2));
        }
    }

    /**
    \param A 2nd order tensor.
    \return scalar (0th order tensor).
    */
    template <typename F>
    static auto ret0(const T& A, F func) -> typename allocate<rank, T>::type
    {
        using return_type = typename allocate<rank, T>::type;
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT2(A.shape())));
        return_type ret = return_type::from_shape(toShapeT0(A.shape()));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            ret.flat(i) = func(&A.flat(i * stride2));
        }
        return ret;
    }

    // Argument: A2, B2
    // Return: scalar

    template <class R, typename F>
    static void B2_ret0(const T& A, const T& B, R& ret, F func)
    {
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT2(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(B, toShapeT2(B.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(ret, toShapeT0(A.shape())));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            ret.flat(i) = func(&A.flat(i * stride2), &B.flat(i * stride2));
        }
    }

    template <typename F>
    static auto B2_ret0(const T& A, const T& B, F func) -> typename allocate<rank, T>::type
    {
        using return_type = typename allocate<rank, T>::type;
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT2(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(B, toShapeT2(A.shape())));
        return_type ret = return_type::from_shape(toShapeT0(A.shape()));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            ret.flat(i) = func(&A.flat(i * stride2), &B.flat(i * stride2));
        }
        return ret;
    }

    // Argument: A2, B2
    // Return: 2nd-order tensor

    template <class R, typename F>
    static void B2_ret2(const T& A, const T& B, R& ret, F func)
    {
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT2(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(B, toShapeT2(B.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(ret, toShapeT2(A.shape())));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            func(&A.flat(i * stride2), &B.flat(i * stride2), &ret.flat(i * stride2));
        }
    }

    template <typename F>
    static auto B2_ret2(const T& A, const T& B, F func) -> typename allocate<rank + 2, T>::type
    {
        using return_type = typename allocate<rank + 2, T>::type;
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT2(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(B, toShapeT2(A.shape())));
        return_type ret = return_type::from_shape(toShapeT2(A.shape()));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            func(&A.flat(i * stride2), &B.flat(i * stride2), &ret.flat(i * stride2));
        }
        return ret;
    }

    // Argument: A2, B2
    // Return: 4th-order tensor

    template <class R, typename F>
    static void B2_ret4(const T& A, const T& B, R& ret, F func)
    {
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT2(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(B, toShapeT2(B.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(ret, toShapeT4(A.shape())));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            func(&A.flat(i * stride2), &B.flat(i * stride2), &ret.flat(i * stride4));
        }
    }

    template <typename F>
    static auto B2_ret4(const T& A, const T& B, F func) -> typename allocate<rank + 4, T>::type
    {
        using return_type = typename allocate<rank + 4, T>::type;
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT2(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(B, toShapeT2(A.shape())));
        return_type ret = return_type::from_shape(toShapeT4(A.shape()));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            func(&A.flat(i * stride2), &B.flat(i * stride2), &ret.flat(i * stride4));
        }
        return ret;
    }

    // Argument: A2
    // Return: 2nd-order tensor

    template <class R, typename F>
    static void ret2(const T& A, R& ret, F func)
    {
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT2(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(ret, toShapeT2(A.shape())));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            func(&A.flat(i * stride2), &ret.flat(i * stride2));
        }
    }

    template <typename F>
    static auto ret2(const T& A, F func) -> typename allocate<rank + 2, T>::type
    {
        using return_type = typename allocate<rank + 2, T>::type;
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT2(A.shape())));
        return_type ret = return_type::from_shape(toShapeT2(A.shape()));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            func(&A.flat(i * stride2), &ret.flat(i * stride2));
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

    template <class U, class R, typename F>
    static void B2_ret2(const T& A, const U& B, R& ret, F func)
    {
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT4(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(B, toShapeT2(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(ret, toShapeT2(A.shape())));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            func(&A.flat(i * stride4), &B.flat(i * stride2), &ret.flat(i * stride2));
        }
    }

    template <class U, typename F>
    static auto B2_ret2(const T& A, const U& B, F func) -> typename allocate<rank + 2, T>::type
    {
        using return_type = typename allocate<rank + 2, T>::type;
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT4(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(B, toShapeT2(A.shape())));
        return_type ret = return_type::from_shape(toShapeT2(A.shape()));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            func(&A.flat(i * stride4), &B.flat(i * stride2), &ret.flat(i * stride2));
        }
        return ret;
    }

    // Argument: A4, B2
    // Return: 4th-order tensor

    template <class U, class R, typename F>
    static void B2_ret4(const T& A, const U& B, R& ret, F func)
    {
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT4(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(B, toShapeT2(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(ret, toShapeT4(A.shape())));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            func(&A.flat(i * stride4), &B.flat(i * stride2), &ret.flat(i * stride2));
        }
    }

    template <class U, typename F>
    static auto B2_ret4(const T& A, const U& B, F func) -> typename allocate<rank + 4, T>::type
    {
        using return_type = typename allocate<rank + 4, T>::type;
        GMATTENSOR_ASSERT(xt::has_shape(A, toShapeT4(A.shape())));
        GMATTENSOR_ASSERT(xt::has_shape(B, toShapeT2(A.shape())));
        return_type ret = return_type::from_shape(toShapeT4(A.shape()));
        #pragma omp parallel for
        for (size_t i = 0; i < toSizeT0(A.shape()); ++i) {
            func(&A.flat(i * stride4), &B.flat(i * stride2), &ret.flat(i * stride2));
        }
        return ret;
    }
};

} // namespace detail
} // namespace GMatTensor

#endif
