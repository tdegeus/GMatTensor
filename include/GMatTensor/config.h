/**
Macros used in the library.

\file
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the MIT License.
*/

#ifndef GMATTENSOR_CONFIG_H
#define GMATTENSOR_CONFIG_H

#include <algorithm>
#include <string>

/**
\cond
*/
#define Q(x) #x
#define QUOTE(x) Q(x)

#define GMATTENSOR_ASSERT_IMPL(expr, file, line) \
    if (!(expr)) { \
        throw std::runtime_error( \
            std::string(file) + ':' + std::to_string(line) + \
            ": assertion failed (" #expr ") \n\t"); \
    }
/**
\endcond
*/

/**
All assertions are implementation as:

    GMATTENSOR_ASSERT(...)

They can be enabled by:

    #define GMATTENSOR_ENABLE_ASSERT

(before including GMatTensor).
The advantage is that:

-   File and line-number are displayed if the assertion fails.
-   GMatTensor's assertions can be enabled/disabled independently from those of other libraries.

\throw std::runtime_error
*/
#ifdef GMATTENSOR_ENABLE_ASSERT
#define GMATTENSOR_ASSERT(expr) GMATTENSOR_ASSERT_IMPL(expr, __FILE__, __LINE__)
#else
#define GMATTENSOR_ASSERT(expr)
#endif

/**
Tensor products / operations.
*/
namespace GMatTensor {

/**
Helper to allocate 'output' which is of the same type of some 'input', but of a different rank.
For example:
\code
// input: array of 2nd order tensors, shape = [..., ndim, ndim].
// output: array of scalars, shape = [...]
template <class T>
auto return_scalar_array(const T& A2) ->
    typename GMatTensor::allocate<xt::get_rank<T>::value - 2, T>::type;
\endcode

\tparam RANK Rank of the output.
\tparam T Type of the 'input'
*/
template <size_t RANK, class T>
struct allocate {
};

/**
\cond
*/
template <size_t RANK, class EC, size_t N, xt::layout_type L, class Tag>
struct allocate<RANK, xt::xtensor<EC, N, L, Tag>> {
    using type = typename xt::xtensor<EC, RANK, L, Tag>;
};

#ifdef XTENSOR_FIXED_HPP
template <size_t RANK, class EC, class S, xt::layout_type L>
struct allocate<RANK, xt::xtensor_fixed<EC, S, L>> {
    using type = typename xt::xtensor<EC, RANK, L>;
};
#endif

#ifdef XTENSOR_FIXED_HPP
template <size_t RANK, class EC, class S, xt::layout_type L, bool SH, class Tag>
struct allocate<RANK, xt::xfixed_container<EC, S, L, SH, Tag>> {
    using type = typename xt::xtensor<EC, RANK, L, Tag>;
};
#endif

#ifdef PY_TENSOR_HPP
template <size_t RANK, class EC, size_t N, xt::layout_type L>
struct allocate<RANK, xt::pytensor<EC, N, L>> {
    using type = typename xt::pytensor<EC, RANK, L>;
};
#endif
/**
\endcond
*/

} // namespace GMatTensor

#endif
