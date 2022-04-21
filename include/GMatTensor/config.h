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
}

#endif
