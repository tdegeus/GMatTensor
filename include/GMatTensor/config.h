/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatTensor

*/

#ifndef GMATTENSOR_H
#define GMATTENSOR_H

// #include <stdexcept>
// #include <limits>
// #include <math.h>
// #include <xtensor/xadapt.hpp>
// #include <xtensor/xfixed.hpp>
// #include <xtensor/xmath.hpp>
// #include <xtensor/xnoalias.hpp>
// #include <xtensor/xoperation.hpp>
// #include <xtensor/xsort.hpp>
#include <xtensor/xtensor.hpp>
// #include <xtensor/xview.hpp>

#ifdef GMATTENSOR_ENABLE_ASSERT

    #define GMATTENSOR_ASSERT(expr) GMATTENSOR_ASSERT_IMPL(expr, __FILE__, __LINE__)
    #define GMATTENSOR_ASSERT_IMPL(expr, file, line) \
        if (!(expr)) { \
            throw std::runtime_error( \
                std::string(file) + ':' + std::to_string(line) + \
                ": assertion failed (" #expr ") \n\t"); \
        }

#else

    #define GMATTENSOR_ASSERT(expr)

#endif

#define GMATTENSOR_VERSION_MAJOR 0
#define GMATTENSOR_VERSION_MINOR 1
#define GMATTENSOR_VERSION_PATCH 0

#define GMATTENSOR_VERSION_AT_LEAST(x, y, z) \
    (GMATTENSOR_VERSION_MAJOR > x || (GMATELASTI_VERSION_MAJOR >= x && \
    (GMATTENSOR_VERSION_MINOR > y || (GMATELASTI_VERSION_MINOR >= y && \
                                       GMATELASTI_VERSION_PATCH >= z))))

#define GMATTENSOR_VERSION(x, y, z) \
    (GMATTENSOR_VERSION_MAJOR == x && \
     GMATTENSOR_VERSION_MINOR == y && \
     GMATTENSOR_VERSION_PATCH == z)

#endif
