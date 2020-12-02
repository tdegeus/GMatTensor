/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatTensor

*/

#ifndef GMATTENSOR_CARTESIAN2DSYM_H
#define GMATTENSOR_CARTESIAN2DSYM_H

#include "config.h"
#include "Cartesian2d.h"

namespace GMatTensor {
namespace Cartesian2dSym {

template <class T, class U>
inline void equivalent_deviatoric(const T& A, U& ret);

template <class T>
inline auto Equivalent_deviatoric(const T& A);

// API for pure-tensor with pointer-only input
// Storage convention:
// - Second order tensor: (xx, xy, yx, yy)
namespace pointer {

    template <class T>
    inline auto deviatoric_ddot_deviatoric(const T* A);

    template <class S, class T>
    inline auto A2_ddot_B2(const S* A, const T* B);

} // namespace pointer

} // namespace Cartesian2dSym
} // namespace GMatTensor

#include "Cartesian2dSym.hpp"

#endif
