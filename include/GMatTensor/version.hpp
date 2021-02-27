/**
\file version.hpp
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef GMATTENSOR_VERSION_HPP
#define GMATTENSOR_VERSION_HPP

#include "version.h"

namespace GMatTensor {

namespace detail {

    inline std::string unquote(const std::string& arg)
    {
        std::string ret = arg;
        ret.erase(std::remove(ret.begin(), ret.end(), '\"'), ret.end());
        return ret;
    }

}

inline std::string version()
{
    return detail::unquote(std::string(QUOTE(GMATTENSOR_VERSION)));
}

} // namespace GMatTensor

#endif
