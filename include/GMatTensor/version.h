/**
\file
\copyright Copyright. Tom de Geus. All rights reserved.
\license This project is released under the MIT License.
*/

#ifndef GMATTENSOR_VERSION_H
#define GMATTENSOR_VERSION_H

#include "config.h"

/**
Current version.

Either:

-   Configure using CMake at install time. Internally uses::

        python -c "from setuptools_scm import get_version; print(get_version())"

-   Define externally using::

        -DGMATTENSOR_VERSION="`python -c "from setuptools_scm import get_version;
print(get_version())"`"

    From the root of this project. This is what ``setup.py`` does.

Note that both ``CMakeLists.txt`` and ``setup.py`` will construct the version using
``setuptools_scm``. Tip: use the environment variable ``SETUPTOOLS_SCM_PRETEND_VERSION`` to
overwrite the automatic version.
*/
#ifndef GMATTENSOR_VERSION
#define GMATTENSOR_VERSION "@PROJECT_VERSION@"
#endif

namespace GMatTensor {

namespace detail {

inline std::string unquote(const std::string& arg)
{
    std::string ret = arg;
    ret.erase(std::remove(ret.begin(), ret.end(), '\"'), ret.end());
    return ret;
}

} // namespace detail

/**
Return version string, e.g. `"0.8.0"`
\return Version string.
*/
inline std::string version()
{
    return detail::unquote(std::string(QUOTE(GMATTENSOR_VERSION)));
}

/**
Return versions of this library and of all of its major dependencies.
The output is a list of strings:

    "gmattensor=0.8.0",
    "xtensor=0.20.1",
    ...

\return List of strings.
*/
inline std::vector<std::string> version_dependencies()
{
    std::vector<std::string> ret;

    ret.push_back("gmattensor=" + GMatTensor::version());

    ret.push_back(
        "xtensor=" + detail::unquote(std::string(QUOTE(XTENSOR_VERSION_MAJOR))) + "." +
        detail::unquote(std::string(QUOTE(XTENSOR_VERSION_MINOR))) + "." +
        detail::unquote(std::string(QUOTE(XTENSOR_VERSION_PATCH))));

    // xtensor suite

#ifdef XSIMD_VERSION_MAJOR
    ret.push_back(
        "xsimd=" + detail::unquote(std::string(QUOTE(XSIMD_VERSION_MAJOR))) + "." +
        detail::unquote(std::string(QUOTE(XSIMD_VERSION_MINOR))) + "." +
        detail::unquote(std::string(QUOTE(XSIMD_VERSION_PATCH))));
#endif

#ifdef XTL_VERSION_MAJOR
    ret.push_back(
        "xtl=" + detail::unquote(std::string(QUOTE(XTL_VERSION_MAJOR))) + "." +
        detail::unquote(std::string(QUOTE(XTL_VERSION_MINOR))) + "." +
        detail::unquote(std::string(QUOTE(XTL_VERSION_PATCH))));
#endif

#if defined(XTENSOR_PYTHON_VERSION_MAJOR)
    ret.push_back(
        "xtensor-python=" + detail::unquote(std::string(QUOTE(XTENSOR_PYTHON_VERSION_MAJOR))) +
        "." + detail::unquote(std::string(QUOTE(XTENSOR_PYTHON_VERSION_MINOR))) + "." +
        detail::unquote(std::string(QUOTE(XTENSOR_PYTHON_VERSION_PATCH))));
#endif

    std::sort(ret.begin(), ret.end(), std::greater<std::string>());

    return ret;
}

/**
Return compiler version.
\return List of strings.
*/
inline std::vector<std::string> version_compiler()
{
    std::vector<std::string> ret;

#ifdef __DATE__
    std::string date = detail::unquote(std::string(QUOTE(__DATE__)));
    std::replace(date.begin(), date.end(), ' ', '-');
    ret.push_back("date=" + date);
#endif

#ifdef __APPLE__
    ret.push_back("platform=apple");
#endif

#ifdef __MINGW32__
    ret.push_back("platform=mingw");
#endif

#ifdef __linux__
    ret.push_back("platform=linux");
#endif

#ifdef __clang_version__
    ret.push_back(
        "clang=" + detail::unquote(std::string(QUOTE(__clang_major__))) + "." +
        detail::unquote(std::string(QUOTE(__clang_minor__))) + "." +
        detail::unquote(std::string(QUOTE(__clang_patchlevel__))));
#endif

#ifdef __GNUC__
    ret.push_back(
        "gcc=" + detail::unquote(std::string(QUOTE(__GNUC__))) + "." +
        detail::unquote(std::string(QUOTE(__GNUC_MINOR__))) + "." +
        detail::unquote(std::string(QUOTE(__GNUC_PATCHLEVEL__))));
#endif

#ifdef _MSC_VER
    ret.push_back("msvc=" + std::string(_MSC_VER));
#endif

    // c++ version

#ifdef __cplusplus
    ret.push_back("c++=" + detail::unquote(std::string(QUOTE(__cplusplus))));
#endif

    std::sort(ret.begin(), ret.end(), std::greater<std::string>());

    return ret;
}

} // namespace GMatTensor

#endif
