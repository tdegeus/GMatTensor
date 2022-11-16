/**
 * @file
 * @copyright Copyright. Tom de Geus. All rights reserved.
 * @license This project is released under the MIT License.
 */

#ifndef GMATTENSOR_VERSION_H
#define GMATTENSOR_VERSION_H

#include "config.h"

/**
 * Current version.
 *
 * Either:
 *
 * -   Configure using CMake at install time. Internally uses::
 *
 *         python -c "from setuptools_scm import get_version; print(get_version())"
 *
 * -   Define externally using::
 *
 *         -DGMATTENSOR_VERSION="`python -c "from setuptools_scm import get_version;
 * print(get_version())"`"
 *
 *     From the root of this project. This is what ``setup.py`` does.
 *
 * Note that both ``CMakeLists.txt`` and ``setup.py`` will construct the version using
 * ``setuptools_scm``. Tip: use the environment variable ``SETUPTOOLS_SCM_PRETEND_VERSION`` to
 * overwrite the automatic version.
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

inline std::string replace(std::string str, const std::string& from, const std::string& to)
{
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
    return str;
}

} // namespace detail

/**
 * Return version string, e.g. `"0.8.0"`
 * @return Version string.
 */
inline std::string version()
{
    return detail::unquote(std::string(QUOTE(GMATTENSOR_VERSION)));
}

/**
 * Return versions of this library and of all of its major dependencies.
 * The output is a list of strings:
 *
 *     "gmattensor=0.8.0",
 *     "xtensor=0.20.1",
 *     ...
 *
 * @param greedy Add as much as possible version information, even if the library is not used here.
 * @return List of strings.
 */
inline std::vector<std::string> version_dependencies(bool greedy = true)
{
    UNUSED(greedy);

    std::vector<std::string> ret;
    ret.push_back("gmattensor=" + GMatTensor::version());

    // Goose suite

#ifdef GMATELASTIC_VERSION
    if (greedy) {
        ret.push_back("gmatelastic=" + detail::unquote(std::string(QUOTE(GMATELASTIC_VERSION))));
    }
#endif

#ifdef GMATELASTOPLASTIC_VERSION
    if (greedy) {
        ret.push_back(
            "gmatelastoplastic=" + detail::unquote(std::string(QUOTE(GMATELASTOPLASTIC_VERSION))));
    }
#endif

#ifdef GMATELASTOPLASTICQPOT_VERSION
    if (greedy) {
        ret.push_back(
            "gmatelastoplasticqpot=" +
            detail::unquote(std::string(QUOTE(GMATELASTOPLASTICQPOT_VERSION))));
    }
#endif

#ifdef GMATELASTOPLASTICQPOT3D_VERSION
    if (greedy) {
        ret.push_back(
            "gmatelastoplasticqpot3d=" +
            detail::unquote(std::string(QUOTE(GMATELASTOPLASTICQPOT3D_VERSION))));
    }
#endif

#ifdef GMATELASTOPLASTICFINITESTRAINSIMO_VERSION
    if (greedy) {
        ret.push_back(
            "gmatelastoplasticfinitestrainsimo=" +
            detail::unquote(std::string(QUOTE(GMATELASTOPLASTICFINITESTRAINSIMO_VERSION))));
    }
#endif

#ifdef GMATNONLINEARELASTIC_VERSION
    if (greedy) {
        ret.push_back(
            "gmatnonlinearelastic=" +
            detail::unquote(std::string(QUOTE(GMATNONLINEARELASTIC_VERSION))));
    }
#endif

#ifdef GOOSEFEM_VERSION
    if (greedy) {
        ret.push_back("goosefem=" + detail::unquote(std::string(QUOTE(GOOSEFEM_VERSION))));
    }
#endif

#ifdef GOOSEEYE_VERSION
    if (greedy) {
        ret.push_back("goosefem=" + detail::unquote(std::string(QUOTE(GOOSEEYE_VERSION))));
    }
#endif

#ifdef QPOT_VERSION
    if (greedy) {
        ret.push_back("qpot=" + detail::unquote(std::string(QUOTE(QPOT_VERSION))));
    }
#endif

#ifdef PRRNG_VERSION
    if (greedy) {
        ret.push_back("prrng=" + detail::unquote(std::string(QUOTE(PRRNG_VERSION))));
    }
#endif

#ifdef FRICTIONQPOTSPRINGBLOCK_VERSION
    if (greedy) {
        ret.push_back(
            "frictionqpotspringblock=" +
            detail::unquote(std::string(QUOTE(FRICTIONQPOTSPRINGBLOCK_VERSION))));
    }
#endif

#ifdef FRICTIONQPOTFEM_VERSION
    if (greedy) {
        ret.push_back(
            "frictionqpotfem=" + detail::unquote(std::string(QUOTE(FRICTIONQPOTFEM_VERSION))));
    }
#endif

#ifdef CPPPATH_VERSION
    if (greedy) {
        ret.push_back("cpppath=" + detail::unquote(std::string(QUOTE(CPPPATH_VERSION))));
    }
#endif

#ifdef CPPCOLORMAP_VERSION
    if (greedy) {
        ret.push_back("cppcolormap=" + detail::unquote(std::string(QUOTE(CPPCOLORMAP_VERSION))));
    }
#endif

    // Boost

#ifdef BOOST_VERSION
    if (greedy) {
        ret.push_back(
            "boost=" + detail::unquote(std::to_string(BOOST_VERSION / 100000)) + "." +
            detail::unquote(std::to_string((BOOST_VERSION / 100) % 1000)) + "." +
            detail::unquote(std::to_string(BOOST_VERSION % 100)));
    }
#endif

    // Eigen

#ifdef EIGEN_WORLD_VERSION
    if (greedy) {
        ret.push_back(
            "eigen=" + detail::unquote(std::string(QUOTE(EIGEN_WORLD_VERSION))) + "." +
            detail::unquote(std::string(QUOTE(EIGEN_MAJOR_VERSION))) + "." +
            detail::unquote(std::string(QUOTE(EIGEN_MINOR_VERSION))));
    }
#endif

    // xtensor suite

#ifdef XTENSOR_VERSION_MAJOR
    ret.push_back(
        "xtensor=" + detail::unquote(std::string(QUOTE(XTENSOR_VERSION_MAJOR))) + "." +
        detail::unquote(std::string(QUOTE(XTENSOR_VERSION_MINOR))) + "." +
        detail::unquote(std::string(QUOTE(XTENSOR_VERSION_PATCH))));
#endif

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
 * Return information on the compiler, the platform, the C++ standard, and the compilation data.
 * @return List of strings.
 */
inline std::vector<std::string> version_compiler()
{
    std::vector<std::string> ret;

#ifdef __DATE__
    std::string date = detail::unquote(std::string(QUOTE(__DATE__)));
    ret.push_back("date=" + detail::replace(detail::replace(date, " ", "-"), "--", "-"));
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
    ret.push_back("msvc=" + std::to_string(_MSC_VER));
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
