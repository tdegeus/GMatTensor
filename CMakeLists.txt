cmake_minimum_required(VERSION 3.1)

# Basic settings
# ==============

project(GMatTensor)

option(BUILD_TESTS "${PROJECT_NAME} Build tests" OFF)
option(BUILD_DOCS "${PROJECT_NAME} Build docs" OFF)
option(CHECK_DOCS "${PROJECT_NAME} Check docs (throw error for every warning)" OFF)

# Version
# =======

IF(DEFINED ENV{PKG_VERSION})
    set(GMATTENSOR_VERSION $ENV{PKG_VERSION})
else()
    execute_process(
        COMMAND python -c "from setuptools_scm import get_version; print(get_version())"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GMATTENSOR_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

message(STATUS "Building GMatTensor ${GMATTENSOR_VERSION}")

# Set target
# ==========

find_package(xtensor REQUIRED)

add_library(GMatTensor INTERFACE)

target_include_directories(GMatTensor INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(GMatTensor INTERFACE xtensor)

# Installation
# ============

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" DESTINATION include)

configure_file("include/${PROJECT_NAME}/version.h"
               "${CMAKE_CURRENT_BINARY_DIR}/version.h"
                @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/version.h"
        DESTINATION "include/${PROJECT_NAME}/")

install(TARGETS GMatTensor EXPORT GMatTensor-targets)

install(
    EXPORT GMatTensor-targets
    FILE "${PROJECT_NAME}Targets.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

set(_GMAT ${CMAKE_SIZEOF_VOID_P})
unset(CMAKE_SIZEOF_VOID_P)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${GMATTENSOR_VERSION}
    COMPATIBILITY AnyNewerVersion)

set(CMAKE_SIZEOF_VOID_P ${_GMAT})

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}.pc.in"
               "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc" @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig/")

# Add builds
# ==========

include("${PROJECT_NAME}Config.cmake")
include(CTest)

# Tests

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# Check docs: only this library, throw error for every warning

if(CHECK_DOCS)
    find_package(Doxygen REQUIRED)

    set(DOXYGEN_EXCLUDE_SYMBOLS detail)
    set(DOXYGEN_CASE_SENSE_NAMES YES)
    set(DOXYGEN_QUIET YES)
    set(WARN_AS_ERROR YES)

    set(DOXYGEN_ALIASES
        "rst=\\verbatim embed:rst:leading-asterisk"
        "endrst=\\endverbatim"
        "license=@par License:")

    doxygen_add_docs(check_docs "${CMAKE_CURRENT_SOURCE_DIR}/include")
endif()

# Build docs: include dependencies

if(BUILD_DOCS)
    find_package(Doxygen REQUIRED)

    set(DOXYGEN_EXCLUDE_SYMBOLS detail)
    set(DOXYGEN_GENERATE_TREEVIEW YES)
    set(DOXYGEN_CASE_SENSE_NAMES YES)
    set(DOXYGEN_JAVADOC_AUTOBRIEF YES)
    set(DOXYGEN_MACRO_EXPANSION YES)
    set(DOXYGEN_PREDEFINED IN_DOXYGEN)
    set(DOXYGEN_SOURCE_BROWSER YES)
    set(DOXYGEN_QUIET YES)
    set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)
    set(DOXYGEN_USE_MATHJAX YES)

    set(DOXYGEN_ALIASES
        "rst=\\verbatim embed:rst:leading-asterisk"
        "endrst=\\endverbatim"
        "license=@par License:")

    set(DOXYGEN_STRIP_FROM_INC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/include")

    set(DOXYGEN_STRIP_FROM_PATH "${CMAKE_CURRENT_SOURCE_DIR}/include")

    doxygen_add_docs(docs "${CMAKE_CURRENT_SOURCE_DIR}/include")
endif()
