
cmake_minimum_required(VERSION 3.0)

# Basic settings
# ==============

project(GMatTensor)

option(BUILD_TESTS "${PROJECT_NAME} Build tests" OFF)
option(BUILD_EXAMPLES "${PROJECT_NAME} Build examples" OFF)

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
target_compile_features(GMatTensor INTERFACE cxx_std_17)

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
    FILE GMatTensorTargets.cmake
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/GMatTensor")

set(_GMAT ${CMAKE_SIZEOF_VOID_P})
unset(CMAKE_SIZEOF_VOID_P)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/GMatTensorConfigVersion.cmake"
    VERSION ${GMATTENSOR_VERSION}
    COMPATIBILITY AnyNewerVersion)

set(CMAKE_SIZEOF_VOID_P ${_GMAT})

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/GMatTensorConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/GMatTensorConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/GMatTensor")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/GMatTensor.pc.in"
               "${CMAKE_CURRENT_BINARY_DIR}/GMatTensor.pc" @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/GMatTensor.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig/")

# Add builds
# ==========

include("GMatTensorConfig.cmake")

set(CTEST_TEST_TARGET_ALIAS RUN_ALL_TESTS)
include(CTest)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
