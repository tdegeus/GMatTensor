
cmake_minimum_required(VERSION 3.0)

# Basic settings
# ==============

project(GMatTensor)

option(BUILD_TESTS "${PROJECT_NAME} Build tests" OFF)
option(BUILD_EXAMPLES "${PROJECT_NAME} Build examples" OFF)

# Version
# =======

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/GMatTensor/config.h"
             _GMAT_version_defines
     REGEX "#define GMATTENSOR_VERSION_(MAJOR|MINOR|PATCH)")

foreach(ver ${_GMAT_version_defines})
    if(ver MATCHES "#define GMATTENSOR_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(_GMAT_VERSION_${CMAKE_MATCH_1}
            "${CMAKE_MATCH_2}"
            CACHE INTERNAL "")
    endif()
endforeach()

set(_GMAT_VERSION ${_GMAT_VERSION_MAJOR}.${_GMAT_VERSION_MINOR}.${_GMAT_VERSION_PATCH})

message(STATUS "Building GMatTensor v${_GMAT_VERSION}")

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

install(TARGETS GMatTensor EXPORT GMatTensor-targets)

install(
    EXPORT GMatTensor-targets
    FILE GMatTensorTargets.cmake
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/GMatTensor")

set(_GMAT ${CMAKE_SIZEOF_VOID_P})
unset(CMAKE_SIZEOF_VOID_P)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/GMatTensorConfigVersion.cmake"
    VERSION ${_GMAT_VERSION}
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
