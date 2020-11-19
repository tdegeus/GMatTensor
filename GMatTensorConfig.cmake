# GMatTensor cmake module
#
# This module sets the target:
#
#     GMatTensor
#
# In addition, it sets the following variables:
#
#     GMatTensor_FOUND - true if the library is found
#     GMatTensor_VERSION - the library's version
#     GMatTensor_INCLUDE_DIRS - directory containing the library's headers
#
# The following support targets are defined to simplify things:
#
#     GMatTensor::compiler_warnings - enable compiler warnings
#     GMatTensor::assert - enable library assertions
#     GMatTensor::debug - enable all assertions (slow)

include(CMakeFindDependencyMacro)

# Define target "GMatTensor"

if(NOT TARGET GMatTensor)
    include("${CMAKE_CURRENT_LIST_DIR}/GMatTensorTargets.cmake")
endif()

# Define "GMatTensor_INCLUDE_DIRS"

get_target_property(
    GMatTensor_INCLUDE_DIRS
    GMatTensor
    INTERFACE_INCLUDE_DIRECTORIES)

# Find dependencies

find_dependency(xtensor)

# Define support target "GMatTensor::compiler_warnings"

if(NOT TARGET GMatTensor::compiler_warnings)
    add_library(GMatTensor::compiler_warnings INTERFACE IMPORTED)
    if(MSVC)
        set_property(
            TARGET GMatTensor::compiler_warnings
            PROPERTY INTERFACE_COMPILE_OPTIONS
            /W4)
    else()
        set_property(
            TARGET GMatTensor::compiler_warnings
            PROPERTY INTERFACE_COMPILE_OPTIONS
            -Wall -Wextra -pedantic -Wno-unknown-pragmas)
    endif()
endif()

# Define support target "GMatTensor::assert"

if(NOT TARGET GMatTensor::assert)
    add_library(GMatTensor::assert INTERFACE IMPORTED)
    set_property(
        TARGET GMatTensor::assert
        PROPERTY INTERFACE_COMPILE_DEFINITIONS
        GMATTENSOR_ENABLE_ASSERT)
endif()

# Define support target "GMatTensor::debug"

if(NOT TARGET GMatTensor::debug)
    add_library(GMatTensor::debug INTERFACE IMPORTED)
    set_property(
        TARGET GMatTensor::debug
        PROPERTY INTERFACE_COMPILE_DEFINITIONS
        XTENSOR_ENABLE_ASSERT GMATTENSOR_ENABLE_ASSERT)
endif()
