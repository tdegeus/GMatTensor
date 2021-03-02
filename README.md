# GMatTensor

[![CI](https://github.com/tdegeus/GMatTensor/workflows/CI/badge.svg)](https://github.com/tdegeus/GMatTensor/actions)
[![Doxygen -> gh-pages](https://github.com/tdegeus/GMatTensor/workflows/gh-pages/badge.svg)](https://tdegeus.github.io/GMatTensor)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/gmattensor.svg)](https://anaconda.org/conda-forge/gmattensor)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/python-gmattensor.svg)](https://anaconda.org/conda-forge/python-gmattensor)

Tensor definitions supporting several GMat models.

- Getting started: this readme.
- Documentation: https://tdegeus.github.io/GMatTensor

# Contents

<!-- MarkdownTOC -->

- [Disclaimer](#disclaimer)
- [Functionality](#functionality)
    - [Unit tensors](#unit-tensors)
    - [Tensor operations](#tensor-operations)
    - [Tensor operations \(semi-public API\)](#tensor-operations-semi-public-api)
- [Implementation](#implementation)
    - [C++ and Python](#c-and-python)
- [Installation](#installation)
    - [C++ headers](#c-headers)
        - [Using conda](#using-conda)
        - [From source](#from-source)
    - [Python module](#python-module)
        - [Using conda](#using-conda-1)
        - [From source](#from-source-1)
- [Compiling](#compiling)
    - [Using CMake](#using-cmake)
        - [Example](#example)
        - [Targets](#targets)
        - [Optimisation](#optimisation)
    - [By hand](#by-hand)
    - [Using pkg-config](#using-pkg-config)
- [Change-log](#change-log)
    - [v0.5.0](#v050)
    - [v0.4.0](#v040)
    - [v0.3.0](#v030)
    - [v0.2.0](#v020)
        - [Pointer API](#pointer-api)
    - [v0.1.2](#v012)
    - [v0.1.1](#v011)
    - [v0.1.0](#v010)

<!-- /MarkdownTOC -->

# Disclaimer

This library is free to use under the
[MIT license](https://github.com/tdegeus/GMatTensor/blob/master/LICENSE).
Any additions are very much appreciated, in terms of suggested functionality, code,
documentation, testimonials, word-of-mouth advertisement, etc.
Bug reports or feature requests can be filed on
[GitHub](https://github.com/tdegeus/GMatTensor).
As always, the code comes with no guarantee.
None of the developers can be held responsible for possible mistakes.

Download: 
[.zip file](https://github.com/tdegeus/GMatTensor/zipball/master) |
[.tar.gz file](https://github.com/tdegeus/GMatTensor/tarball/master).

(c - [MIT](https://github.com/tdegeus/GMatTensor/blob/master/LICENSE))
T.W.J. de Geus (Tom) | tom@geus.me | www.geus.me |
[github.com/tdegeus/GMatTensor](https://github.com/tdegeus/GMatTensor)

# Functionality

## Unit tensors

This library implements for a Cartesian coordinate frame in 2d or in 3d:

*   Second (`I2`) and fourth (`I4`) order null tensors:
    -   0<sub>ik</sub> = `02`<sub>ij</sub> *A*<sub>jk</sub>
    -   0<sub>ij</sub> = `04`<sub>ijkl</sub> *A*<sub>lk</sub>
*   Second (`I2`) and fourth (`I4`) order unit tensors:
    -   *A*<sub>ik</sub> = `I2`<sub>ij</sub> *A*<sub>jk</sub>
    -   *A*<sub>ij</sub> = `I4`<sub>ijkl</sub> *A*<sub>lk</sub>
*   Fourth order projection tensors: symmetric, deviatoric, right- and left-transposed:
    -   tr(*A*) = `I2`<sub>ij</sub> *A*<sub>ji</sub>
    -   dev(*A*) = `I4d`<sub>ijkl</sub> *A*<sub>lk</sub>
    -   sym(*A*) = `I4s`<sub>ijkl</sub> *A*<sub>lk</sub>
    -   transpose(*A*) = `I4rt`<sub>ijkl</sub> *A*<sub>lk</sub>

For convenience also find:

*   Second (`I2`) and fourth (`I4`) order random tensors, 
    with each component drawn from a normal distribution.

In addition it provides an `Array<rank>` of unit tensors. 
Suppose that the array is rank three, with shape (R, S, T), then the output is:

*   Second order tensors: (R, S, T, d, d), with `d` the number of dimensions (2 or 3).
*   Fourth order tensors: (R, S, T, d, d, d, d).

E.g.
```cpp
auto A = GMatTensor::Array<3>({4, 5, 6}).O2();
auto A = GMatTensor::Array<3>({4, 5, 6}).O4();
auto A = GMatTensor::Array<3>({4, 5, 6}).I2();
auto A = GMatTensor::Array<3>({4, 5, 6}).I4();
auto A = GMatTensor::Array<3>({4, 5, 6}).II();
auto A = GMatTensor::Array<3>({4, 5, 6}).I4d();
auto A = GMatTensor::Array<3>({4, 5, 6}).I4s();
auto A = GMatTensor::Array<3>({4, 5, 6}).I4rt();
auto A = GMatTensor::Array<3>({4, 5, 6}).I4lt();
```

Given that the arrays are row-major, the tensors or each array component are thus 
stored contiguously in the memory.
This is heavily used to expose all operations below to nd-arrays of tensors.
In particular, all operations are available on raw pointers to a tensor, 
or for nd-arrays of (x)tensors (include a 'plain' tensor with array rank 0).

## Tensor operations

*   Input: 2nd-order tensor (e.g. `(R, S, T, d, d)`). 
    Returns: scalar (e.g. `(R, S, T)`).
    -   tr(*A*) = `Trace(A)`
    -   tr(*A*) / *d* = `Hydrostatic(A)`
    -   det(*A*) = `Det(A)`   
    -   *A*<sub>ij</sub> B<sub>ji</sub> = *A* : *B* = `A2_ddot_B2(A, B)`
    -   *A*<sub>ij</sub> B<sub>ji</sub> = *A* : *B* = `A2s_ddot_B2s(A, B)`
        (both tensors assumed symmetric, no assertion).   
    -   dev(*A*)<sub>ij</sub> dev(A)<sub>ji</sub> = `Norm_deviatoric(A)`
*   Input: 2nd-order tensor (e.g. `(R, S, T, d, d)`). 
    Returns: 2nd-order tensor (e.g. `(R, S, T, d, d)`).
    -   dev(*A*) = `A` - `Hydrostatic(A)` * `I2`
    -   dev(*A*) = `Deviatoric(A)`
    -   sym(*A*) = `Sym(A)`
    -   *A*<sup>-1</sup> = `Inv(A)`
    -   log(*A*) = `Logs(A)`
        (tensor assumed symmetric, no assertion).
    -   *C*<sub>ik</sub> *A*<sub>ij</sub> A<sub>kj</sub> = *A* . *A*<sup>T</sup> 
        = `A2_dot_A2T(A, B)`
    -   *C*<sub>ik</sub> *A*<sub>ij</sub> B<sub>jk</sub> = *A* . *B* = `A2_dot_B2(A, B)`
    -   *C*<sub>ij</sub> *A*<sub>ijkl</sub> B<sub>lk</sub> = *A* : *B* = `A4_ddot_B2(A, B)`
*   Input: 2nd-order tensor (e.g. `(R, S, T, d, d)`) 
    or 4th-order tensor (e.g. `(R, S, T, d, d, d)`). 
    Returns: 4th-order tensor (e.g. `(R, S, T, d, d, d)`).
    -   *C*<sub>ijkl</sub> *A*<sub>ij</sub> B<sub>kl</sub> = *A* * *B* = `A2_dyadic_B2(A, B)`
    -   *C*<sub>ijkm</sub> *A*<sub>ijkl</sub> B<sub>lm</sub> = *A* . *B* = `A4_dot_B2(A, B)`

Note that the output has:

-   In the case of an input tensor `(d, d)` the output can  be a rank-zero matrix.
    To get a scalar do e.g. `Hydrostatic(A)()`.

Furthermore note that:

*   Functions whose name starts with a capital letter allocate and return their output.
*   Functions whose name starts with a small letter require their output as final
    input parameter(s), which is changed in-place.

## Tensor operations (semi-public API)

A semi-public API exists that is mostly aimed to support *GMat* implementation.
These involve pure-tensor operations based the input (and the output) as a pointer, 
using the following storage convention:

*   Cartesian2d: (xx, xy, yx, yy).
*   Cartesian3d: (xx, xy, xz, yx, yy, yz, zx, zy, zz).

This part of the API does not support arrays of tensors.
In addition to the pointer equivalent (or in fact core) of the above function, 
the following functions are available:

*   `Hydrostatic_deviatoric`: Return the hydrostatic part of a tensor, 
    and write the deviatoric part to a pointer.
*   `A4_ddot_B4_ddot_C4`: *A* : *B* : *C*
*   `A2_dot_B2_dot_C2T`: *A* . *B* . *C*<sup>T</sup> 
*   `eigs`: Compute the eigen values and eigen vectors for a symmetric 2nd order tensor
    (no assertion on symmetry).
*   `from_eigs`: The reverse operation from `eigs`, for a symmetric 2nd order tensor
    (no assertion on symmetry).

# Implementation

## C++ and Python

The code is a C++ header-only library (see [installation notes](#c-headers)), 
but a Python module is also provided (see [installation notes](#python-module)).
The interfaces are identical except:

+   All *xtensor* objects (`xt::xtensor<...>`) are *NumPy* arrays in Python. 
    Overloading based on rank is also available in Python.
+   The Python module cannot change output objects in-place: 
    only functions whose name starts with a capital letter are included, see below.
+   All `::` in C++ are `.` in Python.

# Installation

## C++ headers

### Using conda

```bash
conda install -c conda-forge gmattensor
```

### From source

```bash
# Download GMatTensor
git checkout https://github.com/tdegeus/GMatTensor.git
cd GMatTensor

# Install headers, CMake and pkg-config support
cmake .
make install
```

## Python module

### Using conda

```bash
conda install -c conda-forge python-gmattensor
```

Note that *xsimd* and hardware optimisations are **not enabled**. 
To enable them you have to compile on your system, as is discussed next.

### From source

>   You need *xtensor*, *pyxtensor* and optionally *xsimd* as prerequisites. 
>   Additionally, Python needs to know how to find them. 
>   The easiest is to use *conda* to get the prerequisites:
> 
>   ```bash
>   conda install -c conda-forge pyxtensor
>   conda install -c conda-forge xsimd
>   ```
>   
>   If you then compile and install with the same environment 
>   you should be good to go. 
>   Otherwise, a bit of manual labour might be needed to
>   treat the dependencies.

```bash
# Download GMatTensor
git checkout https://github.com/tdegeus/GMatTensor.git
cd GMatTensor

# Compile and install the Python module
python setup.py build
python setup.py install
# OR you can use one command (but with less readable output)
python -m pip install .
```

# Compiling

## Using CMake

### Example

Using *GMatTensor* your `CMakeLists.txt` can be as follows

```cmake
cmake_minimum_required(VERSION 3.1)
project(example)
find_package(GMatTensor REQUIRED)
add_executable(example example.cpp)
target_link_libraries(example PRIVATE GMatTensor)
```

### Targets

The following targets are available:

*   `GMatTensor`
    Includes *GMatTensor* and the *xtensor* dependency.

*   `GMatTensor::assert`
    Enables assertions by defining `GMATTENSOR_ENABLE_ASSERT`.

*   `GMatTensor::debug`
    Enables all assertions by defining 
    `GMATTENSOR_ENABLE_ASSERT` and `XTENSOR_ENABLE_ASSERT`.

*   `GMatTensor::compiler_warings`
    Enables compiler warnings (generic).

### Optimisation

It is advised to think about compiler optimization and enabling *xsimd*.
Using *CMake* this can be done using the `xtensor::optimize` and `xtensor::use_xsimd` targets.
The above example then becomes:

```cmake
cmake_minimum_required(VERSION 3.1)
project(example)
find_package(GMatTensor REQUIRED)
add_executable(example example.cpp)
target_link_libraries(example PRIVATE 
    GMatTensor 
    xtensor::optimize 
    xtensor::use_xsimd)
```

See the [documentation of xtensor](https://xtensor.readthedocs.io/en/latest/) concerning optimization.

## By hand

Presuming that the compiler is `c++`, compile using:

```
c++ -I/path/to/GMatTensor/include ...
```

Note that you have to take care of the *xtensor* dependency, the C++ version, optimization, 
enabling *xsimd*, ...

## Using pkg-config

Presuming that the compiler is `c++`, compile using:

```
c++ `pkg-config --cflags GMatTensor` ...
```

Note that you have to take care of the *xtensor* dependency, the C++ version, optimization, 
enabling *xsimd*, ...

# Change-log

## v0.5.0

*   Extending tests.
*   Added `Sym`.
*   Added `A2_dot_B2` to *Cartesian2d*.
*   API change: **all** functions that return output, including scalars, now start
    with a capital letter.
*   Updated readme.

## v0.4.0

*   Adding "logs".
*   Unit tensors: now all using pointer API.
*   Porting all public APIs to array of tensors.
*   Pointer API: less aggressive templating.
*   API change: Renaming "equivalent_deviatioric" -> "norm_deviatoric".
*   Introducing null tensors `GMatTensor::Cartesian3d::O2` and `GMatTensor::Cartesian3d::O4`
    (also for Cartesian2d).
*   Adding several new tensor operations / products.
*   Adding more public xtensor interface for tensor products. 
    The aim is mostly to allow the user to be quick and dirty, e.g. when testing.
*   Formatting tests with the latter new API.

## v0.3.0

*   Relaxing assumption on symmetry for dev(A) : dev(A).
*   Adding symmetric only function `A2s_ddot_B2s` to pointer API.

## v0.2.0

### Pointer API

*   Making pointer explicit (template `T` -> `T*`).
*   Adding zero and unit tensors.
*   Add dyadic product between two second order tensors.

## v0.1.2

*   Adding stride members to `Array`.

## v0.1.1

*   Improved sub-classing support

## v0.1.0 

Transfer from other libraries.
