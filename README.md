# GMatTensor

[![CI](https://github.com/tdegeus/GMatTensor/workflows/CI/badge.svg)](https://github.com/tdegeus/GMatTensor/actions)

Tensor definitions supporting several GMat models.

# Contests

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

*   Second (`I2`) and fourth (`I4`) order unit tensors:
    -   *A*<sub>ik</sub> = `I2`<sub>ij</sub> *A*<sub>jk</sub>
    -   *A*<sub>ij</sub> = `I4`<sub>ijkl</sub> *A*<sub>lk</sub>
*   Fourth order projection tensors: symmetric, deviatoric, right- and left-transposed:
    -   tr(*A*) = `I2`<sub>ij</sub> *A*<sub>ji</sub>
    -   dev(*A*) = `I4d`<sub>ijkl</sub> *A*<sub>lk</sub>
    -   sym(*A*) = `I4s`<sub>ijkl</sub> *A*<sub>lk</sub>
    -   transpose(*A*) = `I4rt`<sub>ijkl</sub> *A*<sub>lk</sub>

In addition it provides an `Array<rank>` of unit tensors. 
Suppose that the array is rank three, with shape (R, S, T), then the output is:

*   Second order tensors: (R, S, T, d, d), with `d` the number of dimensions (2 or 3).
*   Fourth order tensors: (R, S, T, d, d, d, d).

E.g.
```cpp
auto i = GMatTensor::Array<3>({4, 5, 6}).I2();
auto i = GMatTensor::Array<3>({4, 5, 6}).I4();
auto i = GMatTensor::Array<3>({4, 5, 6}).II();
auto i = GMatTensor::Array<3>({4, 5, 6}).I4d();
auto i = GMatTensor::Array<3>({4, 5, 6}).I4s();
auto i = GMatTensor::Array<3>({4, 5, 6}).I4rt();
auto i = GMatTensor::Array<3>({4, 5, 6}).I4lt();
```

Given that the arrays are row-major, the tensors or each array component are thus 
stored contiguously in the memory.

## Tensor operations

*   Taking the hydrostatic and deviatoric of a(n array of) second order tensor(s).
    -   tr(*A*) / *d* = `Hydrostatic(A)`
    -   dev(*A*) = A - `Hydrostatic(A)` * `I2`
    -   dev(*A*) = `Deviatoric(A)`
*   An equivalent value of the deviatoric part of a(n array of) second order tensor(s).
    -   dev(*A*)<sub>ij</sub> dev(A)<sub>ji</sub> = `Equivalent_deviatoric(A)`

Note that the output has:

*   The same rank of the input: `Deviatoric`.
*   The rank of the input minus two: `Hydrostatic`, `Equivalent_deviatoric`.
    -   In the case of an input tensor (input shape `(d, d)`), this results in a rank-zero matrix.
        To get a scalar do e.g. `Hydrostatic(A)()`.
    -   In the case of an input array of tensors this results in the array. 
        E.g. for input shape `(R, S, T, d, d)`, the output shape is `(R, S, T)`.

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
The following functions are available:

*   `t = trace(A)`: return the trace *t* = *A*<sub>ii</sub>.
*   `m = hydrostatic_deviatoric(A, B)`: returns the hydrostatic part of *A* (`== trace(A) / d`),
    and writes its deviatoric part to *B*.
*   `deviatoric_ddot_deviatoric(A)`: returns the double-dot product of the deviatoric part of *A*, 
    dev(*A*)<sub>ij</sub> dev(*A*)<sub>ji</sub>.
*   `A2_ddot_B2`: returns the double-dot product of *A* and *B*,
    *A*<sub>ij</sub> *B*<sub>ji</sub>.

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
    Enables assertions by defining `GMATELASTOPLASTICQPOT_ENABLE_ASSERT`.

*   `GMatTensor::debug`
    Enables all assertions by defining 
    `GMATELASTOPLASTICQPOT_ENABLE_ASSERT` and `XTENSOR_ENABLE_ASSERT`.

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
