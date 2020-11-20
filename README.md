# GMatTensor

[![CI](https://github.com/tdegeus/GMatTensor/workflows/CI/badge.svg)](https://github.com/tdegeus/GMatTensor/actions)

Tensor definitions supporting several GMat models.

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

# Implementation

## Naming convention

*   Functions whose name starts with a capital letter allocate and return their output.
*   Functions whose name starts with a small letter require their output as final
    input parameter(s), which is changed in-place.

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
