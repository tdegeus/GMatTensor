
#include <catch2/catch.hpp>
#include <xtensor/xrandom.hpp>
#include <GMatTensor/Cartesian3d.h>

#define ISCLOSE(a,b) REQUIRE_THAT((a), Catch::WithinAbs((b), 1.e-12));

namespace GM = GMatTensor::Cartesian3d;

template <class T, class S>
S A4_ddot_B2(const T& A, const S& B)
{
    S C = xt::empty<double>({3, 3});
    C.fill(0.0);

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 3; k++) {
                for (size_t l = 0; l < 3; l++) {
                    C(i, j) += A(i, j, k, l) * B(l, k);
                }
            }
        }
    }

    return C;
}

TEST_CASE("GMatTensor::Cartesian3d", "Cartesian3d.h")
{

SECTION("Id")
{
    GM::Tensor2 A = xt::random::randn<double>({3, 3});
    GM::Tensor2 I = GM::I2();
    GM::Tensor4 Id = GM::I4d();
    GM::Tensor4 Is = GM::I4s();
    A = A4_ddot_B2(Is, A);
    REQUIRE(xt::allclose(A4_ddot_B2(Id, A), A - GM::Hydrostatic(A) * I));
}

SECTION("Deviatoric - Tensor2")
{
    GM::Tensor2 A = xt::random::randn<double>({3, 3});
    GM::Tensor2 B = A;
    double tr = B(0, 0) + B(1, 1) + B(2, 2);
    B(0, 0) -= tr / 3.0;
    B(1, 1) -= tr / 3.0;
    B(2, 2) -= tr / 3.0;
    REQUIRE(xt::allclose(GM::Deviatoric(A), B));
}

SECTION("Deviatoric - List")
{
    GM::Tensor2 A = xt::random::randn<double>({3, 3});
    GM::Tensor2 B = A;
    double tr = B(0, 0) + B(1, 1) + B(2, 2);
    B(0, 0) -= tr / 3.0;
    B(1, 1) -= tr / 3.0;
    B(2, 2) -= tr / 3.0;
    auto M = xt::xtensor<double,3>::from_shape({3, 3, 3});
    auto R = xt::xtensor<double,3>::from_shape(M.shape());
    for (size_t i = 0; i < M.shape(0); ++i) {
        xt::view(M, i, xt::all(), xt::all()) = static_cast<double>(i) * A;
        xt::view(R, i, xt::all(), xt::all()) = static_cast<double>(i) * B;
    }
    REQUIRE(xt::allclose(GM::Deviatoric(M), R));
}

SECTION("Deviatoric - Matrix")
{
    GM::Tensor2 A = xt::random::randn<double>({3, 3});
    GM::Tensor2 B = A;
    double tr = B(0, 0) + B(1, 1) + B(2, 2);
    B(0, 0) -= tr / 3.0;
    B(1, 1) -= tr / 3.0;
    B(2, 2) -= tr / 3.0;
    auto M = xt::xtensor<double,4>::from_shape({3, 4, 3, 3});
    auto R = xt::xtensor<double,4>::from_shape(M.shape());
    for (size_t i = 0; i < M.shape(0); ++i) {
        for (size_t j = 0; j < M.shape(1); ++j) {
            xt::view(M, i, j, xt::all(), xt::all()) = static_cast<double>(i * M.shape(1) + j) * A;
            xt::view(R, i, j, xt::all(), xt::all()) = static_cast<double>(i * M.shape(1) + j) * B;
        }
    }
    REQUIRE(xt::allclose(GM::Deviatoric(M), R));
}

SECTION("Hydrostatic - Tensor2")
{
    GM::Tensor2 A = xt::random::randn<double>({3, 3});
    A(0, 0) = 1.0;
    A(1, 1) = 1.0;
    A(2, 2) = 1.0;
    REQUIRE(GM::Hydrostatic(A)() == Approx(1.0));
}

SECTION("Hydrostatic - List")
{
    GM::Tensor2 A = xt::random::randn<double>({3, 3});
    A(0, 0) = 1.0;
    A(1, 1) = 1.0;
    A(2, 2) = 1.0;
    auto M = xt::xtensor<double,3>::from_shape({3, 3, 3});
    auto R = xt::xtensor<double,1>::from_shape({M.shape(0)});
    for (size_t i = 0; i < M.shape(0); ++i) {
        xt::view(M, i, xt::all(), xt::all()) = static_cast<double>(i) * A;
        R(i) = static_cast<double>(i);
    }
    REQUIRE(xt::allclose(GM::Hydrostatic(M), R));
}

SECTION("Hydrostatic - Matrix")
{
    GM::Tensor2 A = xt::random::randn<double>({3, 3});
    A(0, 0) = 1.0;
    A(1, 1) = 1.0;
    A(2, 2) = 1.0;
    auto M = xt::xtensor<double,4>::from_shape({3, 4, 3, 3});
    auto R = xt::xtensor<double,2>::from_shape({M.shape(0), M.shape(1)});
    for (size_t i = 0; i < M.shape(0); ++i) {
        for (size_t j = 0; j < M.shape(1); ++j) {
            xt::view(M, i, j, xt::all(), xt::all()) = static_cast<double>(i * M.shape(1) + j) * A;
            R(i, j) = static_cast<double>(i * M.shape(1) + j);
        }
    }
    REQUIRE(xt::allclose(GM::Hydrostatic(M), R));
}

SECTION("Equivalent_deviatoric - Tensor2")
{
    GM::Tensor2 A = xt::zeros<double>({3, 3});
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    double factor = 2.3;
    REQUIRE(GM::Equivalent_deviatoric(A, factor)() == Approx(std::sqrt(2.0 * factor)));
}

SECTION("Equivalent_deviatoric - List")
{
    GM::Tensor2 A = xt::zeros<double>({3, 3});
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    double factor = 2.3;
    auto M = xt::xtensor<double,3>::from_shape({3, 3, 3});
    auto R = xt::xtensor<double,1>::from_shape({M.shape(0)});
    for (size_t i = 0; i < M.shape(0); ++i) {
        xt::view(M, i, xt::all(), xt::all()) = static_cast<double>(i) * A;
        R(i) = static_cast<double>(i) * std::sqrt(2.0 * factor);
    }
    REQUIRE(xt::allclose(GM::Equivalent_deviatoric(M, factor), R));
}

SECTION("Equivalent_deviatoric - Matrix")
{
    GM::Tensor2 A = xt::zeros<double>({3, 3});
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    double factor = 2.3;
    auto M = xt::xtensor<double,4>::from_shape({3, 4, 3, 3});
    auto R = xt::xtensor<double,2>::from_shape({M.shape(0), M.shape(1)});
    for (size_t i = 0; i < M.shape(0); ++i) {
        for (size_t j = 0; j < M.shape(1); ++j) {
            xt::view(M, i, j, xt::all(), xt::all()) = static_cast<double>(i * M.shape(1) + j) * A;
            R(i, j) = static_cast<double>(i * M.shape(1) + j) * std::sqrt(2.0 * factor);
        }
    }
    REQUIRE(xt::allclose(GM::Equivalent_deviatoric(M, factor), R));
}

}
