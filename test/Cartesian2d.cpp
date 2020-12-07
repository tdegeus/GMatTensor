
#include <catch2/catch.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>
#include <GMatTensor/Cartesian2d.h>

namespace GM = GMatTensor::Cartesian2d;

TEST_CASE("GMatTensor::Cartesian2d", "Cartesian2d.h")
{
    SECTION("I4")
    {
        auto A = GM::Random2();
        auto I = GM::I4();
        REQUIRE(xt::allclose(GM::A4_ddot_B2(I, A), A));
    }

    SECTION("I4s")
    {
        auto A = GM::Random2();
        auto Is = GM::I4s();
        REQUIRE(xt::allclose(GM::A4_ddot_B2(Is, A), 0.5 * (A + xt::transpose(A))));
    }

    SECTION("I4d")
    {
        auto A = GM::Random2();
        auto I = GM::I2();
        auto Id = GM::I4d();
        auto B = xt::eval(0.5 * (A + xt::transpose(A)));
        REQUIRE(xt::allclose(GM::A4_ddot_B2(Id, A), B - GM::Hydrostatic(B) * I));
    }

    SECTION("trace")
    {
        auto A = GM::Random2();
        A(0, 0) = 1.0;
        A(1, 1) = 1.0;
        REQUIRE(GM::trace(A) == Approx(2.0));
    }

    SECTION("A2_ddot_B2")
    {
        auto A = GM::O2();
        A(0, 1) = 1.0;
        A(1, 0) = 1.0;
        REQUIRE(GM::A2_ddot_B2(A, A) == Approx(2.0));
    }

    SECTION("A2s_ddot_B2s")
    {
        auto A = GM::O2();
        A(0, 1) = 1.0;
        A(1, 0) = 1.0;
        REQUIRE(GM::A2s_ddot_B2s(A, A) == Approx(2.0));
    }

    SECTION("A2_dyadic_B2")
    {
        auto I2 = GM::I2();
        REQUIRE(xt::allclose(GM::A2_dyadic_B2(I2, I2), GM::II()));
    }

    SECTION("A4_ddot_B2")
    {
        auto A = GM::Random2();
        auto Is = GM::I4s();
        auto B = xt::eval(0.5 * (A + xt::transpose(A)));
        REQUIRE(xt::allclose(GM::A4_ddot_B2(Is, A), B));
    }

    SECTION("Deviatoric - Tensor2")
    {
        auto A = GM::Random2();
        auto B = A;
        double tr = B(0, 0) + B(1, 1);
        B(0, 0) -= 0.5 * tr;
        B(1, 1) -= 0.5 * tr;
        REQUIRE(xt::allclose(GM::Deviatoric(A), B));
    }

    SECTION("Deviatoric - List")
    {
        auto A = GM::Random2();
        auto B = A;
        double tr = B(0, 0) + B(1, 1);
        B(0, 0) -= 0.5 * tr;
        B(1, 1) -= 0.5 * tr;
        auto M = xt::xtensor<double, 3>::from_shape({3, 2, 2});
        auto R = xt::xtensor<double, 3>::from_shape(M.shape());
        for (size_t i = 0; i < M.shape(0); ++i) {
            xt::view(M, i, xt::all(), xt::all()) = static_cast<double>(i) * A;
            xt::view(R, i, xt::all(), xt::all()) = static_cast<double>(i) * B;
        }
        REQUIRE(xt::allclose(GM::Deviatoric(M), R));
    }

    SECTION("Deviatoric - Matrix")
    {
        auto A = GM::Random2();
        auto B = A;
        double tr = B(0, 0) + B(1, 1);
        B(0, 0) -= 0.5 * tr;
        B(1, 1) -= 0.5 * tr;
        auto M = xt::xtensor<double, 4>::from_shape({3, 4, 2, 2});
        auto R = xt::xtensor<double, 4>::from_shape(M.shape());
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
        auto A = GM::Random2();
        A(0, 0) = 1.0;
        A(1, 1) = 1.0;
        REQUIRE(GM::Hydrostatic(A)() == Approx(1.0));
    }

    SECTION("Hydrostatic - List")
    {
        auto A = GM::Random2();
        A(0, 0) = 1.0;
        A(1, 1) = 1.0;
        auto M = xt::xtensor<double, 3>::from_shape({3, 2, 2});
        auto R = xt::xtensor<double, 1>::from_shape({M.shape(0)});
        for (size_t i = 0; i < M.shape(0); ++i) {
            xt::view(M, i, xt::all(), xt::all()) = static_cast<double>(i) * A;
            R(i) = static_cast<double>(i);
        }
        REQUIRE(xt::allclose(GM::Hydrostatic(M), R));
    }

    SECTION("Hydrostatic - Matrix")
    {
        auto A = GM::Random2();
        A(0, 0) = 1.0;
        A(1, 1) = 1.0;
        auto M = xt::xtensor<double, 4>::from_shape({3, 4, 2, 2});
        auto R = xt::xtensor<double, 2>::from_shape({M.shape(0), M.shape(1)});
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                xt::view(M, i, j, xt::all(), xt::all()) = static_cast<double>(i * M.shape(1) + j) * A;
                R(i, j) = static_cast<double>(i * M.shape(1) + j);
            }
        }
        REQUIRE(xt::allclose(GM::Hydrostatic(M), R));
    }

    SECTION("Norm_deviatoric - Tensor2")
    {
        auto A = GM::O2();
        A(0, 1) = 1.0;
        A(1, 0) = 1.0;
        REQUIRE(GM::Norm_deviatoric(A)() == Approx(std::sqrt(2.0)));
    }

    SECTION("Norm_deviatoric - List")
    {
        auto A = GM::O2();
        A(0, 1) = 1.0;
        A(1, 0) = 1.0;
        auto M = xt::xtensor<double, 3>::from_shape({3, 2, 2});
        auto R = xt::xtensor<double, 1>::from_shape({M.shape(0)});
        for (size_t i = 0; i < M.shape(0); ++i) {
            xt::view(M, i, xt::all(), xt::all()) = static_cast<double>(i) * A;
            R(i) = static_cast<double>(i) * std::sqrt(2.0);
        }
        REQUIRE(xt::allclose(GM::Norm_deviatoric(M), R));
    }

    SECTION("Norm_deviatoric - Matrix")
    {
        auto A = GM::O2();
        A(0, 1) = 1.0;
        A(1, 0) = 1.0;
        auto M = xt::xtensor<double, 4>::from_shape({3, 4, 2, 2});
        auto R = xt::xtensor<double, 2>::from_shape({M.shape(0), M.shape(1)});
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                xt::view(M, i, j, xt::all(), xt::all()) = static_cast<double>(i * M.shape(1) + j) * A;
                R(i, j) = static_cast<double>(i * M.shape(1) + j) * std::sqrt(2.0);
            }
        }
        REQUIRE(xt::allclose(GM::Norm_deviatoric(M), R));
    }
}

TEST_CASE("GMatTensor::Cartesian2d::pointer", "Cartesian2d.h")
{
    SECTION("I2")
    {
        auto i = GM::I2();
        auto r = GM::O2();
        GM::pointer::I2(r.data());
        REQUIRE(xt::allclose(i, r));
    }

    SECTION("II")
    {
        auto i = GM::II();
        auto r = GM::O4();
        GM::pointer::II(r.data());
        REQUIRE(xt::allclose(i, r));
    }

    SECTION("I4")
    {
        auto i = GM::I4();
        auto r = GM::O4();
        GM::pointer::I4(r.data());
        REQUIRE(xt::allclose(i, r));
    }

    SECTION("I4rt")
    {
        auto i = GM::I4rt();
        auto r = GM::O4();
        GM::pointer::I4rt(r.data());
        REQUIRE(xt::allclose(i, r));
    }

    SECTION("I4s")
    {
        auto i = GM::I4s();
        auto r = GM::O4();
        GM::pointer::I4s(r.data());
        REQUIRE(xt::allclose(i, r));
    }

    SECTION("I4d")
    {
        auto i = GM::I4d();
        auto r = GM::O4();
        GM::pointer::I4d(r.data());
        REQUIRE(xt::allclose(i, r));
    }

    SECTION("hydrostatic_deviatoric")
    {
        auto A = GM::Random2();
        auto B = A;
        auto C = A;
        double tr = B(0, 0) + B(1, 1);
        B(0, 0) -= 0.5 * tr;
        B(1, 1) -= 0.5 * tr;
        double m = GM::pointer::hydrostatic_deviatoric(A.data(), C.data());
        REQUIRE(m == Approx(0.5 * tr));
        REQUIRE(xt::allclose(C, B));
    }

    SECTION("deviatoric_ddot_deviatoric")
    {
        auto A = GM::O2();
        A(0, 1) = 1.0;
        A(1, 0) = 1.0;
        REQUIRE(GM::pointer::deviatoric_ddot_deviatoric(A.data()) == Approx(2.0));
    }
}
