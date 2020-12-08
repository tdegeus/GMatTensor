
#include <catch2/catch.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>
#include <GMatTensor/Cartesian3d.h>

namespace GM = GMatTensor::Cartesian3d;

TEST_CASE("GMatTensor::Cartesian3d", "Cartesian3d.h")
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
        A(2, 2) = 1.0;
        REQUIRE(GM::Trace(A)() == Approx(3.0));
    }

    SECTION("det")
    {
        auto A = GM::I2();
        REQUIRE(GM::Det(A)() == Approx(1.0));
    }

    SECTION("inv - 1")
    {
        auto A = GM::I2();
        REQUIRE(xt::allclose(A, GM::Inv(A)));
    }

    SECTION("inv - 2")
    {
        auto A = GM::Random2();
        REQUIRE(xt::allclose(GM::A2_dot_B2(A, GM::Inv(A)), GM::I2()));
    }

    SECTION("Logs - Tensor2")
    {
        double gamma = 0.02;

        xt::xtensor<double, 2> F = {
            {1.0 + gamma, 0.0, 0.0},
            {0.0, 1.0 / (1.0 + gamma), 0.0},
            {0.0, 0.0, 1.0}};

        xt::xtensor<double, 2> Eps = {
            {std::log(1.0 + gamma), 0.0, 0.0},
            {0.0, - std::log(1.0 + gamma), 0.0},
            {0.0, 0.0, 0.0}};

        REQUIRE(xt::allclose(GM::Logs(F), Eps));
    }

    SECTION("A2_ddot_B2")
    {
        auto A = GM::O2();
        A(0, 1) = 1.0;
        A(1, 0) = 1.0;
        REQUIRE(GM::A2_ddot_B2(A, A)() == Approx(2.0));
    }

    SECTION("A2s_ddot_B2s")
    {
        auto A = GM::O2();
        A(0, 1) = 1.0;
        A(1, 0) = 1.0;
        REQUIRE(GM::A2s_ddot_B2s(A, A)() == Approx(2.0));
    }

    SECTION("A2_dyadic_B2")
    {
        auto I2 = GM::I2();
        REQUIRE(xt::allclose(GM::A2_dyadic_B2(I2, I2), GM::II()));
    }

    SECTION("A2_dot_B2")
    {
        auto A = GM::Random2();
        REQUIRE(xt::allclose(GM::A2_dot_B2(A, GM::I2()), A));
    }

    SECTION("A2_dot_A2T")
    {
        double a = 1.1;

        xt::xtensor<double, 2> A = {
            {1.0, a, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}};

        xt::xtensor<double, 2> B = {
            {1.0 + std::pow(a, 2.0), a, 0.0},
            {a, 1.0, 0.0},
            {0.0, 0.0, 1.0}};

        REQUIRE(xt::allclose(GM::A2_dot_A2T(A), B));
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
        double tr = B(0, 0) + B(1, 1) + B(2, 2);
        B(0, 0) -= tr / 3.0;
        B(1, 1) -= tr / 3.0;
        B(2, 2) -= tr / 3.0;
        REQUIRE(xt::allclose(GM::Deviatoric(A), B));
    }

    SECTION("Deviatoric - List")
    {
        auto A = GM::Random2();
        auto B = A;
        double tr = B(0, 0) + B(1, 1) + B(2, 2);
        B(0, 0) -= tr / 3.0;
        B(1, 1) -= tr / 3.0;
        B(2, 2) -= tr / 3.0;
        auto M = xt::xtensor<double, 3>::from_shape({3, 3, 3});
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
        double tr = B(0, 0) + B(1, 1) + B(2, 2);
        B(0, 0) -= tr / 3.0;
        B(1, 1) -= tr / 3.0;
        B(2, 2) -= tr / 3.0;
        auto M = xt::xtensor<double, 4>::from_shape({3, 4, 3, 3});
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
        A(2, 2) = 1.0;
        REQUIRE(GM::Hydrostatic(A)() == Approx(1.0));
    }

    SECTION("Hydrostatic - List")
    {
        auto A = GM::Random2();
        A(0, 0) = 1.0;
        A(1, 1) = 1.0;
        A(2, 2) = 1.0;
        auto M = xt::xtensor<double, 3>::from_shape({3, 3, 3});
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
        A(2, 2) = 1.0;
        auto M = xt::xtensor<double, 4>::from_shape({3, 4, 3, 3});
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
        auto M = xt::xtensor<double, 3>::from_shape({3, 3, 3});
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
        auto M = xt::xtensor<double, 4>::from_shape({3, 4, 3, 3});
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

TEST_CASE("GMatTensor::Cartesian3d::pointer", "Cartesian3d.h")
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
        double tr = B(0, 0) + B(1, 1) + B(2, 2);
        B(0, 0) -= tr / 3.0;
        B(1, 1) -= tr / 3.0;
        B(2, 2) -= tr / 3.0;
        double m = GM::pointer::hydrostatic_deviatoric(A.data(), C.data());
        REQUIRE(m == Approx(tr / 3.0));
        REQUIRE(xt::allclose(C, B));
    }

    SECTION("A4_ddot_B4_ddot_C4")
    {
        auto A = GM::Random4();
        auto ret = GM::O4();
        auto I = GM::I4();
        GM::pointer::A4_ddot_B4_ddot_C4(I.data(), I.data(), A.data(), ret.data());
        REQUIRE(xt::allclose(A, ret));
    }

    SECTION("A2_dot_B2_dot_C2T")
    {
        auto A = GM::Random2();
        auto ret = GM::O2();
        auto I = GM::I2();
        GM::pointer::A2_dot_B2_dot_C2T(I.data(), A.data(), I.data(), ret.data());
        REQUIRE(xt::allclose(A, ret));
    }

    SECTION("eigs - from_eigs")
    {
        auto Is = GM::I4s();
        auto A = GM::Random2();
        auto C = GM::O2();
        A = GM::A4_ddot_B2(Is, A);
        xt::xtensor<double, 1> vals = xt::zeros<double>({3});
        xt::xtensor<double, 2> vecs = xt::zeros<double>({3, 3});
        GM::pointer::eigs(A.data(), vecs.data(), vals.data());
        GM::pointer::from_eigs(vecs.data(), vals.data(), C.data());
        REQUIRE(xt::allclose(A, C));
    }
}
