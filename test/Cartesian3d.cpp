
#include <catch2/catch.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>
#include <GMatTensor/Cartesian3d.h>

#define ISCLOSE(a,b) REQUIRE_THAT((a), Catch::WithinAbs((b), 1e-12));

namespace GM = GMatTensor::Cartesian3d;

namespace Prod3d {

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

} // namespace Prod3d

TEST_CASE("GMatTensor::Cartesian3d", "Cartesian3d.h")
{
    SECTION("I4")
    {
        xt::xtensor<double, 2> A = xt::random::randn<double>({3, 3});
        xt::xtensor<double, 4> I = GM::I4();
        REQUIRE(xt::allclose(Prod3d::A4_ddot_B2(I, A), A));
    }

    SECTION("I4s")
    {
        xt::xtensor<double, 2> A = xt::random::randn<double>({3, 3});
        xt::xtensor<double, 4> Is = GM::I4s();
        REQUIRE(xt::allclose(Prod3d::A4_ddot_B2(Is, A), 0.5 * (A + xt::transpose(A))));
    }

    SECTION("I4d")
    {
        xt::xtensor<double, 2> A = xt::random::randn<double>({3, 3});
        xt::xtensor<double, 2> I = GM::I2();
        xt::xtensor<double, 4> Id = GM::I4d();
        xt::xtensor<double, 2> B = 0.5 * (A + xt::transpose(A));
        REQUIRE(xt::allclose(Prod3d::A4_ddot_B2(Id, A), B - GM::Hydrostatic(B) * I));
    }

    SECTION("Deviatoric - Tensor2")
    {
        xt::xtensor<double, 2> A = xt::random::randn<double>({3, 3});
        xt::xtensor<double, 2> B = A;
        double tr = B(0, 0) + B(1, 1) + B(2, 2);
        B(0, 0) -= tr / 3.0;
        B(1, 1) -= tr / 3.0;
        B(2, 2) -= tr / 3.0;
        REQUIRE(xt::allclose(GM::Deviatoric(A), B));
    }

    SECTION("Deviatoric - List")
    {
        xt::xtensor<double, 2> A = xt::random::randn<double>({3, 3});
        xt::xtensor<double, 2> B = A;
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
        xt::xtensor<double, 2> A = xt::random::randn<double>({3, 3});
        xt::xtensor<double, 2> B = A;
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
        xt::xtensor<double, 2> A = xt::random::randn<double>({3, 3});
        A(0, 0) = 1.0;
        A(1, 1) = 1.0;
        A(2, 2) = 1.0;
        REQUIRE(GM::Hydrostatic(A)() == Approx(1.0));
    }

    SECTION("Hydrostatic - List")
    {
        xt::xtensor<double, 2> A = xt::random::randn<double>({3, 3});
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
        xt::xtensor<double, 2> A = xt::random::randn<double>({3, 3});
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
        xt::xtensor<double, 2> A = xt::zeros<double>({3, 3});
        A(0, 1) = 1.0;
        A(1, 0) = 1.0;
        REQUIRE(GM::Equivalent_deviatoric(A)() == Approx(std::sqrt(2.0)));
    }

    SECTION("Equivalent_deviatoric - List")
    {
        xt::xtensor<double, 2> A = xt::zeros<double>({3, 3});
        A(0, 1) = 1.0;
        A(1, 0) = 1.0;
        auto M = xt::xtensor<double,3>::from_shape({3, 3, 3});
        auto R = xt::xtensor<double,1>::from_shape({M.shape(0)});
        for (size_t i = 0; i < M.shape(0); ++i) {
            xt::view(M, i, xt::all(), xt::all()) = static_cast<double>(i) * A;
            R(i) = static_cast<double>(i) * std::sqrt(2.0);
        }
        REQUIRE(xt::allclose(GM::Equivalent_deviatoric(M), R));
    }

    SECTION("Equivalent_deviatoric - Matrix")
    {
        xt::xtensor<double, 2> A = xt::zeros<double>({3, 3});
        A(0, 1) = 1.0;
        A(1, 0) = 1.0;
        auto M = xt::xtensor<double,4>::from_shape({3, 4, 3, 3});
        auto R = xt::xtensor<double,2>::from_shape({M.shape(0), M.shape(1)});
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                xt::view(M, i, j, xt::all(), xt::all()) = static_cast<double>(i * M.shape(1) + j) * A;
                R(i, j) = static_cast<double>(i * M.shape(1) + j) * std::sqrt(2.0);
            }
        }
        REQUIRE(xt::allclose(GM::Equivalent_deviatoric(M), R));
    }
}

TEST_CASE("GMatTensor::Cartesian3d::pointer", "Cartesian3d.h")
{
    SECTION("I2")
    {
        xt::xtensor<double, 2> i = GM::I2();
        xt::xtensor<double, 2> r = xt::empty<double>(i.shape());
        GM::pointer::I2(r.data());
        REQUIRE(xt::allclose(i, r));
    }

    SECTION("II")
    {
        xt::xtensor<double, 4> i = GM::II();
        xt::xtensor<double, 4> r = xt::empty<double>(i.shape());
        GM::pointer::II(r.data());
        REQUIRE(xt::allclose(i, r));
    }

    SECTION("I4")
    {
        xt::xtensor<double, 4> i = GM::I4();
        xt::xtensor<double, 4> r = xt::empty<double>(i.shape());
        GM::pointer::I4(r.data());
        REQUIRE(xt::allclose(i, r));
    }

    SECTION("I4rt")
    {
        xt::xtensor<double, 4> i = GM::I4rt();
        xt::xtensor<double, 4> r = xt::empty<double>(i.shape());
        GM::pointer::I4rt(r.data());
        REQUIRE(xt::allclose(i, r));
    }

    SECTION("I4s")
    {
        xt::xtensor<double, 4> i = GM::I4s();
        xt::xtensor<double, 4> r = xt::empty<double>(i.shape());
        GM::pointer::I4s(r.data());
        REQUIRE(xt::allclose(i, r));
    }

    SECTION("I4d")
    {
        xt::xtensor<double, 4> i = GM::I4d();
        xt::xtensor<double, 4> r = xt::empty<double>(i.shape());
        GM::pointer::I4d(r.data());
        REQUIRE(xt::allclose(i, r));
    }

    SECTION("trace")
    {
        xt::xtensor<double, 2> A = xt::random::randn<double>({3, 3});
        A(0, 0) = 1.0;
        A(1, 1) = 1.0;
        A(2, 2) = 1.0;
        REQUIRE(GM::pointer::trace(A.data()) == Approx(3.0));
    }

    SECTION("hydrostatic_deviatoric")
    {
        xt::xtensor<double, 2> A = xt::random::randn<double>({3, 3});
        xt::xtensor<double, 2> B = A;
        xt::xtensor<double, 2> C = A;
        double tr = B(0, 0) + B(1, 1) + B(2, 2);
        B(0, 0) -= tr / 3.0;
        B(1, 1) -= tr / 3.0;
        B(2, 2) -= tr / 3.0;
        double m = GM::pointer::hydrostatic_deviatoric(A.data(), C.data());
        REQUIRE(m == Approx(tr / 3.0));
        REQUIRE(xt::allclose(C, B));
    }

    SECTION("deviatoric_ddot_deviatoric")
    {
        xt::xtensor<double, 2> A = xt::zeros<double>({3, 3});
        A(0, 1) = 1.0;
        A(1, 0) = 1.0;
        REQUIRE(GM::pointer::deviatoric_ddot_deviatoric(A.data()) == Approx(2.0));
    }

    SECTION("A2_ddot_B2")
    {
        xt::xtensor<double, 2> A = xt::zeros<double>({3, 3});
        A(0, 1) = 1.0;
        A(1, 0) = 1.0;
        REQUIRE(GM::pointer::A2_ddot_B2(A.data(), A.data()) == Approx(2.0));
    }

    SECTION("A2s_ddot_B2s")
    {
        xt::xtensor<double, 2> A = xt::zeros<double>({3, 3});
        A(0, 1) = 1.0;
        A(1, 0) = 1.0;
        REQUIRE(GM::pointer::A2s_ddot_B2s(A.data(), A.data()) == Approx(2.0));
    }

    SECTION("A2_dyadic_B2")
    {
        xt::xtensor<double, 2> I2 = GM::I2();
        xt::xtensor<double, 4> II = GM::II();
        xt::xtensor<double, 4> C = xt::empty<double>({3, 3, 3, 3});
        GM::pointer::A2_dyadic_B2(I2.data(), I2.data(), C.data());
        REQUIRE(xt::allclose(II, C));
    }
}
