
#include <catch2/catch.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>
#include <GMatTensor/Cartesian2d.h>

namespace GM = GMatTensor::Cartesian2d;

TEST_CASE("GMatTensor::Cartesian2d", "Cartesian2d.h")
{
    SECTION("I4 - Tensor")
    {
        auto A = GM::Random2();
        auto I = GM::I4();
        REQUIRE(xt::allclose(GM::A4_ddot_B2(I, A), A));
    }

    SECTION("I4 - Array")
    {
        auto N = GM::Array<3>({4, 2, 3});
        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 2, 2});
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                for (size_t k = 0; k < M.shape(2); ++k) {
                    xt::view(M, i, j, k, xt::all(), xt::all()) = GM::Random2();
                }
            }
        }
        REQUIRE(xt::allclose(GM::A4_ddot_B2(N.I4(), M), M));
    }

    SECTION("I4s - Tensor")
    {
        auto A = GM::Random2();
        auto Is = GM::I4s();
        REQUIRE(xt::allclose(GM::A4_ddot_B2(Is, A), 0.5 * (A + xt::transpose(A))));
    }

    SECTION("I4s - Array")
    {
        auto N = GM::Array<3>({4, 2, 3});
        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 2, 2});
        auto R = xt::xtensor<double, 5>::from_shape(M.shape());
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                for (size_t k = 0; k < M.shape(2); ++k) {
                    auto A = GM::Random2();
                    auto B = GM::Sym(A);
                    xt::view(M, i, j, k, xt::all(), xt::all()) = A;
                    xt::view(R, i, j, k, xt::all(), xt::all()) = B;
                }
            }
        }
        REQUIRE(xt::allclose(GM::A4_ddot_B2(N.I4s(), M), R));
    }

    SECTION("I4d - Tensor")
    {
        auto A = GM::Random2();
        auto I = GM::I2();
        auto Id = GM::I4d();
        auto B = GM::Sym(A);
        REQUIRE(xt::allclose(GM::A4_ddot_B2(Id, A), B - GM::Hydrostatic(B) * I));
    }

    SECTION("I4d - Array")
    {
        auto N = GM::Array<3>({4, 2, 3});
        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 2, 2});
        auto R = xt::xtensor<double, 5>::from_shape(M.shape());
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                for (size_t k = 0; k < M.shape(2); ++k) {
                    auto A = GM::Random2();
                    auto B = xt::eval(0.5 * (A + xt::transpose(A)));
                    auto I = GM::I2();
                    xt::view(M, i, j, k, xt::all(), xt::all()) = A;
                    xt::view(R, i, j, k, xt::all(), xt::all()) = B - GM::Hydrostatic(B) * I;
                }
            }
        }
        REQUIRE(xt::allclose(GM::A4_ddot_B2(N.I4d(), M), R));
        REQUIRE(xt::allclose(GM::A4_ddot_B2(N.I4d(), M), GM::Deviatoric(GM::Sym(M))));
    }

    SECTION("Trace - Tensor")
    {
        auto A = GM::Random2();
        REQUIRE(GM::Trace(A)() == Approx(A(0, 0) + A(1, 1)));
    }

    SECTION("Trace - Array")
    {
        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 2, 2});
        auto R = xt::xtensor<double, 3>::from_shape({4, 2, 3});
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                for (size_t k = 0; k < M.shape(2); ++k) {
                    auto A = GM::Random2();
                    xt::view(M, i, j, k, xt::all(), xt::all()) = A;
                    xt::view(R, i, j, k) = A(0, 0) + A(1, 1);
                }
            }
        }
        REQUIRE(xt::allclose(GM::Trace(M), R));
    }

    SECTION("Hydrostatic - Tensor")
    {
        auto A = GM::Random2();
        REQUIRE(GM::Hydrostatic(A)() == Approx(0.5 * (A(0, 0) + A(1, 1))));
    }

    SECTION("Hydrostatic - Array")
    {
        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 2, 2});
        auto R = xt::xtensor<double, 3>::from_shape({4, 2, 3});
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                for (size_t k = 0; k < M.shape(2); ++k) {
                    auto A = GM::Random2();
                    xt::view(M, i, j, k, xt::all(), xt::all()) = A;
                    xt::view(R, i, j, k) = 0.5 * (A(0, 0) + A(1, 1));
                }
            }
        }
        REQUIRE(xt::allclose(GM::Hydrostatic(M), R));
    }

    SECTION("A2_ddot_B2 - Tensor")
    {
        auto A = GM::Deviatoric(GM::Sym(GM::Random2()));
        auto r = 2.0 * std::pow(A(0, 0), 2.0) + 2.0 * std::pow(A(0, 1), 2.0);
        REQUIRE(GM::A2_ddot_B2(A, A)() == Approx(r));
    }

    SECTION("A2_ddot_B2 - Array")
    {
        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 2, 2});
        auto R = xt::xtensor<double, 3>::from_shape({4, 2, 3});
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                for (size_t k = 0; k < M.shape(2); ++k) {
                    auto A = GM::Deviatoric(GM::Sym(GM::Random2()));
                    auto r = 2.0 * std::pow(A(0, 0), 2.0) + 2.0 * std::pow(A(0, 1), 2.0);
                    xt::view(M, i, j, k, xt::all(), xt::all()) = A;
                    xt::view(R, i, j, k) = r;
                }
            }
        }
        REQUIRE(xt::allclose(GM::A2_ddot_B2(M, M), R));
    }

    SECTION("A2s_ddot_B2s - Tensor")
    {
        auto A = GM::Deviatoric(GM::Sym(GM::Random2()));
        auto r = 2.0 * std::pow(A(0, 0), 2.0) + 2.0 * std::pow(A(0, 1), 2.0);
        REQUIRE(GM::A2s_ddot_B2s(A, A)() == Approx(r));
    }

    SECTION("A2s_ddot_B2s - Array")
    {
        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 2, 2});
        auto R = xt::xtensor<double, 3>::from_shape({4, 2, 3});
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                for (size_t k = 0; k < M.shape(2); ++k) {
                    auto A = GM::Deviatoric(GM::Sym(GM::Random2()));
                    auto r = 2.0 * std::pow(A(0, 0), 2.0) + 2.0 * std::pow(A(0, 1), 2.0);
                    xt::view(M, i, j, k, xt::all(), xt::all()) = A;
                    xt::view(R, i, j, k) = r;
                }
            }
        }
        REQUIRE(xt::allclose(GM::A2s_ddot_B2s(M, M), R));
    }

    SECTION("Norm_deviatoric - Tensor")
    {
        auto A = GM::Sym(GM::Random2());
        auto B = GM::Deviatoric(A);
        auto r = std::sqrt(2.0 * std::pow(B(0, 0), 2.0) + 2.0 * std::pow(B(0, 1), 2.0));
        REQUIRE(GM::Norm_deviatoric(B)() == Approx(r));
    }

    SECTION("Norm_deviatoric - Array")
    {
        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 2, 2});
        auto R = xt::xtensor<double, 3>::from_shape({4, 2, 3});
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                for (size_t k = 0; k < M.shape(2); ++k) {
                    auto A = GM::Sym(GM::Random2());
                    auto B = GM::Deviatoric(A);
                    auto r = std::sqrt(2.0 * std::pow(B(0, 0), 2.0) + 2.0 * std::pow(B(0, 1), 2.0));
                    xt::view(M, i, j, k, xt::all(), xt::all()) = A;
                    xt::view(R, i, j, k) = r;
                }
            }
        }
        REQUIRE(xt::allclose(GM::Norm_deviatoric(M), R));
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

    SECTION("Deviatoric - Array")
    {
        auto A = GM::Random2();
        auto B = A;
        double tr = B(0, 0) + B(1, 1);
        B(0, 0) -= 0.5 * tr;
        B(1, 1) -= 0.5 * tr;

        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 2, 2});
        auto R = xt::xtensor<double, 5>::from_shape(M.shape());
        xt::xtensor<double, 3> r = xt::random::rand<double>({4, 2, 3});
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                for (size_t k = 0; k < M.shape(2); ++k) {
                    xt::view(M, i, j, k, xt::all(), xt::all()) = r(i, j, k) * A;
                    xt::view(R, i, j, k, xt::all(), xt::all()) = r(i, j, k) * B;
                }
            }
        }
        REQUIRE(xt::allclose(GM::Deviatoric(M), R));
    }

    SECTION("Sym - Tensor2")
    {
        auto A = GM::Random2();
        auto B = 0.5 * (A + xt::transpose(A));
        REQUIRE(xt::allclose(GM::Sym(A), B));
    }

    SECTION("Sym - Array")
    {
        auto A = GM::Random2();
        auto B = 0.5 * (A + xt::transpose(A));
        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 2, 2});
        auto R = xt::xtensor<double, 5>::from_shape(M.shape());
        xt::xtensor<double, 3> r = xt::random::rand<double>({4, 2, 3});
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                for (size_t k = 0; k < M.shape(2); ++k) {
                    xt::view(M, i, j, k, xt::all(), xt::all()) = r(i, j, k) * A;
                    xt::view(R, i, j, k, xt::all(), xt::all()) = r(i, j, k) * B;
                }
            }
        }
        REQUIRE(xt::allclose(GM::Sym(M), R));
    }

    SECTION("A2_dot_B2 - Tensor")
    {
        auto A = GM::Random2();
        auto I = GM::I2();
        REQUIRE(xt::allclose(GM::A2_dot_B2(A, I), A));
    }

    SECTION("A2_dot_B2 - Array")
    {
        auto N = GM::Array<3>({4, 2, 3});
        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 2, 2});
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                for (size_t k = 0; k < M.shape(2); ++k) {
                    xt::view(M, i, j, k, xt::all(), xt::all()) = GM::Random2();
                }
            }
        }
        REQUIRE(xt::allclose(GM::A2_dot_B2(M, N.I2()), M));
    }

    SECTION("A2_dyadic_B2 - Tensor")
    {
        auto I2 = GM::I2();
        REQUIRE(xt::allclose(GM::A2_dyadic_B2(I2, I2), GM::II()));
    }

    SECTION("A2_dyadic_B2 - Array")
    {
        auto N = GM::Array<3>({4, 2, 3});
        REQUIRE(xt::allclose(GM::A2_dyadic_B2(N.I2(), N.I2()), N.II()));
    }

    SECTION("A4_ddot_B2 - Tensor2")
    {
        auto A = GM::Random2();
        auto Is = GM::I4s();
        auto B = xt::eval(0.5 * (A + xt::transpose(A)));
        REQUIRE(xt::allclose(GM::A4_ddot_B2(Is, A), B));
    }

    SECTION("A4_ddot_B2 - Array")
    {
        auto N = GM::Array<3>({4, 2, 3});
        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 2, 2});
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                for (size_t k = 0; k < M.shape(2); ++k) {
                    xt::view(M, i, j, k, xt::all(), xt::all()) = GM::Random2();
                }
            }
        }
        REQUIRE(xt::allclose(GM::A4_ddot_B2(N.I4s(), M), GM::Sym(M)));
        REQUIRE(xt::allclose(GM::A4_ddot_B2(N.I4d(), M), GM::Deviatoric(GM::Sym(M))));
    }
}
