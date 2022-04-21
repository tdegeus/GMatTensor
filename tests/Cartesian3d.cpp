#define CATCH_CONFIG_MAIN
#include <GMatTensor/Cartesian3d.h>
#include <catch2/catch.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>

namespace GM = GMatTensor::Cartesian3d;

TEST_CASE("GMatTensor::Cartesian3d", "Cartesian3d.h")
{
    SECTION("Array - shape")
    {
        std::array<size_t, 3> shape = {4, 2, 3};
        size_t size = 4 * 2 * 3;
        auto A2 = xt::xtensor<double, 5>::from_shape({4, 2, 3, 3, 3});
        auto A4 = xt::xtensor<double, 7>::from_shape({4, 2, 3, 3, 3, 3, 3});
        REQUIRE(xt::all(xt::equal(xt::adapt(shape), xt::adapt(GM::underlying_shape_A2(A2)))));
        REQUIRE(xt::all(xt::equal(xt::adapt(shape), xt::adapt(GM::underlying_shape_A4(A4)))));
        REQUIRE(size == GM::underlying_size_A2(A2));
        REQUIRE(size == GM::underlying_size_A4(A4));
    }

    SECTION("Array - view_tensor2")
    {
        auto N = GM::Array<3>({4, 2, 3});
        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 3, 3});
        auto A = xt::xtensor<double, 5>::from_shape({4, 2, 3, 3, 3});
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                for (size_t k = 0; k < M.shape(2); ++k) {
                    xt::view(M, i, j, k, xt::all(), xt::all()) = GM::Random2();
                }
            }
        }
        for (size_t i = 0; i < 4 * 2 * 3; ++i) {
            auto m = N.view_tensor2(M, i);
            auto a = N.view_tensor2(A, i);
            std::copy(m.cbegin(), m.cend(), a.begin());
        }
        REQUIRE(xt::allclose(M, A));
    }

    SECTION("Array - view_tensor4")
    {
        auto N = GM::Array<3>({4, 2, 3});
        auto M = xt::xtensor<double, 7>::from_shape({4, 2, 3, 3, 3, 3, 3});
        auto A = xt::xtensor<double, 7>::from_shape({4, 2, 3, 3, 3, 3, 3});
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                for (size_t k = 0; k < M.shape(2); ++k) {
                    xt::view(M, i, j, k, xt::all(), xt::all(), xt::all(), xt::all()) =
                        GM::Random4();
                }
            }
        }
        for (size_t i = 0; i < 4 * 2 * 3; ++i) {
            auto m = N.view_tensor4(M, i);
            auto a = N.view_tensor4(A, i);
            std::copy(m.cbegin(), m.cend(), a.begin());
        }
        REQUIRE(xt::allclose(M, A));
    }

    SECTION("I4 - Tensor")
    {
        auto A = GM::Random2();
        auto I = GM::I4();
        REQUIRE(xt::allclose(GM::A4_ddot_B2(I, A), A));
    }

    SECTION("I4 - Array")
    {
        auto N = GM::Array<3>({4, 2, 3});
        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 3, 3});
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
        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 3, 3});
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
        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 3, 3});
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
        REQUIRE(GM::Trace(A)() == Approx(A(0, 0) + A(1, 1) + A(2, 2)));
    }

    SECTION("Trace - Array")
    {
        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 3, 3});
        auto R = xt::xtensor<double, 3>::from_shape({4, 2, 3});
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                for (size_t k = 0; k < M.shape(2); ++k) {
                    auto A = GM::Random2();
                    xt::view(M, i, j, k, xt::all(), xt::all()) = A;
                    xt::view(R, i, j, k) = A(0, 0) + A(1, 1) + A(2, 2);
                }
            }
        }
        REQUIRE(xt::allclose(GM::Trace(M), R));
    }

    SECTION("Hydrostatic - Tensor")
    {
        auto A = GM::Random2();
        REQUIRE(GM::Hydrostatic(A)() == Approx((A(0, 0) + A(1, 1) + A(2, 2)) / 3.0));
    }

    SECTION("Hydrostatic - Array")
    {
        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 3, 3});
        auto R = xt::xtensor<double, 3>::from_shape({4, 2, 3});
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                for (size_t k = 0; k < M.shape(2); ++k) {
                    auto A = GM::Random2();
                    xt::view(M, i, j, k, xt::all(), xt::all()) = A;
                    xt::view(R, i, j, k) = (A(0, 0) + A(1, 1) + A(2, 2)) / 3.0;
                }
            }
        }
        REQUIRE(xt::allclose(GM::Hydrostatic(M), R));
    }

    SECTION("Det - Tensor")
    {
        auto A = GM::I2();
        REQUIRE(GM::Det(A)() == Approx(1.0));
    }

    SECTION("Det - Array")
    {
        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 3, 3});
        auto R = xt::xtensor<double, 3>::from_shape({4, 2, 3});
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                for (size_t k = 0; k < M.shape(2); ++k) {
                    auto A = GM::Random2();
                    xt::view(M, i, j, k, xt::all(), xt::all()) = GM::I2();
                    xt::view(R, i, j, k) = 1.0;
                }
            }
        }
        REQUIRE(xt::allclose(GM::Det(M), R));
    }

    SECTION("A2_ddot_B2 - Tensor")
    {
        auto A = GM::O2();
        A(0, 1) = 1.0;
        A(1, 0) = 1.0;
        auto r = 2.0;
        REQUIRE(GM::A2_ddot_B2(A, A)() == Approx(r));
    }

    SECTION("A2_ddot_B2 - Array")
    {
        auto A = GM::O2();
        A(0, 1) = 1.0;
        A(1, 0) = 1.0;
        auto r = 2.0;

        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 3, 3});
        auto R = xt::xtensor<double, 3>::from_shape({4, 2, 3});
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                for (size_t k = 0; k < M.shape(2); ++k) {
                    xt::view(M, i, j, k, xt::all(), xt::all()) = A;
                    xt::view(R, i, j, k) = r;
                }
            }
        }
        REQUIRE(xt::allclose(GM::A2_ddot_B2(M, M), R));
    }

    SECTION("A2s_ddot_B2s - Tensor")
    {
        auto A = GM::O2();
        A(0, 1) = 1.0;
        A(1, 0) = 1.0;
        auto r = 2.0;
        REQUIRE(GM::A2s_ddot_B2s(A, A)() == Approx(r));
    }

    SECTION("A2s_ddot_B2s - Array")
    {
        auto A = GM::O2();
        A(0, 1) = 1.0;
        A(1, 0) = 1.0;
        auto r = 2.0;

        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 3, 3});
        auto R = xt::xtensor<double, 3>::from_shape({4, 2, 3});
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                for (size_t k = 0; k < M.shape(2); ++k) {
                    xt::view(M, i, j, k, xt::all(), xt::all()) = A;
                    xt::view(R, i, j, k) = r;
                }
            }
        }
        REQUIRE(xt::allclose(GM::A2s_ddot_B2s(M, M), R));
    }

    SECTION("Norm_deviatoric - Tensor")
    {
        auto A = GM::O2();
        A(0, 1) = 1.0;
        A(1, 0) = 1.0;
        auto r = std::sqrt(2.0);
        REQUIRE(GM::Norm_deviatoric(A)() == Approx(r));
    }

    SECTION("Norm_deviatoric - Array")
    {
        auto A = GM::O2();
        A(0, 1) = 1.0;
        A(1, 0) = 1.0;
        auto r = std::sqrt(2.0);

        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 3, 3});
        auto R = xt::xtensor<double, 3>::from_shape({4, 2, 3});
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                for (size_t k = 0; k < M.shape(2); ++k) {
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
        double tr = B(0, 0) + B(1, 1) + B(2, 2);
        B(0, 0) -= tr / 3.0;
        B(1, 1) -= tr / 3.0;
        B(2, 2) -= tr / 3.0;
        REQUIRE(xt::allclose(GM::Deviatoric(A), B));
    }

    SECTION("Deviatoric - Array")
    {
        auto A = GM::Random2();
        auto B = A;
        double tr = B(0, 0) + B(1, 1) + B(2, 2);
        B(0, 0) -= tr / 3.0;
        B(1, 1) -= tr / 3.0;
        B(2, 2) -= tr / 3.0;

        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 3, 3});
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
        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 3, 3});
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

    SECTION("Inv - Tensor")
    {
        auto A = GM::I2();
        REQUIRE(xt::allclose(A, GM::Inv(A)));
    }

    SECTION("Inv - Tensor - 2")
    {
        auto A = GM::Random2();
        REQUIRE(xt::allclose(GM::A2_dot_B2(A, GM::Inv(A)), GM::I2()));
    }

    SECTION("Inv - Array")
    {
        auto N = GM::Array<3>({4, 2, 3});
        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 3, 3});
        xt::xtensor<double, 3> r = xt::random::rand<double>({4, 2, 3});
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                for (size_t k = 0; k < M.shape(2); ++k) {
                    xt::view(M, i, j, k, xt::all(), xt::all()) = GM::Random2();
                }
            }
        }
        REQUIRE(xt::allclose(GM::A2_dot_B2(M, GM::Inv(M)), N.I2()));
    }

    SECTION("Logs - Tensor2")
    {
        double gamma = 0.02;

        xt::xtensor<double, 2> F = {
            {1.0 + gamma, 0.0, 0.0}, {0.0, 1.0 / (1.0 + gamma), 0.0}, {0.0, 0.0, 1.0}};

        xt::xtensor<double, 2> Eps = {
            {std::log(1.0 + gamma), 0.0, 0.0}, {0.0, -std::log(1.0 + gamma), 0.0}, {0.0, 0.0, 0.0}};

        REQUIRE(xt::allclose(GM::Logs(F), Eps));
    }

    SECTION("Logs - Array")
    {
        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 3, 3});
        auto R = xt::xtensor<double, 5>::from_shape(M.shape());
        xt::xtensor<double, 3> r = xt::random::rand<double>({4, 2, 3});
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                for (size_t k = 0; k < M.shape(2); ++k) {

                    double gamma = r(i, j, k);

                    xt::xtensor<double, 2> F = {
                        {1.0 + gamma, 0.0, 0.0}, {0.0, 1.0 / (1.0 + gamma), 0.0}, {0.0, 0.0, 1.0}};

                    xt::xtensor<double, 2> Eps = {
                        {std::log(1.0 + gamma), 0.0, 0.0},
                        {0.0, -std::log(1.0 + gamma), 0.0},
                        {0.0, 0.0, 0.0}};

                    xt::view(M, i, j, k, xt::all(), xt::all()) = F;
                    xt::view(R, i, j, k, xt::all(), xt::all()) = Eps;
                }
            }
        }
        REQUIRE(xt::allclose(GM::Logs(M), R));
    }

    SECTION("A2_dot_A2T - Tensor2")
    {
        double a = 1.1;

        xt::xtensor<double, 2> A = {{1.0, a, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

        xt::xtensor<double, 2> B = {
            {1.0 + std::pow(a, 2.0), a, 0.0}, {a, 1.0, 0.0}, {0.0, 0.0, 1.0}};

        REQUIRE(xt::allclose(GM::A2_dot_A2T(A), B));
    }

    SECTION("A2_dot_A2T - Array")
    {
        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 3, 3});
        auto R = xt::xtensor<double, 5>::from_shape(M.shape());
        xt::xtensor<double, 3> r = xt::random::rand<double>({4, 2, 3});
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                for (size_t k = 0; k < M.shape(2); ++k) {

                    double a = r(i, j, k);

                    xt::xtensor<double, 2> A = {{1.0, a, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

                    xt::xtensor<double, 2> B = {
                        {1.0 + std::pow(a, 2.0), a, 0.0}, {a, 1.0, 0.0}, {0.0, 0.0, 1.0}};

                    xt::view(M, i, j, k, xt::all(), xt::all()) = A;
                    xt::view(R, i, j, k, xt::all(), xt::all()) = B;
                }
            }
        }
        REQUIRE(xt::allclose(GM::A2_dot_A2T(M), R));
    }

    SECTION("A2_dot_B2 - Tensor2")
    {
        auto A = GM::Random2();
        REQUIRE(xt::allclose(GM::A2_dot_B2(A, GM::I2()), A));
    }

    SECTION("A2_dot_B2 - Array")
    {
        auto N = GM::Array<3>({4, 2, 3});
        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 3, 3});
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
        auto M = xt::xtensor<double, 5>::from_shape({4, 2, 3, 3, 3});
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

TEST_CASE("GMatTensor::Cartesian3d::pointer", "Cartesian3d.h")
{
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
