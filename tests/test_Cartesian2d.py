import unittest

import GMatTensor.Cartesian2d as GMat
import numpy as np

nd = 2
shape = [4, 3, 2]


class Test_tensor(unittest.TestCase):
    """ """

    def test_O2(self):

        self.assertTrue(np.allclose(GMat.O2(), np.zeros((nd, nd))))

    def test_O4(self):

        self.assertTrue(np.allclose(GMat.O4(), np.zeros((nd, nd, nd, nd))))

    def test_I2(self):

        self.assertTrue(np.allclose(GMat.I2(), np.eye(nd)))

    def test_II(self):

        unit = GMat.I2()
        self.assertTrue(np.allclose(GMat.II(), np.einsum("ij,kl", unit, unit)))

    def test_I4(self):

        A = np.random.random([nd, nd])
        unit = GMat.I4()

        self.assertTrue(np.allclose(A, np.einsum("...ijkl, ...lk", unit, A)))

    def test_I4rt(self):

        A = np.random.random([nd, nd])
        unit = GMat.I4rt()

        res = np.einsum("...ij -> ...ji", A)

        self.assertTrue(np.allclose(res, np.einsum("...ijkl, ...lk", unit, A)))

    def test_I4s(self):

        A = np.random.random([nd, nd])
        unit = GMat.I4s()

        A_T = np.einsum("...ij -> ...ji", A)
        res = 0.5 * (A + A_T)

        self.assertTrue(np.allclose(res, np.einsum("...ijkl, ...lk", unit, A)))

    def test_I4d(self):

        A = np.random.random(shape + [nd, nd])
        unit = GMat.Array3d(shape).I4d

        tr = 0.5 * np.trace(A, axis1=-2, axis2=-1)
        A_T = np.einsum("...ij -> ...ji", A)
        res = 0.5 * (A + A_T)
        res[..., 0, 0] -= tr
        res[..., 1, 1] -= tr

        self.assertTrue(np.allclose(res, np.einsum("...ijkl, ...lk", unit, A)))

    def test_trace(self):

        A = np.random.random(shape + [nd, nd])

        res = np.trace(A, axis1=-2, axis2=-1)

        ret = np.empty_like(res)
        GMat.trace(A, ret)

        self.assertTrue(np.allclose(res, GMat.Trace(A)))
        self.assertTrue(np.allclose(res, ret))

    def test_hydrostatic(self):

        A = np.random.random(shape + [nd, nd])

        res = 0.5 * np.trace(A, axis1=-2, axis2=-1)

        ret = np.empty_like(res)
        GMat.hydrostatic(A, ret)

        self.assertTrue(np.allclose(res, GMat.Hydrostatic(A)))
        self.assertTrue(np.allclose(res, ret))

    def test_A2_ddot_B2(self):

        A = np.random.random(shape + [nd, nd])
        B = np.random.random(shape + [nd, nd])

        res = np.einsum("...ij, ...ji", A, B)

        ret = np.empty_like(res)
        GMat.A2_ddot_B2(A, B, ret)

        self.assertTrue(np.allclose(res, GMat.A2_ddot_B2(A, B)))
        self.assertTrue(np.allclose(res, ret))

    def test_A2s_ddot_B2s(self):

        A = np.random.random(shape + [nd, nd])
        B = np.random.random(shape + [nd, nd])
        A_T = np.einsum("...ij -> ...ji", A)
        B_T = np.einsum("...ij -> ...ji", B)
        A = 0.5 * (A + A_T)
        B = 0.5 * (B + B_T)

        res = np.einsum("...ij, ...ji", A, B)

        ret = np.empty_like(res)
        GMat.A2s_ddot_B2s(A, B, ret)

        self.assertTrue(np.allclose(res, GMat.A2s_ddot_B2s(A, B)))
        self.assertTrue(np.allclose(res, ret))

    def test_norm_deviatoric(self):

        A = np.random.random(shape + [nd, nd])
        tr = 0.5 * np.trace(A, axis1=-2, axis2=-1)
        B = np.copy(A)
        B[..., 0, 0] -= tr
        B[..., 1, 1] -= tr
        res = np.sqrt(np.einsum("...ij, ...ji", B, B))

        ret = np.empty_like(res)
        GMat.norm_deviatoric(A, ret)

        self.assertTrue(np.allclose(res, GMat.Norm_deviatoric(A)))
        self.assertTrue(np.allclose(res, ret))

    def test_deviatoric(self):

        A = np.random.random(shape + [nd, nd])
        tr = 0.5 * np.trace(A, axis1=-2, axis2=-1)
        res = np.copy(A)
        res[..., 0, 0] -= tr
        res[..., 1, 1] -= tr

        ret = np.empty_like(res)
        GMat.deviatoric(A, ret)

        self.assertTrue(np.allclose(res, GMat.Deviatoric(A)))
        self.assertTrue(np.allclose(res, ret))

    def test_sym(self):

        A = np.random.random(shape + [nd, nd])
        A_T = np.einsum("...ij -> ...ji", A)
        res = 0.5 * (A + A_T)

        ret = np.empty_like(res)
        GMat.sym(A, ret)

        self.assertTrue(np.allclose(res, GMat.Sym(A)))
        self.assertTrue(np.allclose(res, ret))

    def test_A2_dot_B2(self):

        A = np.random.random(shape + [nd, nd])
        B = np.random.random(shape + [nd, nd])

        res = np.einsum("...ij, ...jk", A, B)

        ret = np.empty_like(res)
        GMat.A2_dot_B2(A, B, ret)

        self.assertTrue(np.allclose(res, GMat.A2_dot_B2(A, B)))
        self.assertTrue(np.allclose(res, ret))

    def test_A2_dyadic_B2(self):

        A = np.random.random(shape + [nd, nd])
        B = np.random.random(shape + [nd, nd])

        res = np.einsum("...ij, ...kl", A, B)

        ret = np.empty_like(res)
        GMat.A2_dyadic_B2(A, B, ret)

        self.assertTrue(np.allclose(res, GMat.A2_dyadic_B2(A, B)))
        self.assertTrue(np.allclose(res, ret))

    def test_A4_ddot_B2(self):

        A = np.random.random(shape + [nd, nd, nd, nd])
        B = np.random.random(shape + [nd, nd])

        res = np.einsum("...ijkl, ...lk", A, B)

        ret = np.empty_like(res)
        GMat.A4_ddot_B2(A, B, ret)

        self.assertTrue(np.allclose(res, GMat.A4_ddot_B2(A, B)))
        self.assertTrue(np.allclose(res, ret))

    def test_underlying_shape_A2(self):

        A = np.empty(shape + [nd, nd])
        self.assertEqual(shape, GMat.underlying_shape_A2(A))
        self.assertEqual(np.prod(shape), GMat.underlying_size_A2(A))

    def test_underlying_shape_A4(self):

        A = np.empty(shape + [nd, nd, nd, nd])
        self.assertEqual(shape, GMat.underlying_shape_A4(A))
        self.assertEqual(np.prod(shape), GMat.underlying_size_A4(A))


class Test_Array(unittest.TestCase):
    """ """

    def test_shape(self):

        array = GMat.Array3d(shape)
        self.assertEqual(shape, array.shape)
        self.assertEqual(shape + [nd, nd], array.shape_tensor2)
        self.assertEqual(shape + [nd, nd, nd, nd], array.shape_tensor4)

    def test_O2(self):

        res = np.zeros(shape + [nd, nd])
        self.assertTrue(np.allclose(res, GMat.Array3d(shape).O2))

    def test_O4(self):

        res = np.zeros(shape + [nd, nd, nd, nd])
        self.assertTrue(np.allclose(res, GMat.Array3d(shape).O4))

    def test_I2(self):

        res = np.zeros(shape + [nd, nd])
        res[..., 0, 0] = 1
        res[..., 1, 1] = 1

        self.assertTrue(np.allclose(res, GMat.Array3d(shape).I2))

    def test_II(self):

        unit = np.zeros(shape + [nd, nd])
        unit[..., 0, 0] = 1
        unit[..., 1, 1] = 1

        res = np.einsum("...ij, ...kl", unit, unit)

        self.assertTrue(np.allclose(res, GMat.Array3d(shape).II))

    def test_I4(self):

        A = np.random.random(shape + [nd, nd])
        unit = GMat.Array3d(shape).I4

        self.assertTrue(np.allclose(A, np.einsum("...ijkl, ...lk", unit, A)))

    def test_I4rt(self):

        A = np.random.random(shape + [nd, nd])
        unit = GMat.Array3d(shape).I4rt
        res = np.einsum("...ij -> ...ji", A)

        self.assertTrue(np.allclose(res, np.einsum("...ijkl, ...lk", unit, A)))

    def test_I4s(self):

        A = np.random.random(shape + [nd, nd])
        unit = GMat.Array3d(shape).I4s
        A_T = np.einsum("...ij -> ...ji", A)
        res = 0.5 * (A + A_T)

        self.assertTrue(np.allclose(res, np.einsum("...ijkl, ...lk", unit, A)))

    def test_I4d(self):

        A = np.random.random(shape + [nd, nd])
        unit = GMat.Array3d(shape).I4d

        tr = 0.5 * np.trace(A, axis1=-2, axis2=-1)
        A_T = np.einsum("...ij -> ...ji", A)
        res = 0.5 * (A + A_T)
        res[..., 0, 0] -= tr
        res[..., 1, 1] -= tr

        self.assertTrue(np.allclose(res, np.einsum("...ijkl, ...lk", unit, A)))


if __name__ == "__main__":

    unittest.main()
