import unittest

import GMatTensor.Cartesian2d as GMat
import numpy as np


class Test_tensor(unittest.TestCase):
    def test_I4(self):

        A = np.random.random([2, 2])
        I2 = GMat.I2()
        I2 = GMat.I4()

        self.assertTrue(np.allclose(GMat.A4_ddot_B2(I2, A), A))

    def test_I4s(self):

        A = np.random.random([2, 2])
        Is = GMat.I4s()

        self.assertTrue(np.allclose(GMat.A4_ddot_B2(Is, A), 0.5 * (A + A.T)))

    def test_I4d(self):

        A = np.random.random([2, 2])
        I2 = GMat.I2()
        Id = GMat.I4d()
        B = 0.5 * (A + A.T)

        self.assertTrue(np.allclose(GMat.A4_ddot_B2(Id, A), B - GMat.Hydrostatic(B) * I2))

    def test_hydrostatic(self):

        A = np.random.random([2, 2])
        B = np.array(A, copy=True)
        tr = B[0, 0] + B[1, 1]
        self.assertTrue(np.isclose(float(GMat.Hydrostatic(A)), 0.5 * tr))

    def test_deviatoric(self):

        A = np.random.random([2, 2])
        B = np.array(A, copy=True)
        tr = B[0, 0] + B[1, 1]
        B[0, 0] -= 0.5 * tr
        B[1, 1] -= 0.5 * tr
        self.assertTrue(np.allclose(GMat.Deviatoric(A), B))


if __name__ == "__main__":

    unittest.main()
