import unittest
import numpy as np
import GMatTensor.Cartesian3d as GMat

def A4_ddot_B2(A, B):
    return np.einsum('ijkl,lk->ij', A, B)

class Test_tensor(unittest.TestCase):

    def test_I4(self):

        A = np.random.random([3, 3])
        I = GMat.I2()
        I = GMat.I4()
        self.assertTrue(np.allclose(A4_ddot_B2(I, A), A))

    def test_I4s(self):

        A = np.random.random([3, 3])
        I = GMat.I2()
        Is = GMat.I4s()
        self.assertTrue(np.allclose(A4_ddot_B2(Is, A), 0.5 * (A + A.T)))

    def test_I4d(self):

        A = np.random.random([3, 3])
        I = GMat.I2()
        Id = GMat.I4d()
        B = 0.5 * (A + A.T)
        self.assertTrue(np.allclose(A4_ddot_B2(Id, A), B - GMat.Hydrostatic(B) * I))

    def test_hydrostatic(self):

        A = np.random.random([3, 3])
        B = np.array(A, copy=True)
        tr = B[0, 0] + B[1, 1] + B[2, 2]
        self.assertTrue(np.isclose(float(GMat.Hydrostatic(A)), tr / 3.0))

    def test_deviatoric(self):

        A = np.random.random([3, 3])
        B = np.array(A, copy=True)
        tr = B[0, 0] + B[1, 1] + B[2, 2]
        B[0, 0] -= tr / 3.0
        B[1, 1] -= tr / 3.0
        B[2, 2] -= tr / 3.0
        self.assertTrue(np.allclose(GMat.Deviatoric(A), B))

if __name__ == '__main__':

    unittest.main()
