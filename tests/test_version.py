import unittest

import GMatTensor as GMat


class Test_version(unittest.TestCase):
    """ """

    def test_version_dependencies(self):

        deps = GMat.version_dependencies()
        deps = [i.split("=")[0] for i in deps]
        self.assertIn("xtl", deps)
        self.assertIn("xtensor", deps)
        self.assertIn("gmattensor", deps)


if __name__ == "__main__":

    unittest.main()
