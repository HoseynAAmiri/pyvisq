import unittest
from sls_fractional import FractionalSLS


class TestFractionalSLS(unittest.TestCase):
    def setUp(self):
        self.config_full = {
            'A': 1,
            'B': 0.25,
            'E1': 10.0,
            'E2': 5.0,
            'T1': 10,
            'T2': 0.1
        }
        self.config_direct = {
            'A': 1,
            'B': 0.25,
            'CA': 100,
            'CB': 5.0*0.1**0.25
        }

    def test_full_config_initialization(self):
        sls = FractionalSLS(self.config_full)
        self.assertAlmostEqual(sls.CA, 100, places=2)
        self.assertAlmostEqual(sls.CB, 5.0*0.1**0.25, places=2)

    def test_direct_config_initialization(self):
        sls = FractionalSLS(self.config_direct)
        self.assertEqual(sls.CA, 100)
        self.assertEqual(sls.CB, 5.0*0.1**0.25)

    def test_comparison_full_and_direct(self):
        sls_full = FractionalSLS(self.config_full)
        sls_direct = FractionalSLS(self.config_direct)
        T_full = sls_full.T
        T_direct = sls_direct.T
        E_full = sls_full.E
        E_direct = sls_direct.E

        self.assertAlmostEqual(T_full, T_direct, places=2)
        self.assertAlmostEqual(E_full, E_direct, places=2)


if __name__ == "__main__":
    unittest.main()
