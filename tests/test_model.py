import unittest

import numpy as np

from pyvisq import Test, TestMethod
from pyvisq.models import Model


class TestModel(Model):
    def G(self, t: float) -> float:
        return 1.0

    def J(self, t: float) -> float:
        return 1.0


I = 1.0
D1 = 0.1
L1 = 0.2
D2 = 0.3
L2 = 0.4
creep = Test(method=TestMethod.CREEP, I=I, D1=D1, L1=L1, D2=D2, L2=L2)
relaxation = Test(method=TestMethod.RELAXATION,
                  I=I, D1=D1, L1=L1, D2=D2, L2=L2)
model = TestModel()


class TestModelMethodsCreep(unittest.TestCase):

    def setUp(self):
        self.model = model
        self.model.set_test(creep)
        self.model.set_time()

    def test_set_time(self):
        self.model.set_time()
        time = self.model.data.time
        self.assertEqual(len(time), 81)
        self.assertAlmostEqual(time[-1], 1.0)

    def test_set_test(self):
        self.model.set_test(creep)
        self.assertEqual(self.model.test.I, 1.0)
        self.assertEqual(self.model.test.D1,  0.1)
        self.assertEqual(self.model.test.L1,  0.2)
        self.assertEqual(self.model.test.D2,  0.3)
        self.assertEqual(self.model.test.L2,  0.4)

    def test_input1(self):
        self.model.set_test(creep)
        self.model.set_time()
        input = self.model._input()
        self.assertEqual(len(input), 81)
        self.assertAlmostEqual(input[-1], 0.0)

    def test_input2(self):
        self.model.set_test(creep)
        self.model.set_time()
        input = self.model._input()
        self.assertEqual(len(input), 81)
        self.assertAlmostEqual(input[21], 1.0)


if __name__ == '__main__':
    unittest.main()
