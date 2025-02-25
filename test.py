from pyvisq import Test, TestMethod
from pyvisq.models import zener

from matplotlib import pyplot as plt

# Define the test method and parameters
method = TestMethod.RELAXATION
test_params = {
    "I": 1.0,
    "D1": 0.1,
    "L1": 2,
    "D2": 0,
    "L2": 0
}
test = Test(method=method, **test_params)

# Define the Zener model parameters
dashpot_a = zener.DashpotParams(c=1)
springpot_b = zener.SpringpotParams(e=0.2, ce=1)
spring_c = zener.SpringParams(k=1)
params = zener.FracSolidZenerParams(
    dashpot_a=dashpot_a,
    springpot_b=springpot_b,
    spring_c=spring_c
)
fsls = zener.FracSolidZener(params=params)

print(fsls)
fsls.set_test(test)
fsls.set_time(D1_size=20, L1_size=100)
fsls.set_input()  # Optional: set the input profile for visualization
fsls.run()

plt.plot(fsls.data.time, fsls.data.stress)
plt.show()
