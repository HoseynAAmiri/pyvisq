from pyvisq import Test, TestMethod
from pyvisq.models import zener

# Define the test method and parameters
method = TestMethod.CREEP
test_params = {
    "I": 1.0,
    "D1": 0.01,
    "L1": 2,
    "D2": 0.01,
    "L2": 2
}
test = Test(method=method, **test_params)

# Define the Zener model parameters
dashpot_a = zener.DashpotParams(c=1)
spring_b = zener.SpringParams(k=1)
spring_c = zener.SpringParams(k=1)
sls_params = zener.SLSParams(
    dashpot_a=dashpot_a,
    spring_b=spring_b,
    spring_c=spring_c
)
sls = zener.SLS(params=sls_params)

# Print the SLS model diagram and parameters
print(sls)

# Set up and run the test
sls.set_test(test)
sls.set_time()
sls.set_input()  # Optional: set the input profile for visualization
sls.run()