# pyvisq
An open-source package to model/visualize viscoelastic responses.

# Models
Run the code below to see available models:

```python
from pyvisq.models import hierarchy

hierarchy.show()
```

Output:
```
models\burgers
  ├── Burgers
models\elements
  ├── Spring
  ├── Dashpot
  ├── Springpot
models\generalized
  ├── Generalized
models\kelvinvoigt
  ├── KelvinVoigt
  ├── FracDashpotKelvinVoigt
  ├── FracSpringKelvinVoigt
  ├── FracKelvinVoigt
models\maxwell
  ├── Maxwell
  ├── FracDashpotMaxwell
  ├── FracSpringMaxwell
  ├── FracMaxwell
models\powerlaw
  ├── Powerlaw
models\poynting_thomson
  ├── SLSPT
  ├── JeffreysPT
  ├── FracSLSPT
  ├── FracJeffreysPT
  ├── FracPT
models\zener
  ├── SLS
  ├── JeffreysZener
  ├── FracJeffreysZener
  ├── FracSolidZener
  ├── FracSLSZener
  ├── FracZener
```

# Example:
```python
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
"""
input profile:
    I   ==========
      //:        :\\
     // :        : \\
    //  :D1      :L1\\D2==========L2
"""

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
""" Output:
                    ___
                 ____| |______╱╲  ╱╲  ╱╲  _____
                |   _|_|  ca    ╲╱  ╲╱  ╲╱  kb |
            ____|                              |____
                |                              |
                |__________╱╲  ╱╲  ╱╲  ________|
                             ╲╱  ╲╱  ╲╱  kc

SLSParams(dashpot_a=DashpotParams(c=1), spring_b=SpringParams(k=1), spring_c=SpringParams(k=1))
"""

# Set up and run the test
sls.set_test(test)
sls.set_time()
sls.set_input()  # Optional: set the input profile for visualization
sls.run()
```

# References
+ J. L. Kaplan, A. Bonfanti, A. J. Kabla (2019). RHEOS.jl -- A Julia Package for Rheology Data Analysis. Journal of Open Source Software, 4(41), 1700, https://doi.org/10.21105/joss.01700

+ A. Bonfanti, J. L. Kaplan, G. Charras, A. J. Kabla (2020). Fractional viscoelastic models for power-law materials. Soft Matter, 16, 6002-6020, https://doi.org/10.1039/D0SM00354A