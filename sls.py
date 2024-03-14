from sympy import symbols, laplace_transform, inverse_laplace_transform, Heaviside, Function, solve, lambdify
import numpy as np
import matplotlib.pyplot as plt

s = symbols('s')
t = symbols('t', positive=True)
u = Heaviside

I, D = symbols('I D', real=True)
epsilon_t = (I * t / D) * (1 - u(t - D)) + I * u(t - D)

Epsilon = laplace_transform(epsilon_t, t, s, noconds=True)

sigma = symbols('sigma')
E1, E2, T = symbols('E1 E2 T', real=True)
Sigma = Function('Sigma')(s)

LHS = Sigma + T * (s * Sigma)
RHS = E1 * Epsilon + T * (E1 + E2) * s * Epsilon
transformed_equation = LHS - RHS

Sigma_s = solve(transformed_equation, Sigma)[0]

sigma_t = inverse_laplace_transform(Sigma_s, s, t)
print(sigma_t)
params = {I: 0.5,
          D: 0.1,
          E1: 1.0,
          E2: 2.0,
          T: 0.1}


time = np.linspace(0, 2, 100)

sigma_t = sigma_t.subs(params)
print(sigma_t)
sigma_numeric = lambdify(t, sigma_t, 'numpy')
sigma_values = sigma_numeric(time)

epsilon_t = epsilon_t.subs(params)
epsilon_numeric = lambdify(t, epsilon_t, 'numpy')
epsilon_values = epsilon_numeric(time)

plt.plot(time, sigma_values, 'b')
plt.plot(time, epsilon_values, 'r')
plt.show()
