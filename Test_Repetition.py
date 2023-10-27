import main
import numpy


repetitions = 100
mu = numpy.double(0.3)
omega = numpy.double(0.6)
delta_time = numpy.double(0.1)
time_steps = 10
lamb = numpy.cdouble(mu/2 + 1j * omega)
errors_tilde = numpy.zeros(repetitions)
errors = numpy.zeros(repetitions)
for k in range(0, repetitions):
    rho_0 = main.generate_starting_condition(10)
    rho_approx_transformed = main.solve_num(rho_0, mu, time_steps, delta_time)
    solution_approx = main.transform_back(rho_approx_transformed, lamb, delta_time)
    rho_exact_transformed = main.solve_exact(rho_0, mu, time_steps, delta_time)
    solution_exact = main.transform_back(rho_exact_transformed, lamb, delta_time)
    errors_tilde[k] = main.calculate_difference(rho_approx_transformed[-1], rho_exact_transformed[-1])
    errors[k] = main.calculate_difference(solution_approx[-1], solution_exact[-1])
    print("Test Nummer " + str(k+1) + ": Fehler rho tilde = " + str(errors_tilde[k]) + ", Fehler rho = "
          + str(errors[k]))
print("---------------------------------------------------------------")
print("average error rho tilde = " + str(sum(errors_tilde) / repetitions))
print("average error rho = " + str(sum(errors) / repetitions))