import numpy
import main
import matplotlib.pyplot as pyplot

dimensions = 10
mu = numpy.double(0.3)
omega = numpy.double(0.6)
delta_time = numpy.double(0.1)
time_steps = 45
step_skip = 3
lamb = numpy.cdouble(mu/2 + 1j * omega)

rho_0 = numpy.zeros((dimensions, dimensions))
rho_0[-2,-2] = 1
rho_approx_transformed = main.solve_num(rho_0, mu, time_steps, delta_time)
solution_approx = main.transform_back(rho_approx_transformed, lamb, delta_time)
rho_exact_transformed = main.solve_exact(rho_0, mu, time_steps, delta_time)
solution_exact = main.transform_back(rho_exact_transformed, lamb, delta_time)


def plot_bar(approx, exact, title: str):
    width = 0.4
    x_cords = numpy.arange(dimensions) + 1
    for time_step in range(0, time_steps + 1, step_skip):
        fig, ax = pyplot.subplots()
        diag_approx = numpy.diag(approx[time_step])
        diag_exact = numpy.diag(exact[time_step])
        ax.bar(x_cords - width, diag_approx, width, align='edge')
        ax.bar(x_cords, diag_exact, width, align='edge')
        ax.set_title('{:} for t = {:.1f}'.format(title, time_step * delta_time))
        ax.legend(("approximate", "exact"))
        pyplot.show()
        pyplot.close(fig)


plot_bar(rho_approx_transformed, rho_exact_transformed, "transformed solution rho tilde")
plot_bar(solution_approx, solution_exact, "solution rho")
