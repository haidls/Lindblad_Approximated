import math
import random
import numpy


def solve_num(rho_0: numpy.array, mu: numpy.double, time_steps: int, delta_time: numpy.double):
    dim = len(rho_0[0])
    rho = numpy.zeros((time_steps+1, dim, dim))
    rho[0] = numpy.array(rho_0, copy=True)
    for time_step in range(0,time_steps):
        rho[time_step + 1] = numpy.array(rho[time_step], copy=True)
        for i in range(0,dim-1):
            for j in range(0,dim-1):
                rho[time_step+1,i,j] = rho[time_step+1,i,j] + delta_time * mu * numpy.exp(-mu*time_step*delta_time) * \
                                       numpy.sqrt((i+1)*(j+1)) * rho[time_step,i+1,j+1]
    return rho


def solve_exact(rho_0: numpy.array, mu: numpy.double, time_steps: int, delta_time: numpy.double):
    dim = len(rho_0[0])
    rho = numpy.zeros((time_steps+1, dim, dim))
    E = numpy.zeros((time_steps, dim))
    for time_step in range(0, time_steps):
        rho[time_step] = numpy.array(rho_0, copy=True)
        E[time_step, 0] = 1 - math.exp(-mu*(time_step+1)*delta_time)
        for k in range(1,dim):
            E[time_step,k] = E[time_step,k-1] * E[time_step,0]
    rho[-1] = numpy.array(rho_0, copy=True)
    for i in range(0,dim):
        for j in range(0,dim):
            product = 1
            for k in range(1,dim-max(i,j)):
                product = product * numpy.sqrt((i+k) * (j+k)) / k
                for time_step in range(0,time_steps):
                    rho[time_step+1,i,j] = rho[time_step+1,i,j] + product * E[time_step,k-1] * rho_0[i+k,j+k]
    return rho


def transform_back(rho_transformed: numpy.array, lamb: numpy.cdouble, delta_time: numpy.double):
    dim = len(rho_transformed[0,0])
    time_steps = len(rho_transformed[:,0,0])
    rho = numpy.zeros(numpy.shape(rho_transformed), dtype=numpy.cdouble)
    for time_step in range(0,time_steps):
        for i in range(0,dim):
            for j in range(0,dim):
                rho[time_step, i, j] = numpy.exp(-time_step*delta_time*(lamb*i+numpy.conjugate(lamb)*j)) * \
                                       rho_transformed[time_step,i,j]
    return rho


def generate_starting_condition(dimensions: int):
    rho_0 = numpy.zeros((dimensions,dimensions))
    det_sgn = 0
    while(det_sgn <= 0):
        for i in range(0,dimensions):
            for j in range(0,i+1):
                rho_0[i,j] = random.uniform(-1,1)
                rho_0[j,i] = rho_0[i,j]
        det = numpy.linalg.det(rho_0)
        det_sgn = numpy.sign(det)
    det_root = math.pow(det, 1/dimensions)
    rho_0 = rho_0 / det_root
    return rho_0


def calculate_difference(matrix_1: numpy.array, matrix_2: numpy.array):
    difference = matrix_1 - matrix_2
    eigen_vals = numpy.linalg.eigvals(difference)
    return numpy.sum(numpy.abs(eigen_vals))


if __name__ == '__main__':
    mu = 0.3
    omega = 0.6
    delta_time = 0.1
    time_steps = 10
    lamb = numpy.cdouble(mu/2 + 1j * omega)
    rho_0 = generate_starting_condition(3)
    rho_approx_transformed = solve_num(rho_0, mu, time_steps, delta_time)
    solution_approx = transform_back(rho_approx_transformed, lamb, delta_time)
    rho_exact_transformed = solve_exact(rho_0, mu, time_steps, delta_time)
    solution_exact = transform_back(rho_exact_transformed, lamb, delta_time)
    print("rho_0 =")
    print(rho_0)
    for time_step in range(0,time_steps):
        print("--------------------------------------------------------------------------------------------")
        print("t = " + str((time_step+1)*delta_time) + ":")
        print("rho tilde approximiert:")
        print(rho_approx_transformed[time_step+1])
        print("rho tilde exakt:")
        print(rho_exact_transformed[time_step+1])
        error_tilde = calculate_difference(rho_approx_transformed[time_step+1], rho_exact_transformed[time_step+1])
        print("Der Fehler von rho tilde in der Spurklassenorm ist " + str(error_tilde))
        print("rho approximiert:")
        print(solution_approx[time_step+1])
        print("rho exakt:")
        print(solution_exact[time_step+1])
        error = calculate_difference(solution_approx[time_step+1], solution_exact[time_step+1])
        print("Der Fehler von rho in der Spurklassenorm ist " + str(error))


