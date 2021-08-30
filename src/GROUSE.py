import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import seaborn as sns
import IPython

from Initialization import *
from Plot_Functions import *


'''
Solution for the least square
'''
def ls_solution(U, v, Omega):
    a = Omega @ U
    b = Omega @ v

    w = np.linalg.lstsq(a = a, b = b, rcond = None)[0]
    return w


'''
GROUSE gradient descent for 1 step
'''
def GROUSE_gradient_descent(U, v, Omega, eta):

    w = ls_solution(U, v, Omega)
    #print(w.shape)

    p = U @ w

    r = Omega @ (v - p)

    sigma = np.linalg.norm(r) * np.linalg.norm(p)

    U = U + np.outer( ((np.cos(eta * sigma) - 1) * p / np.linalg.norm(p) + (np.sin(eta * sigma)) * r / np.linalg.norm(r)),   w / np.linalg.norm(w) )

    return U

'''
Main process for GROUSE, responsible for calling gradient descent
'''
def GROUSE(GROUSE_dict, max_iter = 100, eta = 1):
    U = GROUSE_dict['U_0']
    v_array = GROUSE_dict['v_array']
    Omega_array = GROUSE_dict['Omega_array']

    U_record = [U]

    for iter in range(max_iter):

        for t in range(v_array.shape[0]):

            v_t = v_array[t]
            Omega_t = Omega_array[t]

            U = GROUSE_gradient_descent(U = U, v = v_t, Omega = Omega_t, eta = eta)

            U_record.append(U.copy())


    return np.array(U_record)


'''
Initialization of GROUSE. Generate observed_vectors partially observed vectors in v_array.
Starting point can be load with U_0, or random generated with U_0_load = False.
Responsible for finding the geodesic path.
'''
def GROUSE_init(U_array, observed_vectors , missing_percentage , U_0_load = False, U_0 = None):

    m = U_array[0].shape[0]
    r = U_array[0].shape[1]
    n = U_array.shape[0]


    #v_array init
    v_multiplier = np.random.random((r, observed_vectors))
    v_array = (U_array[0] @ v_multiplier).T

    #Omega_array init
    Omega_array = []
    for i in range(observed_vectors):
        Omega_array.append(np.identity(v_array.shape[1]))
    Omega_array = np.array(Omega_array)


    for i in range(observed_vectors):
        choice = np.random.choice(m, size = int(m * (missing_percentage)), replace = False )
        for c in choice:
            v_array[i,c] = 0
            Omega_array[i,c,c] = 0

    #U_0 init
    if U_0_load == False:
        q_i,r_i = np.linalg.qr(np.random.randn(m,r))
        U_0 = q_i


    #path = find_geodesic(U_0 = U_0, U_t = U_array[0], step_size = 0.001, steps = geodesic_steps)
    GROUSE_dict = {'Omega_array': Omega_array, "v_array": v_array, "U_0": U_0}#, 'geodesic': path}

    return GROUSE_dict


'''
Finding the geodesic path from U_0 to U_t.
'''
def find_geodesic(U_0, U_t, steps, step_size = 0.001):
    m = U_0.shape[0]
    r = U_0.shape[1]

    U_i = U_0
    path = []
    path.append(U_0.copy())

    cummulative_step_distance = 0
    while (True):
        dg_UU, distance = gradient_geodesic(U_i, U_t)

        if distance == 0:
            break

        real_step = step_size * distance

        Gamma_i, Del_i, ET_i = np.linalg.svd( -1 * real_step * dg_UU , full_matrices= False)
        first_term = np.concatenate((U_i @ET_i.T, Gamma_i), axis = 1)
        second_term = np.concatenate((np.diag(np.cos( Del_i)), np.sin(np.diag(Del_i))), axis = 0)

        U_i = first_term @ second_term @ ET_i

        cummulative_step_distance += real_step

        if cummulative_step_distance > step_size * 0.5:
            path.append(U_i.copy())
            cummulative_step_distance = 0

        #print(distance)

    if len(path) > steps:
        inter = len(path) // steps + 1

        new_path = [path[i * inter] for i in range(len(path) // inter)]
        path = new_path

    while len(path) < steps:
        path.append(path[-1])

    return np.array(path)



'''
Gradient for Geodesic on Grassmannian.
'''
def gradient_geodesic(U_0, U_t):
    m = U_0.shape[0]
    r = U_0.shape[1]

    u_j, s_j, vt_j = np.linalg.svd(U_t @ U_t.T @ U_0)

    dg_UU = np.zeros((m,r))
    for r_index in range(r):
        if s_j[r_index] < 1:
            dg_UU += -1 * np.arccos(s_j[r_index]) / np.sqrt(1 - s_j[r_index]**2) * np.outer(u_j[:, r_index] , vt_j[r_index, :])
        else:
            dg_UU += -1 * np.outer(u_j[:, r_index] , vt_j[r_index, :])

    #cap singular values
    for r_index in range(r):
        if (s_j[r_index] - 1 > 0):
            s_j[r_index] = 1

    distance = np.sqrt( np.sum( np.arccos(s_j)**2 ))

    if distance > 1e-6:
        dg_UU = (np.identity(m) - U_0 @ U_0.T) / distance @ dg_UU
    else:
        distance = 0


    return dg_UU, distance
