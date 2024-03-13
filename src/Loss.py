import numpy as np
from Distances import *
from ProbMatrix import *

'''
Objective function for the opimization
'''
'''
def L_obj(P_Ball_mat, P_Gr_mat):

    filled_P_Ball_mat = P_Ball_mat.copy()
    filled_P_Gr_mat = P_Gr_mat.copy()

    for i in range(P_Ball_mat.shape[0]):
        for j in range(P_Ball_mat.shape[1]):
            if P_Ball_mat[i,j] == 0:
                filled_P_Ball_mat[i,j] = 1
            if P_Gr_mat[i,j] == 0:
                filled_P_Gr_mat[i,j] = 1

    sum = 0
    mat = np.log(np.divide(1, filled_P_Ball_mat))
    mat = np.multiply(mat, P_Gr_mat)
    sum += np.sum(mat)

    return sum
'''
def L_obj(P_Ball_mat, P_Gr_mat):
    mat = np.log(1 / np.maximum(P_Ball_mat, 1e-10))  # Avoid division by zero
    mat = mat * P_Gr_mat
    total_sum = np.sum(mat)

    return total_sum



'''
Gradient of objective function for the opimization
'''
'''
def del_L(b_array, P_Ball_mat, P_Gr_mat, dist_b_array_mat):

    K = P_Ball_mat.shape[0]

    del_L_array = np.zeros(b_array.shape)


    for i in range(K):
        del_array = [del_d_p(b_array[i], b_array[k]) if k != i else 0 for k in range(K)]

        del_L_array[i] = 4 * sum([  dist_b_array_mat[i,j] * (P_Gr_mat[i,j] - P_Ball_mat[i,j]) * del_array[j]   for j in range(K)])


    return del_L_array
'''

'''
def del_L(b_array, P_Ball_mat, P_Gr_mat, dist_b_array_mat):
    K = P_Ball_mat.shape[0]

    del_L_array = np.zeros(b_array.shape)

    for i in range(K):
        del_array = np.array([del_d_p(b_array[i], b_array[k]) if k != i else 0 for k in range(K)])
        del_L_array[i] = 4 * np.sum(dist_b_array_mat[i, :K] * (P_Gr_mat[i, :K] - P_Ball_mat[i, :K]) * del_array[:K])

    return del_L_array

'''

'''
import multiprocessing as mp

def process_iteration(i, b_array, P_Ball_mat, P_Gr_mat, dist_b_array_mat):
    K = P_Ball_mat.shape[0]
    del_array = np.array([del_d_p(b_array[i], b_array[k]) if k != i else 0 for k in range(K)])
    return 4 * np.sum(dist_b_array_mat[i, :K] * (P_Gr_mat[i, :K] - P_Ball_mat[i, :K]) * del_array[:K])

def del_L(b_array, P_Ball_mat, P_Gr_mat, dist_b_array_mat):
    K = P_Ball_mat.shape[0]
    del_L_array = np.zeros(b_array.shape)

    with mp.Pool() as pool:
        results = pool.starmap(process_iteration, [(i, b_array, P_Ball_mat, P_Gr_mat, dist_b_array_mat) for i in range(K)])

    del_L_array = np.array(results)
    return del_L_array

'''


def del_L(b_array, P_Ball_mat, P_Gr_mat, dist_b_array_mat):
    K = P_Ball_mat.shape[0]

    del_L_array = np.zeros(b_array.shape)

    for i in range(K):
        del_array = [del_d_p(b_array[i], b_array[k]) if k != i else np.zeros(2) for k in range(K)]
        del_array = np.array(del_array)
        # print(del_array.shape)
        # del_L_array[i] = 4 * np.sum(dist_b_array_mat[i, :K] * (P_Gr_mat[i, :K] - P_Ball_mat[i, :K]) * del_array[:K])
        del_L_array[i] = 4 * (dist_b_array_mat[i, :K] * (P_Gr_mat[i, :K] - P_Ball_mat[i, :K])) @ del_array[:K]

    return del_L_array
