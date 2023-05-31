from Distances import *
from scipy.spatial.distance import cdist
import numpy as np


'''
P_Ball matrix
'''
'''
def P_Ball(b_array, dist_b_array_mat, beta = 2):
    K = b_array.shape[0]
    P_Ball_mat = np.zeros((K,K))

    exp_distances = np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            if i == j:
                continue

            exp_distances[i,j] = np.exp(-1 * dist_b_array_mat[i,j]**2 /beta )

    denominator = np.sum(exp_distances)

    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            assert denominator != 0
            #assert distances[j] != 0

            P_Ball_mat[i,j] = exp_distances[i,j] / denominator

    return P_Ball_mat, exp_distances
'''
def P_Ball(b_array, dist_b_array_mat, beta):
    K = b_array.shape[0]
    P_Ball_mat = np.zeros((K, K))

    exp_distances = np.exp(-dist_b_array_mat ** 2 / beta)
    np.fill_diagonal(exp_distances, 0)

    denominator = np.sum(exp_distances)

    if denominator != 0:
        P_Ball_mat = exp_distances / denominator

    return P_Ball_mat, exp_distances



'''
P_Gr matrix
'''
def P_Gr(U_array):
    K = U_array.shape[0]
    gamma_array = np.ones(K)

    P_Gr_mat = np.zeros((K,K))
    d_G_mat = np.zeros((K,K))
    #pool = mp.Pool(mp.cpu_count())
    for row in range(K):
        for col in range(K):
            d_G_mat[row, col] = d_G(U_array[row], U_array[col])
            #d_G_mat[row, col] = pool.apply(d_G, args=(U_array[row], U_array[col]))
    #pool.close()   

    for row in range(K):
        gamma_array[row] = np.std(d_G_mat[row,:])


    denominator = []
    for i in range(K):
        denominator.append( sum([np.exp(-1 * d_G_mat[i, k]**2 / 2 / gamma_array[i]**2) if k != i else 0 for k in range(K)]) )

    for i in range(K):
        for j in range(K):
            if i == j:
                continue

            P_Gr_mat[i, j] = np.exp(-1 * d_G_mat[i, j]**2 / 2 / gamma_array[i]**2) / denominator[i]
            P_Gr_mat[i, j] += np.exp(-1 * d_G_mat[j, i]**2 / 2 / gamma_array[j]**2) / denominator[j]
            P_Gr_mat[i, j] /= (2 * K)

    return P_Gr_mat

