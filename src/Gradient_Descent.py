import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import seaborn as sns
import IPython


'''
Geodesic Distance for Grassmannian
'''
def d_G(U_i, U_j):
    u,s,vt = np.linalg.svd(U_j @ U_j.T @ U_i)

    for i in range(len(s)):
        if s[i] - 1 > 1e-5:
            raise Exception('s[',i,'] = ', s[i])
        elif s[i] > 1:
            s[i] = 1

    distance = np.sqrt( np.sum( np.arccos(s)**2 ))

    return distance

'''
Geodesic Distance Gradient for Grassmannian
'''
def del_d_G(U_i, U_j):
    m = U_i.shape[0]
    r = U_i.shape[1]
    assert U_j.shape == U_i.shape

    v, s, wt = np.linalg.svd(U_j @ U_j.T @ U_i)
    dg_UU = np.zeros((m,r))

    for l in range(r):
        if s[l] < 1:
            dg_UU += -1 * np.arccos(s[l]) / np.sqrt(1 - s[l]**2) * np.outer(v[:, l] , wt[l, :])
        else:
            dg_UU += -1 * np.outer(v[:, l] , wt[l, :])

    #cap singular values
    for l in range(r):
        if s[l] - 1 > 1e-5:
            raise Exception('s[',l,'] = ', s[l])
        elif s[l] > 1:
            s[l] = 1

    distance = np.sqrt( np.sum( np.arccos(s)**2 ))

    if distance > 1e-10:
        dg_UU = (np.identity(m) - U_i @ U_i.T) / distance @ dg_UU
        return distance, dg_UU
    else:
        return distance, np.zeros((m,r))

'''
Distance in 2-d Embedding. The calculation is based on parameter method
'''
def d_p(p_i, p_j, method = 'Euclidean'):
    if method == 'Poincare':
        dist = 1 + 2 * (p_i - p_j).T @ (p_i - p_j) / (1 - p_i.T @ p_i) / (1 - p_j.T @ p_j)

        if dist < 1:
            print(p_i, p_j, dist)
            print( (1 - p_i.T @ p_i))
            print( (1 - p_j.T @ p_j))
        assert dist >= 1
        return np.arccosh(dist)

    elif method == 'Euclidean':
        dist = np.sqrt((p_i - p_j).T @ (p_i - p_j))
        return dist

    elif method == 'EuclideanL2':
        dist = (p_i - p_j).T @ (p_i - p_j)
        return dist

    elif method == 'Angle':
        dist = min(abs(p_i - p_j), 2 * np.pi - abs(p_i - p_j))
        return dist

    else:
        raise Exception('No method found for ' + method)


'''
Gradient of Distance in 2-d Embedding. The calculation is based on parameter method
'''
def del_d_p(p_i, p_j, method = 'Euclidean'):

    if method == 'Poincare':
        if (p_i - p_j).T @ (p_i - p_j) < 1e-8:
            return np.zeros(np.shape(p_i))

        alpha = 1 - p_i.T @ p_i
        beta = 1 - p_j.T @ p_j

        assert alpha != 0
        assert beta != 0

        gamma = 1 + 2 / alpha / beta * (p_i - p_j).T @ (p_i - p_j)

        assert gamma > 1

        deri = 4 / beta / np.sqrt(gamma**2 - 1)
        deri *= (p_j.T @ p_j - 2 * p_i.T @ p_j + 1) / (alpha**2) * p_i - p_j / alpha

        return deri

    elif method == 'Euclidean':
        if (p_i - p_j).T @ (p_i - p_j) < 1e-8:
            return np.zeros(np.shape(p_i))

        deri = (p_i - p_j) / np.sqrt((p_i - p_j).T @ (p_i - p_j))

        return deri

    elif method == 'EuclideanL2':
        deri = (p_i - p_j) * 2
        return deri

    elif method == 'Angle':
        if abs(p_i - p_j) <= np.pi and p_i >= p_j:
            return 1
        elif abs(p_i - p_j) >= np.pi and p_i >= p_j:
            return 1
        else:
            return -1

    else:
        raise Exception('No method found for ' + method)

'''
Calculate the distance matrix on the 2-d Embedding
'''
def dist_b_array(b_array, method):
    N = b_array.shape[0]

    dist_mat = np.zeros((N,N))
    for i in range(N):
        for j in range(i,N):
            dist_mat[i,j] = d_p(b_array[i], b_array[j], method = method)
            dist_mat[j,i] = dist_mat[i,j]

    return dist_mat
    
def dist_U_array(U_array):
    N = U_array.shape[0]

    dist_mat = np.zeros((N,N))
    for i in range(N):
        for j in range(i,N):
            dist_mat[i,j] = d_G(U_array[i], U_array[j])
            dist_mat[j,i] = dist_mat[i,j]

    return dist_mat

'''
P_Ball matrix
'''
def P_Ball(b_array, method, dist_b_array_mat, cost_func, beta = 1):
    K = b_array.shape[0]
    P_Ball_mat = np.zeros((K,K))

    if cost_func == 't-SNE':
        one_plus_dist_square = np.zeros((K,K))

        for i in range(K):
            for j in range(K):
                if i == j:
                    continue

                one_plus_dist_square[i,j] = 1/( 1 + dist_b_array_mat[i,j]**2 )

        denominator = np.sum(one_plus_dist_square)

        P_Ball_mat = one_plus_dist_square / denominator

        return P_Ball_mat, one_plus_dist_square
    if cost_func == 's-SNE':

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

    if cost_func == 'p-SNE':

        exp_distances = np.zeros((K,K))
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue

                exp_distances[i,j] = np.exp(-1 * dist_b_array_mat[i,j] /beta )

        denominator = np.sum(exp_distances)

        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                assert denominator != 0
                #assert distances[j] != 0

                P_Ball_mat[i,j] = exp_distances[i,j] / denominator

        return P_Ball_mat, exp_distances

    else:
        raise Exception('P_Ball Call: Cost_function ' + cost_func + ' not found!')
        return None

'''
P_Gr matrix
'''
def P_Gr(U_array, cost_func):
    K = U_array.shape[0]
    gamma_array = np.ones(K)

    P_Gr_mat = np.zeros((K,K))
    d_G_mat = np.zeros((K,K))
    for row in range(K):
        for col in range(K):
            d_G_mat[row, col] = d_G(U_array[row], U_array[col])

    for row in range(K):
        gamma_array[row] = np.std(d_G_mat[row,:])

    if cost_func == 't-SNE' or cost_func == 's-SNE' or cost_func == 'p-SNE':
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


    else:
        raise Exception('P_Gr Call: Cost_function ' + cost_func + ' not found!')
        return None

'''
Objective function for the opimization
'''
def L_obj(P_Ball_mat, P_Gr_mat, cost_func):

    filled_P_Ball_mat = P_Ball_mat.copy()
    filled_P_Gr_mat = P_Gr_mat.copy()

    for i in range(P_Ball_mat.shape[0]):
        for j in range(P_Ball_mat.shape[1]):
            if P_Ball_mat[i,j] == 0:
                filled_P_Ball_mat[i,j] = 1
            if P_Gr_mat[i,j] == 0:
                filled_P_Gr_mat[i,j] = 1

    if cost_func == 't-SNE' or cost_func == 's-SNE' or cost_func == 'p-SNE':
        sum = 0
        mat = np.log(np.divide(1, filled_P_Ball_mat))
        mat = np.multiply(mat, P_Gr_mat)
        sum += np.sum(mat)

        return sum

    else:
        raise Exception('L_obj Call: Cost_function ' + cost_func + ' not found!')
        return None

'''
Gradient of objective function for the opimization
'''
def del_L(b_array, P_Ball_mat, P_Gr_mat, cost_func, method, dist_b_array_mat, beta, support_mat):

    K = P_Ball_mat.shape[0]

    del_L_array = np.zeros(b_array.shape)

    if cost_func == 't-SNE':
        one_plus_dist_square = support_mat
        for i in range(K):
            del_array = [del_d_p(b_array[i], b_array[k], method = method) if k != i else 0 for k in range(K)]

            del_L_array[i] = 4 * sum([   dist_b_array_mat[i,j] * (P_Gr_mat[i,j] - P_Ball_mat[i,j]) * one_plus_dist_square[i,j] * del_array[j]   for j in range(K)])

    elif cost_func == 's-SNE':

        for i in range(K):
            del_array = [del_d_p(b_array[i], b_array[k], method = method) if k != i else 0 for k in range(K)]

            del_L_array[i] = 4 * sum([  dist_b_array_mat[i,j] * (P_Gr_mat[i,j] - P_Ball_mat[i,j]) * del_array[j]   for j in range(K)])


    elif cost_func == 'p-SNE':

        for i in range(K):
            del_array = [del_d_p(b_array[i], b_array[k], method = method) if k != i else 0 for k in range(K)]

            del_L_array[i] = 2 * sum([(P_Gr_mat[i,j] - P_Ball_mat[i,j]) * del_array[j]   for j in range(K)])


    return del_L_array

'''
Step in the Gradient direction. Then perform rectraction if necessary.
Retraction mainly for Poincare disk. It is used after the gradient step calculation. It can also be used
to shrink the points back to the disk for Euclidean Disk.
'''
def retraction(b_array, del_L_array, eta, method, eps = 1e-5):
    new_b_array = np.zeros(b_array.shape)
    K = b_array.shape[0]

    for i in range(K):
        #gradient step
        if method == 'Poincare':
            projectee = b_array[i] - eta * (1 - b_array[i].T @ b_array[i])**2 / 4 * del_L_array[i]
        elif method == 'Euclidean':
            projectee = b_array[i] - eta * del_L_array[i]
        elif method == 'EuclideanL2':
            projectee = b_array[i] - eta * del_L_array[i]
        elif method == 'Angle':
            projectee = b_array[i] - eta * del_L_array[i]
        else:
            raise Exception('No method found for:' + method)


        #retraction on circle
        if method == 'Angle':
            while (projectee[0] < 0):
                projectee += 2 * np.pi
            while (projectee[0] > 2 * np.pi):
                projectee -= 2 * np.pi

            new_b_array[i] = projectee

        if (method == 'Euclidean' or method == 'EuclideanL2'):
            new_b_array[i] = projectee

        else:
            #doing projection on disk
            norm = projectee.T @ projectee
            if norm >= 1:
                new_b_array[i] = projectee / (norm + eps)
            else:
                new_b_array[i] = projectee

    return new_b_array
