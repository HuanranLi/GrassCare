import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import seaborn as sns
import IPython


from Plot_Functions import *

'''
Initialize a group of orthonormal vectors/matrix. If the clusters > 0, then the
points will be distributed in several clusters. Otherwise, points will be
distributed randomly.

Parameter:
1. vector/matrix dimension: ambient_dimension * rank
2. count: the number of points the caller want to initialized
3. clusters: the number of clusters the caller want to initialized
4. bound_zero: if x < bound_zero, it will be treated as 0
5. err_var: the variance of the clusters
'''
def U_array_init(ambient_dimension, rank, count, clusters, bound_zero = 1e-10,  err_var = 0.1):
    m = ambient_dimension
    r = rank

    if clusters == 0:
        U_array = [np.random.randn(m,r) for i in range(count)]
        for i in range(count):
            if r == 1:
                U_array[i] /= np.linalg.norm(U_array[i])
                if m == 3:
                    U_array[i][2] = abs(U_array[i][2])
            else:
                q_i,r_i = np.linalg.qr(U_array[i])
                U_array[i] = q_i

        return np.array(U_array)
    else:
        labels = []
        assert count % clusters == 0

        #calculate points per cluster
        n = count // clusters

        U_array = [np.random.randn(m,r) for i in range(clusters)]
        new_U_array = []
        for i in range(clusters):
            if r == 1:
                U_array[i] /= np.linalg.norm(U_array[i])
                if m == 3:
                    U_array[i][2] = abs(U_array[i][2])
            else:
                q_i,r_i = np.linalg.qr(U_array[i])
                U_array[i] = q_i

            #make sure its orthogonal
            assert np.linalg.norm(U_array[i].T @ U_array[i] - np.identity(r)) < bound_zero
            #make sure its normal
            assert  np.linalg.norm( np.linalg.norm(U_array[i], axis = 0) - np.ones(r) )  < bound_zero


            #generate points per cluster with parameter err_var
            for j in range(n):
                new_U = U_array[i] + np.random.randn(U_array[i].shape[0], U_array[i].shape[1])* err_var
                u,s,vt = np.linalg.svd(new_U, full_matrices= False)

                new_U_array.append(u@vt)

                if r == 1 and m == 3:
                    new_U_array[-1][2] = abs(new_U_array[-1][2])

                labels.append(i)


    return np.array(new_U_array), np.array(labels), np.array(U_array)

'''
Initialize a group of points on the determined domain.

Parameter:
1. count: number of points the caller want to initialized
2. domain: disk = a unit ball; circle = a unit circle (no area included)
3. style: 'zero' = init all points at origin; 'random' = random
'''
def b_array_init(count, domain = 'disk', style = 'random', U_array = None, eps = 1e-5):
    if style == 'random':
        theta_array = np.random.rand(count) * 2 * np.pi

        if domain == 'circle':
            return theta_array.reshape((count, 1))
        if domain == 'disk':
            r_array = np.random.rand(count)

            x = np.cos(theta_array) * r_array
            y = np.sin(theta_array) * r_array

        return np.array([x,y]).T

    if style == 'zeros':
        if domain == 'circle':
            return np.zeros((count, 1))
        if domain == 'disk':
            x = np.zeros(count)
            y = np.zeros(count)

        return np.array([x,y]).T

    if style == 'PCA':
        U_array_flat = U_array.reshape((U_array.shape[0], -1))
        u, s, vt = np.linalg.svd(U_array_flat.T, full_matrices= False)
        b_array = (np.diag(s[:2]) @ vt[:2,:]).T

        #scale back to the disk
        for b in b_array:
            if np.linalg.norm(b) >= 1:
                b_array /= (np.linalg.norm(b) + eps)

        return b_array


    else:
        raise Exception('No style found for' + style)
