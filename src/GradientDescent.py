import numpy as np

from Distances import *
from ProbMatrix import *
from Loss import *

'''
def retraction_GD(b_array, del_L_array, eta, moment, old_del, eps = 1e-5):
    new_b_array = np.zeros(b_array.shape)
    K = b_array.shape[0]


    for i in range(K):
        #gradient step

        #projectee = b_array[i] - eta * (1 - b_array[i].T @ b_array[i])**2 / 4 * del_L_array[i]
        new_change = eta * (1 - b_array[i].T @ b_array[i])**2 / 4 * del_L_array[i] + moment * (eta * (1 - b_array[i].T @ b_array[i])**2 / 4 * old_del[i])

        new_b_array[i] = b_array[i] - new_change


    return new_b_array
'''

def retraction_GD(b_array, del_L_array, eta, moment, old_del, eps=1e-5, boundary_limit = 1e-5):
    K = b_array.shape[0]
    proj_factor = eta * (1 - np.square(b_array).sum(axis=1, keepdims=True)) ** 2 / 4

    new_change = eta * (1 - np.square(b_array).sum(axis=1, keepdims=True)) ** 2 / 4 * del_L_array + moment * (eta * (1 - np.square(b_array).sum(axis=1, keepdims=True)) ** 2 / 4 * old_del)
    new_b_array = b_array - new_change

    # Precompute norms of all elements
    norms = np.linalg.norm(new_b_array, axis=1)

    # Find indices where norm > 1 - boundary_limit
    indices = norms > 1 - boundary_limit

    # Normalize elements in-place
    for i in range(len(new_b_array)):
        if indices[i]:
            new_b_array[i] = new_b_array[i] / (norms[i] + 2*boundary_limit)

    return new_b_array


def retraction_ADAM(b_array, del_L_array, alpha, beta1, beta2, t, m, v, eps=1e-8, boundary_limit = 1e-5):

    # ADAM update
    m = beta1 * m + (1 - beta1) * del_L_array
    v = beta2 * v + (1 - beta2) * np.square(del_L_array)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    new_b_array = b_array - alpha * m_hat / (np.sqrt(v_hat) + eps)

    # Precompute norms of all elements
    norms = np.linalg.norm(new_b_array, axis=1)

    # Find indices where norm > 1 - boundary_limit
    indices = norms > 1 - boundary_limit

    # Normalize elements in-place
    #if np.any(indices):
    #    new_b_array[indices] /= (norms[indices] + boundary_limit)

    for i in range(len(new_b_array)):
        if indices[i]:
            new_b_array[i] = new_b_array[i] / (norms[i] + 2*boundary_limit)

    return new_b_array, m, v
