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

def retraction_GD(b_array, del_L_array, eta, moment, old_del, eps=1e-5):
    K = b_array.shape[0]
    proj_factor = eta * (1 - np.square(b_array).sum(axis=1, keepdims=True)) ** 2 / 4

    new_change = eta * (1 - np.square(b_array).sum(axis=1, keepdims=True)) ** 2 / 4 * del_L_array + moment * (eta * (1 - np.square(b_array).sum(axis=1, keepdims=True)) ** 2 / 4 * old_del)
    new_b_array = b_array - new_change

    return new_b_array
