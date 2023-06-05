

import numpy as np
import matplotlib.pyplot as plt

from Distances import *
from ProbMatrix import *
from Loss import *
from GradientDescent import *
from Init import *
from Plots import *


from scipy.spatial.distance import cdist


def b_array_init(count):
    theta_array = np.random.rand(count) * 2 * np.pi
    r_array = np.random.rand(count)

    x = np.cos(theta_array) * r_array
    y = np.sin(theta_array) * r_array

    return np.array([x,y]).T




def grasscare(U_array, gradient_method = 'MomentumGD', eta = 1, moment = 0.9, max_iter = 400, init = 'random', beta = 2):
    U_array = np.array(U_array)
    #random init Poincare Embedding
    if init == 'random':
        b_array = b_array_init(U_array.shape[0])
    elif init == '3D':
        b_array = U_array[:,:2,0] * 0.95
    else:
        assert False

    # Calculate Probability Matrix in Grassmanian
    P_Gr_mat = P_Gr(U_array = U_array)


    #Init momentum with 0
    old_del = np.zeros(b_array.shape)

    #record all the training history
    history = []

    #ADAM variables
    if gradient_method == 'ADAM':
        m = np.zeros_like(b_array)
        v = np.zeros_like(b_array)

    for iter in range(max_iter):

        # Calcuate Probability Matrix in Poincare Disk
        dist_b_array_mat = cdist(b_array, b_array, metric=d_p)
        P_Ball_mat, _ = P_Ball(b_array, dist_b_array_mat, beta = beta)

        #Calculate Loss and Gradient
        Loss = L_obj(P_Ball_mat, P_Gr_mat)
        del_L_array = del_L(b_array,
                            P_Ball_mat,
                            P_Gr_mat,
                            dist_b_array_mat)

        if gradient_method == 'MomentumGD':
            new_b_array = retraction_GD(b_array,
                                        del_L_array,
                                        eta,
                                        moment,
                                        old_del)
        elif gradient_method == 'ADAM':
            new_b_array, m, v = retraction_ADAM(b_array,
                                        del_L_array,
                                        eta,
                                        beta1 = 0.5,
                                        beta2 = 0.7,
                                        t = iter + 1,
                                        m = m,
                                        v = v)
        else:
            assert False


        old_del =  del_L_array
        b_array = new_b_array

        record = {'iter': iter, 'obj': Loss, 'b_array': b_array}
        history.append(record)

        print(iter, Loss)

    return history
