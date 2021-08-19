
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1,'./src')



from Initialization import *
from Gradient_Descent import *
from grasscare import *
from GROUSE import *
from Plot_Functions import *

import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import seaborn as sns
import IPython


def main():
    ambient_dimension = m = 25
    rank = r = 5
    count = N = 30
    clusters = K = 3

    S, labels, optional_params = demo_init(m, r, N, K)
    optional_params['video_tail'] = 5

    embedding = grasscare_plot(S = S, labels = labels, video = True, optional_params = optional_params)

    #print(embedding.shape)



def demo_init(m,r,N,K):
    print('\n########## ClusterPath initialization ##########')
    U_array, labels, centers = U_array_init(ambient_dimension = m, rank = r, count = N, clusters = K,  err_var = 1)
    steps = 10
    S = np.zeros((steps, N, m, r))

    for U_0_index in range(U_array.shape[0]):
        U_0 = U_array[U_0_index]
        label = labels[U_0_index]
        U_t = centers[label]

        path = find_geodesic(U_0 = U_0, U_t = U_t, steps = steps)

        S[:, U_0_index] = path

    print('S Shape:', S.shape)

    optional_params = {'Targets': centers}

    return S, labels, optional_params





if __name__ == '__main__':
    main()
