
import numpy as np
import matplotlib.pyplot as plt
from Grasscare import *

import time
from mpl_toolkits.mplot3d import Axes3D

'''
Time Log for 100 ITER

3-D, 200 pt
Base: 132 s
GPT-Improved: 92s

Parallelize del_L: +90s

'''


def main():

    U_array, labels = U_array_3d_init(1000)
    print(U_array.shape)
                                    
    plot_3d(U_array, labels)
    
    start_time = time.time()
    training_history = grasscare(U_array, eta = 2, moment = 0.9, max_iter = 500, init = '3D', beta = 2)
    
    time_cost = time.time() - start_time
    print('Time:', np.round(time_cost))
     
    plot_video(training_history, labels, cmap_name = 'rainbow')
    #plot_process(training_history, labels, save = True)
    plot_embedding(training_history, labels, save = True)
    
    np.savez_compressed('history-input', training_history = training_history, U_array = U_array, labels = labels, allow_pickle = True)
    
    
  

if __name__ == '__main__':
    main()
