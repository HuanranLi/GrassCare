
import numpy as np
import matplotlib.pyplot as plt
from Grasscare import *

import time
from mpl_toolkits.mplot3d import Axes3D


'''
Time Log for 100 ITER

m = 100, r = 5, n = 200
Base: 159 s
GPT-Improved: 97s

'''

def main():
    U_array, labels, Centroids = U_array_init(ambient_dimension = 100, 
                           rank = 5, 
                           count = 200, 
                           clusters = 2,
                           err_var = 0.0)
                           
    #U_array, labels = U_array_3d_init(200)
    print(U_array.shape)
                                    
    #plot_3d(U_array, labels)
    
    start_time = time.time()
    training_history = grasscare(U_array, eta = 2, moment = 0.9, max_iter = 100, beta = 1)
    
    time_cost = time.time() - start_time
    print('Time:', np.round(time_cost))
     
    plot_video(training_history, labels, cmap_name = 'rainbow')
    #plot_process(training_history, labels, save = True)
    plot_embedding(training_history, labels, save = True)
    
    
    
    np.save('history.npy', training_history, allow_pickle=True)
  

if __name__ == '__main__':
    main()
