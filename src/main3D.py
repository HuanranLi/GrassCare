
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

    U_array, labels = U_array_3d_init(300)
    print(U_array.shape)
                                    
    #plot_3d(U_array, labels)
    

    
    start_time = time.time()
    training_history_GD = grasscare(U_array, gradient_method = 'MomentumGD', eta = 2, moment = 0.9, max_iter = 300, init = '3D', beta = 2)
    time_cost = time.time() - start_time
    print('Time:', np.round(time_cost))
     
    plot_video(training_history_GD, labels, cmap_name = 'rainbow', name = 'Video_GD')
    plot_embedding(training_history_GD, labels, save = True, name = 'Embedding_GD')
    
    obj_array_GD = [i['obj'] for i in training_history_GD]

    
    start_time = time.time()
    training_history_ADAM = grasscare(U_array, gradient_method = 'ADAM', eta = 0.01, max_iter = 300, init = '3D', beta = 2)
    time_cost = time.time() - start_time
    print('Time:', np.round(time_cost))
     
    plot_video(training_history_ADAM, labels, cmap_name = 'rainbow', name = 'Video_ADAM')
    plot_embedding(training_history_ADAM, labels, save = True, name = 'Embedding_ADAM')
    
    obj_array_ADAM = [i['obj'] for i in training_history_ADAM]
    
    plt.plot(obj_array_GD, label = 'Gradient Descent')
    plt.plot(obj_array_ADAM, label = 'ADAM')
    plt.show()
    
    
    
  

if __name__ == '__main__':
    main()
