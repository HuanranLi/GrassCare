import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import seaborn as sns
import IPython

from datetime import datetime

import shutil
import os

from Gradient_Descent import *
from Plot_Functions import *
from Initialization import *
from GROUSE import *

def grasscare_plot(S, labels, video, optional_params = {}):


    print('\n######################### Grasscare 1.1.7 #########################')

    U_array = S



    ####################################################
    # Defaulting parameters
    ####################################################

    #optional parameter for how b_array is initialized.
    if 'b_array_init_syle' not in optional_params:
        b_array_init_syle = 'random'
    else:
        b_array_init_syle = optional_params['b_array_init_syle']

    #optional parameter for how b_array is embedded.
    if 'embedding_method' not in optional_params:
        embedding_method = 'Poincare'
    else:
        embedding_method = optional_params['embedding_method']

    #optional parameter for cost function.
    if 'cost_function' not in optional_params:
        cost_function = 's-SNE'
    else:
        cost_function = optional_params['cost_function']

    #optional parameter for cost function.
    if 'max_epoch' not in optional_params:
        max_epoch = 500
    else:
        max_epoch = optional_params['max_epoch']

    #optional parameter for cost function.
    if 'step_size' not in optional_params:
        step_size = 1
    else:
        step_size = optional_params['step_size']

    #optional parameter for beta for P_B
    if 'beta' not in optional_params:
        beta = 1
    else:
        beta = optional_params['beta']

    #optional parameter for beta for P_B
    if 'objective_plot' not in optional_params:
        objective_plot = True
    else:
        objective_plot = optional_params['objective_plot']

    #optional parameter for the tail behind the path for video
    if 'video_tail' not in optional_params:
        video_tail = -1
    else:
        video_tail = optional_params['video_tail']

	#optional parameter for printing tools at GoogleColab
    if 'GoogleColab' not in optional_params:
        GoogleColab = False
    else:
        GoogleColab = optional_params['GoogleColab']

    if 'final_picture' not in optional_params:
        final_picture = False
    else:
        final_picture = optional_params['final_picture']


    if 'no_graph' not in optional_params:
        no_graph = False
    else:
        no_graph = optional_params['no_graph']

    if 'min_value_gain' not in optional_params:
        min_value_gain = 1e-5
    else:
        min_value_gain = optional_params['min_value_gain']

    if 'min_eta' not in optional_params:
        min_eta = 1e-5
    else:
        min_eta = optional_params['min_eta']

    if 'folder_name' not in optional_params:
        folder_name = 'Time'
    else:
        folder_name = optional_params['folder_name']

    if 'folder_path' not in optional_params:
        folder_path = ''
    else:
        folder_path = optional_params['folder_path']

    ############################################################
    # Part 1: Single Time Frame Plotting
    ############################################################
    t = len(U_array.shape) - 2
    if t == 1:
        print('Single Time Frame Mode: On')

        #Color Map For the plot
        cmap=plt.get_cmap("jet")
        c_array = [cmap(i / max(labels)) for i in labels]

        #name for the video
        if video and not no_graph:
            print('\nNote: Optimization video will be generated instead of path video!')
            gif_output = 'Optimization_Process.gif'
        else:
            gif_output = None

        #count of subspaces
        N = U_array.shape[0]
        b_array = b_array_init(N, domain = 'disk',
                                style = b_array_init_syle,
                                U_array = U_array)


        arrays_dict = {}
        arrays_dict['U_array'] = U_array
        arrays_dict['b_array'] = b_array
        arrays_dict['c_array'] = c_array

        new_b_array, info = grasscare_train(arrays_dict = arrays_dict,
                        method = embedding_method,
                        epoch = max_epoch,
                        eta = step_size,
                        beta = beta,
                        cost_func = cost_function,
                        obj_plot = objective_plot,
                        b_array_path = True,
                        gif_output = gif_output,
                        video_tail = video_tail,
                        google_colab = GoogleColab,
                        min_value_gain = min_value_gain,
                        min_eta = min_eta)


        if not no_graph:
            plot_b_array(new_b_array,
                    save = True,
                    title = 'Grasscare',
                    format = 'pdf',
                    color_map= c_array,
                    labels = labels,
                    plot = final_picture)

        if video and not no_graph:
            plot_b_array(new_b_array,
                save = True,
                title = 'Optimization_Process',
                format = 'pdf',
                color_map= c_array,
                plot = final_picture,
                labels = labels,
                tail = -1,
                b_array_path = info['b_array_path'])


        if not no_graph:
            clean_up(folder_name, folder_path = folder_path)


        print('######################### Grasscare END ###########################\n')

        return new_b_array, info

    ############################################################
    # Part 2: Multiple Time Frame Plotting
    ############################################################
    else:
        print('Multiple Time Frames Mode: On')

        merged_U_array = U_array[:,0]
        for i in range(1,U_array.shape[1]):
            merged_U_array = np.concatenate( (merged_U_array, U_array[:,i]), axis = 0 )

        if 'Targets' in optional_params:
            merged_U_array = np.concatenate( (optional_params['Targets'], merged_U_array), axis = 0 )


        #count of subspaces
        N = merged_U_array.shape[0]
        b_array = b_array_init(N, domain = 'disk',
                                style = b_array_init_syle,
                                U_array = merged_U_array)

        arrays_dict = {}
        arrays_dict['U_array'] = merged_U_array
        arrays_dict['b_array'] = b_array

        print('Reshaped S shape:', merged_U_array.shape)

        new_b_array, info = grasscare_train(arrays_dict = arrays_dict,
                        method = embedding_method,
                        epoch = max_epoch,
                        eta = step_size,
                        beta = beta,
                        cost_func = cost_function,
                        obj_plot = objective_plot,
						google_colab = GoogleColab,
                        min_value_gain = min_value_gain,
                        min_eta = min_eta)

        if 'Targets' in optional_params:
            targets_count = len(optional_params['Targets'])
        else:
            targets_count = 0

        if 'path_names' in optional_params:
            path_names = optional_params['path_names']
        else:
            path_names = []


        if not no_graph:
            plot_b_array_path(b_array = new_b_array,
                            labels = labels,
                            paths_count = U_array.shape[1],
                            path_length = U_array.shape[0],
                            targets_count = targets_count,
                            video = video,
                            title = 'Grasscare',
                            save = True,
                            plot = final_picture,
                            format = 'pdf',
                            tail = video_tail,
                            path_names = path_names
                            )




        b_array = np.zeros((U_array.shape[0],U_array.shape[1],2))
        for col in range(U_array.shape[1]):
            b_array[:,col] = new_b_array[targets_count + col * U_array.shape[0] : targets_count + (col+1) * U_array.shape[0]]

        if not no_graph:
            clean_up(folder_name, folder_path = folder_path)

        print('######################### Grasscare END ###########################\n')

        return b_array, info



'''
Major Grasscare algorithms. Including gradient descent and graphing.
'''
def grasscare_train(arrays_dict, #data
            method, #Decide the embedding: Poincare, Euclidean, EuclideanL2
            cost_func, #cost functions: s-SNE, t-SNE, p-SNE(Symmetric SNE with distance instead of distrance^2 for P_Ball)
            epoch, #max epoch
            eta, #step_size
            pretrained_P_Gr_mat = [0], #pre-calculated P_Gr (save time)
            #b_array_choice = 'PCA', #Choice for b_array initialization: PCA, random, unrestrained_PCA (points are not necessary to be in the disk), PCA_GROUSE(GROUSE mode)
            printing = False, #printing the graphs during the training (cost time)
            beta = 1, #parameter for P_Ball
            plot_per_n_iter = 5, #plots per n iterations (The graphs are shown if printing is True, else graphs will be merged to gif)
            plot_all_before = 50, #ignore the plot_per_n_iter for the first n iterations. (check early iterations)
            gif_output = None, #name for the gif file. If None, then no gif is generated.
            debug = False, #printing values for debuging.
            obj_plot = True, #plot the objective values in the end of training
            printing_update = True, #update of objective values during the training
            b_array_path = False, #save all b_array intermediate values during the training, will be returned in info['b_array_path']
            video_tail = 2,
			google_colab = False,
            min_value_gain = 1e-5,
            min_eta = 1e-5):

    np.seterr(divide = 'ignore')

    c_array_name = 'c_array'
    b_array = arrays_dict['b_array']


    ############################################################################
    if printing_update:
        if google_colab:
            print('GoogleColab Printing Mode: On')
            out = display(IPython.display.Pretty('Starting'), display_id=True)
        else:
            print('Starting', end = '\r')


    U_array = arrays_dict['U_array']
    if np.shape(pretrained_P_Gr_mat)[0] == 1:
        P_Gr_mat = P_Gr(U_array = U_array, cost_func = cost_func)
    else:
        P_Gr_mat = pretrained_P_Gr_mat.copy()

    if debug:
        print('P_Gr')
        print(P_Gr_mat)

    dist_b_array_mat = dist_b_array(b_array = b_array, method = method)

    P_Ball_mat, support_mat = P_Ball(b_array = b_array,
                        method = method,
                        dist_b_array_mat = dist_b_array_mat,
                        cost_func = cost_func,
                        beta = beta)

    if debug:
        print('P_Ball')
        print(P_Ball_mat)
    assert abs(np.sum(P_Ball_mat) - 1) < 1e-5


    del_L_array = del_L(b_array = b_array,
                        P_Ball_mat = P_Ball_mat,
                        P_Gr_mat = P_Gr_mat,
                        cost_func = cost_func,
                        dist_b_array_mat = dist_b_array_mat,
                        method = method,
                        beta = beta,
                        support_mat = support_mat)

    if debug:
        print('del_L')
        print(del_L_array)

    new_b_array = retraction(b_array = b_array,
                             del_L_array = del_L_array,
                             eta = eta,
                             method = method)

    if debug:
        print('new_b_array')
        print(new_b_array)

        print('new_b - old_b')
        print(np.linalg.norm(new_b_array - b_array ))

    #array record all objective values
    obj_record = [L_obj(P_Ball_mat = P_Ball_mat, P_Gr_mat = P_Gr_mat, cost_func = cost_func)]

    #info needed to be returned with the final array
    info = {'iter': epoch, 'obj': -1, 'b_array_path': [b_array]}

    #names of graphs if gif needs to be ploted
    filenames = []
    for iter in range(epoch):
        name =  'Optimization Iteration: ' + str(iter)
        #graphing
        if printing and iter % plot_per_n_iter == 0:
            if gif_output != None:

                plot_b_array(b_array, color_map= arrays_dict[c_array_name],
                            title= name, save = True ,
                            tail = video_tail, b_array_path = info['b_array_path'])
                filenames.append(name+'.png')
            else:
                plot_b_array(b_array, color_map= arrays_dict[c_array_name] )
        elif gif_output != None and (iter % plot_per_n_iter == 0 or iter < plot_all_before):
            plot_b_array(b_array, color_map= arrays_dict[c_array_name],
                            title= name, save = True, plot = False ,
                            tail = video_tail, b_array_path = info['b_array_path'])
            filenames.append(name+'.png')



        if printing:
            print(iter)

        dist_b_array_mat = dist_b_array(b_array = new_b_array, method = method)

        P_Ball_mat, support_mat = P_Ball(b_array = new_b_array,
                            method = method,
                            dist_b_array_mat = dist_b_array_mat,
                            cost_func = cost_func,
                            beta = beta)

        assert abs(np.sum(P_Ball_mat) - 1) < 1e-5


        del_L_array = del_L(b_array = b_array,
                        P_Ball_mat = P_Ball_mat,
                        P_Gr_mat = P_Gr_mat,
                        cost_func = cost_func,
                        dist_b_array_mat = dist_b_array_mat,
                        method = method,
                        beta = beta,
                        support_mat = support_mat)

        obj = L_obj(P_Ball_mat = P_Ball_mat,
                    P_Gr_mat = P_Gr_mat,
                    cost_func = cost_func)

        if obj > obj_record[-1]:
            eta /= 2
        if abs(obj - obj_record[-1]) < min_value_gain or eta < min_eta:
            info['iter'] = iter
            break
        else:
            b_array = new_b_array
            new_b_array = retraction(b_array, del_L_array, eta, method)
            obj_record.append(obj)

            if b_array_path:
                info['b_array_path'].append(b_array)

        if printing:
            print('Obj:', obj_record[-1])
            print('eta:', eta)

        if printing_update:
            string = "Iter: " + str(iter) + ', Obj: ' + str(obj_record[-1]) + ', eta: ' + str(eta) + '                  '
            if google_colab:
                out.update(IPython.display.Pretty(string))
            else:
                print(string, end = '\r')

    if printing_update:
        if google_colab:
            out.update(IPython.display.Pretty('Found the optimizer with ' + str(iter) + ' iterations!     '))
        else:
            print('Found the optimizer with ' + str(iter) + ' iterations!         ')

    print('Optimum Objective:',obj)
    info['obj'] = obj

    if obj_plot:
        plot_obj(obj_record)

    if gif_output != None:
        name = method + ': Last'
        plot_b_array(b_array, color_map= arrays_dict[c_array_name],
                        title= name, save = True, plot = False ,
                        tail = video_tail, b_array_path = info['b_array_path'])
        filenames.append(name+'.png')
        gif_plot(filenames, gif_output)

    return b_array, info

'''
After training, creating a folder with current date and time, move all generated graphs into the folder.
'''
def clean_up(folder_name, folder_path = ''):
    if folder_name == 'Time':
        folder_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_results'

    os.mkdir(folder_path + folder_name)
    path = folder_path + folder_name + '/'

    allfiles = os.listdir('./')
    for f in allfiles:
        if ('pdf' in f) or ('png' in f) or ('eps' in f) or ('gif' in f):
            shutil.move(f, path + f)


'''
Provide a estimate upper Bound
'''

'''
def upperBound(S, labels, beta = 1):
    labels = np.array(labels)
    t = len(S.shape) - 2
    if t > 1:
        raise Exception('Only a single time frame should be contained in S.')
    N = S.shape[0]
    K = max(labels) + 1
    nk = N // K

    D = N * (nk - 1) + N * (N - nk) * np.exp(-1 * np.arccosh(1 + 2*np.sin(np.pi/K)/(0.75**2) )**2 / beta)

    U_dist_mat = dist_U_array(S)
    d = 0
    c = np.inf
    ind_mat = np.zeros(U_dist_mat.shape)
    for i in range(K):
        sub_ind = labels == i

        for row in range(N):
            if sub_ind[row]:
                ind_mat[row, sub_ind] = 1

    for i in range(N):
        for j in range(N):
            if ind_mat[i,j] == 1:
                d = max(d, U_dist_mat[i,j])
            else:
                c = min(c, U_dist_mat[i,j])

    print('min_distance = ', c)
    gamma_array = np.zeros(N)
    for row in range(N):
        gamma_array[row] = np.std(U_dist_mat[row,:])

    print('max_std = ', max(gamma_array))
    c = c / np.sqrt(2) / max(gamma_array)


    P_Gr_mat = P_Gr(U_array = S, cost_func = 's-SNE')
    P_Gr_mat_nonzero = P_Gr_mat.copy()
    P_Gr_mat_nonzero[P_Gr_mat == 0] = 0.1

    bound = 0
    #bound += np.sum(P_Gr_mat * np.log(P_Gr_mat_nonzero))

    for i in range(N):
        for j in range(N):
            if ind_mat[i,j] == 1:
                #bound += P_Gr_mat[i,j]*np.log(D)
                #bound += np.exp(d) / (K * (nk - 1)) * np.log(D)
                continue
            else:
                #bound += P_Gr_mat[i,j]* ( 2.2**2/beta + np.log(D))
                #bound += P_Gr_mat[i,j]* ( 2.2**2/beta)
                #bound += np.exp(d-c**2) / (K * (nk - 1)) * ( 2.2**2/beta + np.log(D))
                continue


    bound += np.log(D)
    #bound += (N - nk) * np.exp(d-c**2) / (nk-1)*  (2.2**2)/beta
    bound += 2 * K * np.exp(d-c**2) *  (2.2**2)/beta

    print('d = ', d)
    print('c = ', c)
    print('c^2 = ', c**2)
    return bound, d - c**2
'''
