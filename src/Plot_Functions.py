import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import seaborn as sns
import IPython
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting



'''
Plot U_array in a 3-d Hemisphere.
'''
def plot_U_array(U_array, labels, title = None):
    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:0.5*pi:180j, 0.0:2.0*pi:720j] # phi = alti, theta = azi
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)
    #Set colours and render
    fig = plt.figure(figsize=(5, 3), dpi = 200)

    SMALL_SIZE = 10
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 20

    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)

    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        x, y, z,  rstride=4, cstride=4, color='w', alpha=0.1, linewidth=0)
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_zlim([0,1.0])


    plt.locator_params(axis='y', nbins=6)
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='z', nbins=6)
    #ax.plot_wireframe(x, y, z, color="k")

    cmap=plt.get_cmap("jet")
    c_array = [cmap(i / max(labels)) for i in labels]

    ax.scatter(U_array[:,0,0],U_array[:,1,0],U_array[:,2,0], c = c_array, s = 10)

    if title != None:
        plt.title(title, fontsize = 20)

    plt.savefig('U_array_init.pdf',bbox_inches='tight',pad_inches=0.5)
    plt.show()

'''
Combine all files under filenames and generate the gif.
'''
def gif_plot(filenames, gif_name, times_per_pic = 2, time_for_last = 30):
    # build gif
    with imageio.get_writer(gif_name, mode='I') as writer:
        for filename_id in range(len(filenames)):
            image = imageio.imread(filenames[filename_id] )

            if filename_id != len(filenames) - 1:
                for i in range(times_per_pic):
                    writer.append_data(image)
            else:
                for i in range(time_for_last):
                    writer.append_data(image)

    # Remove files
    for filename in set(filenames):
        os.remove(filename )

'''
Plot objective values.
'''
def plot_obj(obj_record):
    plt.figure()
    plt.plot(obj_record)
    plt.xlabel('Iteration')
    plt.ylabel('L')
    plt.title('Objective')
    plt.show()


def plot_b_array_path(b_array, #data
                labels, #color for each point
                path_length,
                paths_count,
                targets_count,
                video,
                path_names = [],
                title = None, #the title of the graph
                save = False, #save the graph if true
                plot = True, #show the graph if true
                format = 'png', #format of graph: png, pdf, eps
                tail = 0
                ) :

    x = b_array[:,0]
    y = b_array[:,1]
    SMALL_SIZE = 10
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 20

    fig = plt.figure(figsize=(4,4),dpi = 200)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)


    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)

    ax = plt.gca()
    circle2 = plt.Circle((0, 0), 1, color='b', fill=False)
    ax.add_patch(circle2)

    cmap=plt.get_cmap("jet")
    labels_normalized = labels / max(labels)

    for i in range(targets_count):
        plt.scatter(x[i],y[i], c = 'b', s = 20)

    for path_index in range(paths_count):

        index = targets_count + (path_index + 1) * path_length - 1
        plt.scatter(x[index], y[index], c = [cmap(labels_normalized[path_index])], s = 10)

        start = targets_count + path_index * path_length

        if len(path_names) > 0:
            plt.plot(x[start : index + 1],y[start : index + 1], linewidth=0.5, c = cmap(labels_normalized[path_index]), label = path_names[path_index])
            plt.legend()
        else:
            plt.plot(x[start : index + 1],y[start : index + 1], linewidth=0.5, c = cmap(labels_normalized[path_index]))

    if title:
        plt.title(title)

    plt.savefig('Grasscare.pdf',pad_inches=0, format = 'pdf')
    plt.close(fig)


    if video:
        filenames = []
        for step in range(path_length):

            name =  'Grasscare: ' + str(step)
            filenames.append(name+'.png')

            fig = plt.figure(figsize=(4,4),dpi = 200)
            plt.xlim(-1.1, 1.1)
            plt.ylim(-1.1, 1.1)


            plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
            plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
            plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
            plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
            plt.rc('figure', titlesize=BIGGER_SIZE)

            ax = plt.gca()
            circle2 = plt.Circle((0, 0), 1, color='b', fill=False)
            ax.add_patch(circle2)

            cmap=plt.get_cmap("jet")
            labels_normalized = labels / max(labels)

            for i in range(targets_count):
                plt.scatter(x[i],y[i], c = 'b', s = 20)

            for path_index in range(paths_count):

                index = step + targets_count + path_index * path_length
                plt.scatter(x[index], y[index], c = [cmap(labels_normalized[path_index])], s = 10)

                if tail > 0:
                    start = step + targets_count + path_index * path_length - tail
                if tail == -1:
                    start = targets_count + path_index * path_length

                if len(path_names) > 0:
                    plt.plot(x[start : index + 1],y[start : index + 1], linewidth=0.5, c = cmap(labels_normalized[path_index]), label = path_names[path_index])
                    plt.legend()
                else:
                    plt.plot(x[start : index + 1],y[start : index + 1], linewidth=0.5, c = cmap(labels_normalized[path_index]))


            if title:
                plt.title(name)

            plt.savefig(name, pad_inches=0)
            plt.close(fig)

        gif_plot(filenames, title + '.gif')


    return None


'''
Plot b_array on a 2-d disk.
'''
def plot_b_array(b_array, #data
                color_map = None, #color for each point
                zoom = False, #if zoom, no boundary of the circle will be plotted.
                title =None, #the title of the graph
                save = False, #save the graph if true
                plot = True, #show the graph if true
                format = 'png', #format of graph: png, pdf, eps
                GROUSE_mode = False, #plot GROUSE graphs
                v_count = -1, #parameter for GROUSE.
                geodesic_steps = 0, #paramter for GROUSE
                labels = ['0% missing','33% missing','66% missing' ], #labeling for GROUSE
                tail = 0,
                b_array_path = None) :


    #graph setting
    fig = plt.figure(figsize=(4,4),dpi = 200)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)

    SMALL_SIZE = 10
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 20

    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)

    ax = plt.gca()

    #preprocess b_array
    if b_array.shape[1] == 1:
        x = np.cos(b_array)
        y = np.sin(b_array)
    else:
        x = b_array[:,0]
        y = b_array[:,1]


    plt.scatter(x,y, c = color_map, s = 10)

    try:
        if tail != 0 and len(b_array_path) > 1:
            b_array_path = np.array(b_array_path)
            n_points = b_array_path.shape[1]
            steps = b_array_path.shape[0]
            assert b_array_path.shape[2] == 2

            for i in range(n_points):
                path = b_array_path[:, i, :].reshape((-1,2))

                if steps > tail and tail > 0:
                    plt.plot(path[-1 * tail - 1:,0],path[-1 * tail - 1:,1], linewidth=0.5, c = color_map[i])
                else:
                    plt.plot(path[:,0],path[:,1], linewidth=0.5, c = color_map[i])
    except TypeError:
        print('ERROR: Fail to plot the tails. \n Note: If tail is intended to be plotted,'
        + ' please make sure b_array_path is passed into Function plot_b_array.'
        + ' It can be found in info[\'b_array_path\'] returned from grasscare_train().')


    if not zoom:
        circle2 = plt.Circle((0, 0), 1, color='b', fill=False)
        ax.add_patch(circle2)

    if title:
        plt.title(title)

    if save and format == 'png':
        plt.savefig(title,pad_inches=0)
    elif save and format == 'eps':
        plt.savefig(title + '.eps',pad_inches=0,format = 'eps')
    elif save and format == 'pdf':
        plt.savefig(title + '.pdf',pad_inches=0,format = 'pdf')

    if plot:
        plt.show()

    plt.close(fig)
