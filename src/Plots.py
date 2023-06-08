import numpy as np
import matplotlib.pyplot as plt
import os
import imageio


def plot_color_bar(classes, color_scheme='jet'):

    labels = [i for i in range(len(classes))]

    cmap = plt.get_cmap(color_scheme)
    colors = [cmap(j / max(labels)) for j in labels]

    num_groups = 10
    group_size = len(classes) // num_groups

    fig, axs = plt.subplots(num_groups, 1, figsize=(16, num_groups*0.75))

    for i in range(num_groups):
        start_index = i * group_size
        end_index = (i + 1) * group_size if i < num_groups - 1 else len(classes)

        group_classes = classes[start_index:end_index]
        group_colors = colors[start_index:end_index]

        axs[i].bar(range(len(group_classes)), height=1, color=group_colors)
        axs[i].set_xticks(range(len(group_classes)))
        axs[i].set_xticklabels(group_classes, fontsize = 8)
        axs[i].set_xlim([-0.5, len(group_classes) - 0.5])
        axs[i].set_yticks([])
        axs[i].set_yticklabels([])

    plt.tight_layout()
    plt.show()





def plot_embedding(training_history, labels,
                    save = True,
                    name = 'Embedding',
                    color_scheme = "jet",
                    scatter_size = 20):
    #Color Map For the plot
    cmap=plt.get_cmap(color_scheme)
    c_array = [cmap(i / max(labels)) for i in labels]

    #Drawing Unit Circle
    angles = np.linspace(0, 2*np.pi, 100)
    x = np.cos(angles)
    y = np.sin(angles)


    embedding = training_history[-1]['b_array']

    #plotting
    plt.gca().set_aspect('equal')
    plt.scatter(embedding[:,0], embedding[:, 1], c = c_array, s = scatter_size)
    plt.xlim(-1.1,1.1)
    plt.ylim(-1.1,1.1)
    # Plot the circle
    plt.plot(x, y)
    plt.axis('off')

    if save:
        plt.savefig(name + '.pdf', format = 'pdf')

    plt.show()


def plot_process(training_history, labels, save = True):

    #Color Map For the plot
    cmap=plt.get_cmap("jet")
    c_array = [cmap(i / max(labels)) for i in labels]

    #Start Plotting
    num_subplot = 6
    fig, axs = plt.subplots(1, num_subplot, figsize = (30,5), dpi = 100)

    #Drawing Unit Circle
    angles = np.linspace(0, 2*np.pi, 100)
    x = np.cos(angles)
    y = np.sin(angles)

    #For each subplot
    for i, ax in enumerate(axs.flat):

        #retrieve the step image
        index = min(
                    int( len(training_history) / num_subplot * (i+1)),
                    len(training_history)-1
                    )

        embedding = training_history[index]['b_array']

        #plotting
        ax.set_aspect('equal')
        ax.set_title(index)
        ax.scatter(embedding[:,0], embedding[:, 1], c = c_array)
        ax.set_xlim(-1.1,1.1)
        ax.set_ylim(-1.1,1.1)
        # Plot the circle
        ax.plot(x, y)
        ax.axis('off')

    if save:
        plt.savefig('Process.pdf', format = 'pdf')

    plt.show()


'''
Combine all files under filenames and generate the gif.
'''
def gif_plot(filenames, gif_name, times_per_pic = 1, time_for_last = 30):
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



def plot_video(training_history, labels, cmap_name = 'jet', name = 'Video'):

    #Color Map For the plot
    cmap=plt.get_cmap(cmap_name)
    c_array = [cmap(i / max(labels)) for i in labels]


    #Drawing Unit Circle
    angles = np.linspace(0, 2*np.pi, 100)
    x = np.cos(angles)
    y = np.sin(angles)

    filenames = []

    for i,h in enumerate(training_history):
        plt.figure(figsize = (8,8), dpi = 100)


        # plot circle
        plt.plot(x, y)

        #plot the tail
        if i > 0:
            x_path = np.array([training_history[j]['b_array'][:,0] for j in range(i)])
            y_path = np.array([training_history[j]['b_array'][:,1] for j in range(i)])

            for pt in range( h['b_array'].shape[0]):
                plt.plot( x_path[:, pt], y_path[:,pt], c = c_array[pt] , alpha=0.03)



        # plot parameter
        plt.xlim(-1.1,1.1)
        plt.ylim(-1.1,1.1)
        plt.gca().set_aspect('equal')
        plt.axis('off')
        plt.title('Opt Iter: '+ str(h['iter']))


        # plot points
        plt.scatter(h['b_array'][:,0], h['b_array'][:,1], c = c_array)


        # plot save
        plt.savefig( str(h['iter']), bbox_inches='tight')

        # record filename
        filenames.append( str(h['iter']) + '.png' )
        plt.close()

    gif_plot(filenames, gif_name = name + '.gif')



def plot_3d(U_array, labels):
    cmap=plt.get_cmap("rainbow")
    c_array = [cmap(i / max(labels)) for i in labels]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot the points
    ax.scatter(U_array[:, 0, 0], U_array[:, 1, 0], U_array[:, 2, 0], c = c_array)

    radius = 0.98
    theta, phi = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    ax.plot_surface(x, y, z, alpha = 0.2)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)

    # Set the aspect ratio to 'equal'
    ax.set_box_aspect([1, 1, 1])



    plt.show()
