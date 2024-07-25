import numpy as np
import matplotlib.pyplot as plt

def plotGraph(image, title = None):
    arr = np.asarray(image)
    
    mycmap = plt.cm.get_cmap('tab20')
    mycmap.set_under('w')
    
    plt.imshow(arr, cmap=mycmap, vmin=.1)
    if title is not None:
        plt.title(title)
    #plt.show()
    
def plotGraphs(image_a, image_b, title = None):
    fig = plt.figure(figsize=(16, 9))
    # Adds a subplot at the 1st position
    fig.add_subplot(1, 2, 1)
    
    # showing image
    plt.imshow(image_a)
    plt.axis('off')
    plt.title("First")
    
    # Adds a subplot at the 2nd position
    fig.add_subplot(1, 2, 2)
    
    # showing image
    plt.imshow(image_b)
    plt.axis('off')
    plt.title("Second")
    
    #plt.show()
    
def pointcloudOnImage(point_cloud, image, title='PC on Image', dot_size=5):
    # Init axes.
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    fig.canvas.set_window_title(title)
    ax.imshow(image)
    ax.scatter(point_cloud[0, :], point_cloud[1, :], c=point_cloud[2, :], s=dot_size)
    ax.axis('off')
    
    #plt.show()
