import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import random
import os

def create_circle(loc,h,color):
    circle = mpatches.Circle(loc, h/2, color=color)
    return circle

def create_rectangle(loc,h,color):
    rect = mpatches.Rectangle(loc, h, h*1.62, color=color)
    return rect

def create_polygon(loc,h,color):
    polygon = mpatches.RegularPolygon(loc, 5, h/2, color=color)
    return polygon

def create_ellipse(loc,w,color):
    ellipse = mpatches.Ellipse(loc, w, w*0.8, color=color)
    return ellipse

def create_wedge(loc,h,color):
    wedge = mpatches.Wedge(loc, h/2, random.randint(0,90), random.randint(180,300), color=color)
    return wedge

shapes = [create_circle,create_rectangle,create_polygon,create_ellipse,create_wedge]

dataset=[]
DATASET_WIDTH = 20
IMAGE_WIDTH = 256
plt.rcParams['figure.facecolor'] = 'black'
DIR_PATH = './data'

for i in range(DATASET_WIDTH):
    nb_shapes = random.randint(3,5)
    patches = []

    ax = plt.gca()
    ax.set_xlim(0,IMAGE_WIDTH)
    ax.set_ylim(0,IMAGE_WIDTH)
    plt.axis('off')

    for j in range (nb_shapes):
        loc = [random.randint(int(round(IMAGE_WIDTH/8)),int(round(IMAGE_WIDTH*7/8))),random.randint(int(round(IMAGE_WIDTH/8)),int(round(IMAGE_WIDTH*7/8)))]
        height = (1+random.random())*IMAGE_WIDTH/8
        color = str(np.random.randint(70,255)/255)
        shape = random.choice(shapes)(loc,height,color)
        ax.add_patch(shape)
    file_name = "img{:d}.png".format(i)
    file_path = os.path.join(DIR_PATH,file_name)
    plt.savefig(file_path,dpi=100, facecolor='black', bbox_inches = 'tight', pad_inches = 0.)
    #print(nb_shapes)
    #plt.show()
    plt.clf()

