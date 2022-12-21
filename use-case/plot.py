import os
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import regularizers
from tensorflow.keras.losses import Loss
from tensorflow import keras

tf.keras.backend.set_floatx('float64')
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def plot_original_model(dataset,
                        model, 
                        X_train, 
                        y_train, 
                        path,
                        x_range=3.0):
    
    xx,yy = np.meshgrid(np.linspace(-x_range,x_range,200),np.linspace(-x_range,x_range,200))
    viz=np.c_[xx.ravel(),yy.ravel()]

    z = model.predict(viz)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,  alpha=0.7)
    plt.imshow(z.reshape((200,200)), origin='lower', extent=(-x_range,x_range,-x_range,x_range),alpha=0.3, vmin=0, vmax=1)
    plt.contour(xx,yy,z.reshape((200,200)),[0.5])

    plt.gcf().set_size_inches((6,6))
    plt.tick_params(
        axis='both',          # changes apply to both axes
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,         # ticks along the top edge are off
        right=False,         # ticks along the top edge are off
        labelbottom=False, # labels along the bottom edge are off
        labelleft=False) # labels along the bottom edge are off
    
    plt.savefig(os.path.join(path, 'data', dataset, 'original_model.pdf'), bbox_inches='tight')
    plt.close()
    
def plot_copy_model(copy,
                    X_train,
                    y_train,
                    X_errors,
                    y_errors,
                    t,
                    run,
                    path,
                    x_range=3.0):

    xx,yy = np.meshgrid(np.linspace(-x_range,x_range,200),np.linspace(-x_range,x_range,200))
    viz=np.c_[xx.ravel(),yy.ravel()]
    
    z = np.argmax(copy.predict(viz), axis=1)
    
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,  alpha=0.7)
    plt.scatter(X_errors[:, 0], X_errors[:, 1], c='red', marker='^', alpha=0.2)
    plt.imshow(z.reshape((200,200)), origin='lower', extent=(-x_range,x_range,-x_range,x_range),alpha=0.3, vmin=0, vmax=1)
    plt.contour(xx,yy,z.reshape((200,200)),[0.5])
    
    plt.xlim(-x_range,x_range)
    plt.ylim(-x_range,x_range)
    plt.autoscale(False)
    plt.gcf().set_size_inches((6,6))
    
    plt.savefig(os.path.join(path, 'plots', f'RUN_{run}_ITER_{t}.pdf'), bbox_inches='tight')
    plt.close()