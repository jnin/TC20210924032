import os
import csv
import pandas as pd
from sklearn import datasets
from collections import Counter
import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from tensorflow.keras import regularizers
from tensorflow.keras.losses import Loss
from tensorflow import keras

from keras.losses import LossFunctionWrapper
from keras.utils import losses_utils

import warnings
warnings.filterwarnings("ignore")

tf.keras.backend.set_floatx('float64')
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#################################
## SUPPORTING FUNCTIONS (MAIN) ##
#################################

def get_params_dict(model_path):
    with open(model_path+'/params.txt', 'r') as handle:
        data = handle.read()

    dic={}
    for row in data.split('\n')[:-1]:
        index, value = row.split('=')
        try:
            dic[index.strip()]=float(value)
        except:
            dic[index.strip()]=value

    return dic

##########################
## SUPPORTING FUNCTIONS ##
##########################

def params_to_vec(model,return_dims = False):
    ## get the all model parameters
    final = []
    dims = 0
    for layer in model.layers:
        t0 = tf.reshape(layer.trainable_variables[0], [np.shape(layer.trainable_variables[0])[0]*np.shape(layer.trainable_variables[0])[1]])
        t1 = tf.reshape(layer.trainable_variables[1], [np.shape(layer.trainable_variables[1])[0]])
        final.append(tf.concat([t0,t1],0))
        dims+= np.shape(t0)[0]+np.shape(t1)[0]
    
    if return_dims:
        return final,dims
    else:
        return final

def norm_theta(vec1,vec2):    
    ## given two parameter's vectors, return the Euclidean norm
    t = tf.constant(0.0, dtype = tf.float64)
    for i in range(np.shape(vec1)[0]):
        t = tf.add(t,tf.reduce_sum(tf.square(vec1[i] - vec2[i])))
    return tf.cast(t, tf.float64)
    
def evaluation(model, X, y,d,n_classes):
    ## evaluate the model "accuracy"
    try:
        y_pred_ohe = model.predict(X, verbose=0)
        y_pred = np.argmax(y_pred_ohe, axis=1)
    except:
        y_pred = model.predict(X)
        
    try:
        if np.shape(y)[1]==n_classes:
            y = np.argmax(y, axis=1)
    except:
        pass

    return sum(y_pred == y)/len(X)

#########################
## LOSS FUNCTION CLASS ##
#########################

#class my_loss(Loss):
#    # initialize instance attributes
#    def __init__(self, n_classes):
#        super(my_loss, self).__init__()
#        self.n_classes = tf.constant(n_classes, dtype = tf.float64)
#        self.rho_max = tf.constant(1.0, dtype = tf.float64)
#        self.size = None #called from the custum_model train step

#    # Compute loss
#    def call(self, y, y_pred):
#        y = tf.cast(y, tf.float64)
#        y_pred = tf.cast(y_pred, tf.float64)
#        #rho = tf.math.divide(self.rho_calculation(y, y_pred), self.rho_max)
#        rho = tf.math.divide(rho_mean_loss(y, y_pred, self.n_classes, self.size), self.rho_max)
#        return rho
    
#    def norm_calculation(self,vec1,vec2):    
#        ## given two parameter's vectors, return the Euclidean norm
#        t = tf.constant(0.0, dtype = tf.float64)
#        for i in range(np.shape(vec1)[0]):
#            t = tf.add(t,tf.reduce_sum(tf.square(vec1[i] - vec2[i])))
#        return tf.cast(t, tf.float64)
    
#    #def rho_calculation(self, y, y_pred):
#    #    try:
#    #        rho = tf.reduce_sum(tf.divide(tf.reduce_sum(tf.square(tf.subtract(y,y_pred)), 1, keepdims=True), self.n_classes))
#    #        return tf.divide(rho, tf.constant(self.size, dtype = tf.float64))
#    #    except:
#    #        raise NameError("El problema está en las dims de y={}, y_pred={}".format(np.shape(y),np.shape(y_pred)))
#    #        return tf.constant(0.0, dtype = tf.float64)
                      
class UncertaintyError(LossFunctionWrapper):
    """Computes the norm between hard labels and soft predictions.
    """

    def __init__(
        self, reduction=losses_utils.ReductionV2.AUTO, name="uncertainty_error"
    ):
        """Initializes `UncertaintyError` instance.
        Args:
          reduction: Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used with
            `tf.distribute.Strategy`, outside of built-in training loops such as
            `tf.keras` `compile` and `fit`, using `AUTO` or
            `SUM_OVER_BATCH_SIZE` will raise an error. Please see this custom
            training [tutorial](
            https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
          name: Optional name for the instance. Defaults to
            'uncertainty_error'.
        """
        super().__init__(uncertainty_error, name=name, reduction=reduction)
    

def uncertainty_error(y_true, y_pred):
    """
    """
    y_pred = tf.convert_to_tensor(y_pred)# ESTO PUEDE DAR PROBLEMAS
    y_true = tf.cast(y_true, y_pred.dtype)# ESTO PUEDE DAR PROBLEMAS
    num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)

    return tf.divide(tf.reduce_sum(tf.math.squared_difference(y_pred, y_true),1, keepdims=True), num_classes)

##########################
## REGULARIZATION CLASS ##
##########################

class MyRegularizer(regularizers.Regularizer):

    def __init__(self, layer_num, model):
        self.layer_num = layer_num
        self.model = model

    def __call__(self, x):
        self.new_weigths = self.get_weigths(self.model.layers[self.layer_num])
        self.theta0 = self.model.theta0[self.layer_num]
        return tf.divide(tf.reduce_sum(tf.square(self.theta0-self.new_weigths)),
                         tf.constant(self.model.weights_dims, dtype = tf.float64))
    
    def get_weigths(self, layer):
        #weights of the layer
        t0 = tf.reshape(layer.trainable_variables[0], [np.shape(layer.trainable_variables[0])[0]*np.shape(layer.trainable_variables[0])[1]])
        #bias of the layer
        t1 = tf.reshape(layer.trainable_variables[1], [np.shape(layer.trainable_variables[1])[0]])
        return tf.concat([t0,t1],0)
    
    
#################
## MODEL CLASS ##
#################

class CustomCopyModel(keras.Model):
    def __init__(self,
                 input_dim,
                 hidden_layers,
                 output_dim,
                 activation='relu',
                 seed=42):
        
        super(CustomCopyModel, self).__init__()
        
        tf.random.set_seed(seed)
        
        self.dense = []
        
        if len(hidden_layers) > 0:
            self.dense.append(tf.keras.layers.Dense(hidden_layers[0], 
                                                  input_shape=(input_dim,), 
                                                  activation=activation,
                                                  kernel_initializer='he_normal', 
                                                  kernel_regularizer=MyRegularizer(layer_num=0, model=self),  
                                                  name='layer0', autocast=False))
            if len(hidden_layers) > 1:
                for layer, n_neurons in enumerate(hidden_layers[1:]):
                    self.dense.append(tf.keras.layers.Dense(n_neurons, 
                                                            activation=activation, 
                                                            kernel_initializer='he_normal',
                                                            kernel_regularizer=MyRegularizer(layer_num=layer+1, model=self),
                                                            name='layer{}'.format(layer+1)))
        self.dense.append(tf.keras.layers.Dense(output_dim, 
                                                activation='softmax',
                                                kernel_regularizer=MyRegularizer(layer_num=len(hidden_layers), model=self),
                                                name='layer{}'.format(len(hidden_layers))))
        self.weights_dims = None
    
    def call(self, inputs):
        x = self.layers[0](inputs)
        if len(self.layers) > 1:
            for layer in range(1,len(self.layers)):
                x = self.layers[layer](x)
        return x
    
    def fit(self, x, y, theta0, lmda, acc, rho_max, size, epochs, batch, verbose, callbacks):
        self.lmda = lmda
        self.theta0 = theta0
        #self.size = size
        #self.loss.rho_max = rho_max
        self.rho_max = rho_max
        self.acc = acc
        return super(CustomCopyModel,self).fit(x,y, epochs=epochs, batch_size=batch, verbose=0, callbacks=callbacks)
    
    def compile(self, optimizer, loss, acc_threshold):
        super(CustomCopyModel, self).compile()
        self.optimizer = optimizer
        self.acc_threshold = acc_threshold
        self.loss = loss 
        
    def train_step(self, data):
        x, y = data
        #self.loss.size = self.size
                
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            regularization_term = tf.tanh(tf.math.reduce_sum(self.losses))
            
            # AQUÍ HAY QUE NORMALIZAR LA LOSS USANDO EL RHO_MAX
            # Compute the loss value (the loss function is configured in `compile()`)
            loss_raw = self.loss(y, y_pred)/self.rho_max
            loss = loss_raw + self.lmda*regularization_term
            #loss = self.loss(y, y_pred) + self.lmda*regularization_term
            
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        
        # Return a dict mapping metric names to current value
        return {'loss' :loss,
                'reg' : tf.math.reduce_sum(self.losses),
                'rho' : loss_raw}

    
################################
## TRAIN SUPPORTING FUNCTIONS ##
################################


#def rho_calculation_loss(y, y_pred, n_classes):
#    '''
#    Compute value of rho as the quadratic distance between
#    true labels and probabilistic outputs
#    '''
#    return tf.divide(tf.reduce_sum(tf.square(tf.subtract(y, y_pred)),1, keepdims=True), n_classes)#

#def rho_mean_loss(y, y_pred, n_classes, size):
#    rho = rho_calculation_loss(y, y_pred, n_classes)
#    return tf.divide(rho, tf.constant(size, dtype = tf.float64))#

    
#def rho_calculation(copy_model, X_train, y_train, d, n_classes):
#    '''
#    Compute value of rho as the quadratic distance between
#    true labels and probabilistic outputs
#    '''
#    y_pred_ohe = copy_model.predict(X_train)
#    y_train_ohe = tf.one_hot(y_train, n_classes)
#    #rho = np.sqrt(tf.math.reduce_sum((y_train_ohe - y_pred_ohe)**2, axis=1))
#    rho = tf.cast(tf.reduce_sum(tf.divide(tf.reduce_sum(tf.square(tf.subtract(y_train_ohe,y_pred_ohe)), 1, keepdims=True), n_classes)), tf.float64)
#    return tf.divide(rho, tf.constant(len(y_train_ohe), dtype = tf.float64)).numpy()

#def rho_weights(copy_model, X_train, y_train, d, n_classes):
#    '''
##    Compute value of rho as the quadratic distance between
#    true labels and probabilistic outputs, returns vector for weights
#    '''
##    y_pred_ohe = copy_model.predict(X_train)
#    y_train_ohe = tf.one_hot(y_train, n_classes)
#    rho = tf.divide(tf.reduce_sum(tf.square(tf.subtract(y_train_ohe,y_pred_ohe)), 1, keepdims=True), n_classes)
#    return np.asarray(rho)
#'''


def remove_point_rho(copy_model, X, y, threshold, d, n_classes, loss):
    X_train = np.empty((0, d))
    y_train = np.empty((0), dtype=int)
    
    y_pred = copy_model.predict(X, verbose=0)
    #y_train_ohe = tf.one_hot(y, n_classes)
    
    #loss = loss_function(reduction=tf.keras.losses.Reduction.NONE)
    rho = loss(tf.one_hot(y, n_classes), y_pred).numpy()
    
    #rho = tf.divide(tf.reduce_sum(tf.square(tf.subtract(y_train_ohe,y_pred_ohe)),1, keepdims=True),n_classes).numpy()
    for i, r in enumerate(rho>=threshold):
        if r:
            y_train = np.append(y_train,y[i])
            X_train = np.append(X_train,[X[i]], axis=0)
    
    if len(X_train)>0:
        return X_train, y_train
    else:
        try:
            nN = np.random.randint(0,len(X),int(len(X)/2))
            for n in nN:
                y_train = np.append(y_train,y[n])
                X_train = np.append(X_train,[X[n]], axis=0)
            return X_train, y_train
        except:
            return X, y