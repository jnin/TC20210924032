import os
import csv
import pandas as pd
import numpy as np
import tensorflow as tf

import functions as fn
import datasets as dt

import warnings
warnings.filterwarnings("ignore")

tf.keras.backend.set_floatx('float64')
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

##########################
## SUPPORTING FUNCTIONS ##
##########################
def generate_and_save_samples(original_model, size, run, d, path, factor=1.5):

    X_new = np.random.multivariate_normal(np.zeros((d,)), np.eye(d,d)*factor, size=size)
    y_new = original_model.predict(X_new)
    
    with open(os.path.join(path,'RUN_'+str(run)+'_data.npy'), 'wb') as f:
        np.save(f, X_new)
        np.save(f, y_new)
        
def generate_samples(original_model, size, d=2, factor=1.5):
    ''' 
    Generate new data points by samplin them for a multivariate 
    normal distribution and labelling them using the model
    
    original_model: the model used to generate the synthetic data labels   
    size: number of new data points    
    d: dimension of the input data    
    '''
    
    X_new = np.random.multivariate_normal(np.zeros((d,)), np.eye(d,d)*factor, size=size)
    y_new = original_model.predict(X_new)
    return X_new, y_new

def sample(d, n_classes, original, n_sampling, balancer=True):
    
    if balancer:
        X, y = rBalancer(N=0, d=d, K=n_classes, model=original, max_iter=10, N_batch=n_sampling, low=0,high=3*np.sqrt(d))
    else:
        X, y = generate_samples(original, n_sampling, d)
    
    return X, y

def update_synthetic_set(X, y, X_train, y_train, pure_online=False):
    
    if pure_online:  
        return X, y
    else:
        return np.vstack((X, X_train)), np.append(y, y_train)
    
def shifted_sample(d, original, n_sampling, iteration, mode='linear'):
    
    X, y = fn.generate_samples(original, n_sampling, d)
    test_size = 0.2
    random_state = 42
    #Split dataset into subsets that minimize the potential for bias in your evaluation and validation process.
    X_train, X_test, y_train, y_test = train_test_split(X, y.astype(int), test_size=test_size, random_state=random_state)

    print(len(X_train))
       
    #scaler = StandardScaler(copy=True)
    #scaler.fit(X_train)
    if mode=='linear':
        shift = [0.05*iteration,-0.05*iteration]
        for i,x in enumerate(X_train):
            X_train[i] = np.sum([x, shift], axis=0)

        for i,x in enumerate(X_test):
            X_test[i] = np.sum([x, shift], axis=0)
    else:
        ang = 15.0 #15 degs at each iteration
        shift = [[np.cos(ang*iteration*np.pi/180),-np.sin(ang*iteration*np.pi/180)],[np.sin(ang*iteration*np.pi/180),np.cos(ang*iteration*np.pi/180)]]
        for i,x in enumerate(X_train):
            X_train[i] = np.matmul(shift,x)
            
        for i,x in enumerate(X_test):
            X_test[i] = np.matmul(shift,x)
        
    return X_train, y_train, X_test, y_test

def rBalancer(N,d,K,model,max_iter=10,N_batch=100000, low=0, high=1):
    # N is the amount of samples per class required
    # max_iter is the max number of iterations to get to N
    # K is the number of classes in the problem
    # N_batch is the number of elements sampled at each iteration
    bins = np.arange(K+1)-0.5
    classes = np.arange(K)
    #Generate random direction
    v = np.random.multivariate_normal(np.zeros((d,)),np.eye(d,d),size = N_batch)
    v = v/np.linalg.norm(v,axis=1)[:,np.newaxis]
    #Scale the direction between low and high
    #print('Generating between => low: '+str(low)+' and high: '+str(high))
    alpha = np.random.uniform(low=low,high=high,size = N_batch)
    #alpha = np.random.exponential(scale=5.,size=N_batch)

    qsynth = np.dot(alpha[:,np.newaxis],np.ones((1,d)))*v
    y_synth = model.predict(qsynth)
    nSamplesPerClass=np.histogram(y_synth,bins=bins)[0]
    #print(np.unique(y_synth))
    #print(nSamplesPerClass)
    toAdd = classes[nSamplesPerClass<N]
    #print(toAdd)
    #print(nSamplesPerClass[toAdd])
    for i in range(max_iter):
        #Generate random direction
        v = np.random.multivariate_normal(np.zeros((d,)),np.eye(d,d),size = N_batch)
        v = v/np.linalg.norm(v,axis=1)[:,np.newaxis]
        #print('Generating between => low: '+str(low)+' and high: '+str(high))
        #alpha = np.random.exponential(scale=0.1,size=N_batch)
        alpha = np.random.uniform(low=low,high=high,size = N_batch)
        qtmp = np.dot(alpha[:,np.newaxis],np.ones((1,d)))*v
        y_synth = model.predict(qtmp)

        #Select samples to add
        idx = [i for i in range(N_batch) if y_synth[i] in toAdd]
        #Add samples to the synthetic set
        qsynth = np.r_[qsynth,qtmp[idx,:]]
        y_synth = model.predict(qsynth)

        nSamplesPerClass=np.histogram(y_synth,bins=bins)[0]
        toAdd = classes[nSamplesPerClass<N]
        #print('To ADD:' + str(i)+str(toAdd))
        #print(nSamplesPerClass[toAdd])
        if len(toAdd)<1:
            return qsynth,y_synth
    return qsynth,y_synth
