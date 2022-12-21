import os
import csv
import logging
import time as tm
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from tensorflow.keras.losses import Loss
from tensorflow import keras

import functions as fn
import datasets as dt
import sampling as sp
import plot as pl

import warnings
warnings.filterwarnings("ignore")

tf.keras.backend.set_floatx('float64')
logging.getLogger('tensorflow').setLevel(logging.FATAL)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
def define_loss(loss_name,d):
    
    if loss_name == 'self': 
        return fn.UncertaintyError(), fn.UncertaintyError
    elif d <= 2:
        return tf.keras.losses.BinaryCrossentropy(from_logits=True), tf.keras.losses.BinaryCrossentropy
    else:
        return tf.keras.losses.CategoricalCrossentropy(from_logits=True), tf.keras.losses.CategoricalCrossentropy

def create_copy_model(d, 
                      n_classes,
                      lr,
                      optimizer='Adam',
                      acc_thresh=None,
                      input_shape=64,
                      hidden_layers=[64, 32, 10],
                      loss_name='self',
                      activation='relu'):
    
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    loss, loss_function = define_loss(loss_name, d)   
    copy_model = fn.CustomCopyModel(input_dim=d, 
                                    hidden_layers=hidden_layers,
                                    output_dim=n_classes,
                                    activation=activation)
    
    copy_model.build(input_shape=(input_shape,d))
    copy_model.compile(optimizer=opt, 
                       loss=loss, 
                       acc_threshold=acc_thresh)
  
    return copy_model, opt, loss_function

def train_copy_model(copy_model, 
                     X_train, 
                     y_train, 
                     d, 
                     n_classes, 
                     opt, 
                     loss,
                     theta_prev, 
                     sample_weights, 
                     lmda, 
                     acc,
                     acc_thresh, 
                     rho_max,
                     n_epochs, 
                     batch_size,
                     callbacks):
    
    y_train_ohe = tf.one_hot(y_train, n_classes) 
    copy_model.fit(X_train, 
                       y_train_ohe, 
                       theta_prev, 
                       lmda, 
                       acc,
                       rho_max, 
                       size=y_train_ohe.shape[0], 
                       epochs=n_epochs, 
                       batch=batch_size, 
                       verbose=0, 
                       callbacks=callbacks)
    
    # update the rho_norm
    rho_max = loss(tf.one_hot(y_train, n_classes), copy_model.predict(X_train, verbose=0)).numpy()
    #rho_max = tf.constant(fn.rho_calculation(copy_model,X_train,y_train,d,n_classes), dtype = tf.float64)

    return copy_model, rho_max

def save_to_csv(data, path, file):
        
    with open(os.path.join(path, file), 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(np.asarray([data], dtype=object))

def single_train(d, 
                 n_classes,
                 lr_, 
                 size,
                 X_test,
                 y_test,
                 n_epochs=1000, 
                 batch_size=32,
                 dataset = '',
                 path='',
                 run=0, 
                 loss = 'self',
                 optimizer='adam',
                 hidden_layers = [64,32,10],
                 activation='relu',
                 callbacks = False,
                 verbose=False):
    
    with open(os.path.join('../sampling/'+dataset+'/RUN_'+str(run)+'_data.npy'), 'rb') as f:
        X_train = np.load(f)
        y_train = np.load(f)
              
    for n in range(0, 5050, size):
              
        # Copy
        print("{} dataset: {}, RUN={}, n={}...".format(tm.strftime("%y%m%d-%H:%M:%S", tm.gmtime()), dataset, run, n+size))


        copy_model_single, opt_single, loss_function_single = create_copy_model(d, 
                                                               n_classes, 
                                                               lr_, 
                                                               optimizer=optimizer,
                                                               acc_thresh=1,
                                                               input_shape=batch_size, 
                                                               hidden_layers=hidden_layers,
                                                               loss_name=loss,
                                                               activation=activation)

        theta_prev, copy_model_single.weights_dims = fn.params_to_vec(copy_model_single, return_dims = True)
              
        X_train_single = X_train[:n+size]
        y_train_single = y_train[:n+size]

        copy_model_single, rho_max_single = train_copy_model(copy_model_single, 
                                                             X_train_single, 
                                                             y_train_single, 
                                                             d, 
                                                             n_classes, 
                                                             opt_single, 
                                                             loss_function_single(reduction=tf.keras.losses.Reduction.AUTO), 
                                                             theta_prev, None, 0.0, 1.0, 0.0, 1.0, n_epochs, batch_size,[])

        # Compute copy accuracy
        acc_train_single = fn.evaluation(copy_model_single, X_train_single,y_train_single,d,n_classes)
        acc_test_single = fn.evaluation(copy_model_single,X_test,y_test,d,n_classes)

        save_to_csv(acc_train_single, path, f'RUN_{run}_acc_train_single.csv')
        save_to_csv(acc_test_single, path, f'RUN_{run}_acc_test_single.csv')
        
        loss = loss_function_single(reduction=tf.keras.losses.Reduction.AUTO)
        rho_mean = loss(tf.one_hot(y_train_single, n_classes), copy_model_single.predict(X_train_single, verbose=0)).numpy()
        
        #rho_mean = fn.rho_calculation(copy_model,X_train,y_train,d,n_classes)
        save_to_csv(rho_mean, path, f'RUN_{run}_rhoMatrix_single.csv')

def sequential_train(d, 
                      n_classes,
                      lr_, 
                      n_sampling, 
                      original, 
                      X_test, 
                      y_test,
                      plot=True, 
                      max_iter=15, 
                      n_epochs=1000, 
                      batch_size=32,
                      max_subtraining=2, 
                      deleting=False, 
                      thresh = 0.001, 
                      path='',
                      dataset_name='',
                      run=0, 
                      sample_weights=False,
                      lmda = 0.0,
                      acc_thresh=0.85,
                      balancer = False,
                      loss = 'self',
                      optimizer='adam',
                      hidden_layers = [64,32,10],
                      activation='relu',
                      pure_online = False,
                      shiftting = False,
                      callbacks = False,
                      verbose=False):
  
    #if verbose:
    #    print("{} starting run {}...".format(tm.strftime("%y%m%d-%H:%M:%S", tm.gmtime()), run))
    
    # Initiliaze run
    t=0   
    n_subtraining = 0
    X_train_uniques = np.empty((0, d))
    X_train = np.empty((0, d))
    y_train = np.empty((0), dtype=int)
    
    if run>=50:
        print('Ey! we just have 50 different sampling files! so Im reusing one of them :)')
        run_=run//50
    else:
        run_=run
    
    with open(os.path.join('../sampling/'+dataset_name+'/RUN_'+str(run_)+'_data.npy'), 'rb') as f:
        X_new_full = np.load(f)
        y_new_full = np.load(f)

    rho_max = tf.constant(1.0, dtype = tf.float64)
    
    acc_train = 0
    acc_test_vec = []
    
    # Define copy model
    copy_model, opt, loss_function = create_copy_model(d, 
                                                       n_classes, 
                                                       lr_, 
                                                       optimizer=optimizer,
                                                       acc_thresh=acc_thresh,
                                                       input_shape=batch_size, 
                                                       hidden_layers=hidden_layers,
                                                       loss_name=loss,
                                                       activation=activation)
    
    
    while t < max_iter: 
        
        # create new synthetic data points
        #X_new, y_new = sp.sample(d, n_classes, original, n_sampling, balancer)
        

        X_new = X_new_full[n_sampling*t:n_sampling*(t+1)]
        y_new = y_new_full[n_sampling*t:n_sampling*(t+1)]
        
        X_train, y_train = sp.update_synthetic_set(X_new, y_new, X_train, y_train, pure_online)
        
        nN_prev = len(X_train)
        
        if deleting and t > 0:
            X_train, y_train = fn.remove_point_rho(copy_model, 
                                                   X_train, 
                                                   y_train, 
                                                   thresh, 
                                                   d,
                                                   n_classes,
                                                   loss_function(reduction=tf.keras.losses.Reduction.NONE))
        
        nN = len(X_train)
                
        
        if (nN_prev - nN)<n_sampling:
            lmda = lmda/2
            #lmda = lmda
        elif (nN_prev - nN)>=n_sampling: 
            lmda = lmda*1.5
            #lmda = lmda
        
        save_to_csv(lmda, path, f'RUN_{run}_lmda.csv')     
        
        X_train_uniques = np.vstack((X_train_uniques, X_train))
        X_train_uniques = np.unique(X_train_uniques, axis=0)
        
        save_to_csv(len(X_train_uniques), path, f'RUN_{run}_unique_points.csv')
        
        y_errors = y_train        
        
        loss = loss_function(reduction=tf.keras.losses.Reduction.AUTO)
        rho_mean = loss(tf.one_hot(y_train, n_classes), copy_model.predict(X_train, verbose=0)).numpy()
        
        #rho_mean = fn.rho_calculation(copy_model,X_train,y_train,d,n_classes)
        save_to_csv(rho_mean, path, f'RUN_{run}_rhoMatrix.csv')
        save_to_csv(nN, path, f'RUN_{run}_nN.csv')  
        
        theta_prev, copy_model.weights_dims = fn.params_to_vec(copy_model, return_dims = True) 
        
        while len(y_errors) !=0 and n_subtraining <= max_subtraining:
            
            if n_subtraining > 0:
                if verbose:
                    print("{} RUN {} ITER {} entering subtraining {}...".format(tm.strftime("%y%m%d-%H:%M:%S", tm.gmtime()), run, t, n_subtraining-1))
                
                # shuffle data
                idx = np.random.permutation(len(X_train))
                X_train, y_train = X_train[idx], y_train[idx]
                
                # update learning rate
                opt.lr=lr_*(n_subtraining-1+1)/(n_subtraining-1+0.5)

            #rho_mean = fn.rho_calculation(copy_model,X_train,y_train,d,n_classes)
            
            # Create a callback that saves the model's weights
            if callbacks:
                cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=0)
            else:
                cp_callback = []
            copy_model, rho_max = train_copy_model(copy_model, X_train, y_train, d, n_classes, opt, loss_function(reduction=tf.keras.losses.Reduction.AUTO), theta_prev, sample_weights, lmda, acc_train, acc_thresh, rho_max, n_epochs, batch_size,[cp_callback])
            n_subtraining +=1
            
            # Identify errors
            y_pred_ohe = copy_model.predict(X_train, verbose=0)
            y_pred = np.argmax(y_pred_ohe, axis=1)
            X_errors = X_train[y_pred!=y_train,:]
            y_errors = y_train[y_pred!=y_train]

            # Compute copy accuracy
            acc_train = fn.evaluation(copy_model, X_train,y_train,d,n_classes)
            acc_test = fn.evaluation(copy_model,X_test,y_test,d,n_classes)
            
            
            if verbose:
                print("{} RUN {} ITER {} N {} ERRORS {} acc train: {:.2f}, acc test: {:.2f}, lambda: {:.5f}".format(tm.strftime("%y%m%d-%H:%M:%S", tm.gmtime()), run, t, len(X_train), len(X_errors), acc_train, acc_test, lmda))
        
        save_to_csv(acc_train, path, f'RUN_{run}_acc_train.csv')
        acc_test_vec.append(acc_test)
        save_to_csv(acc_test, path, f'RUN_{run}_acc_test.csv')
        
        theta_last = fn.params_to_vec(copy_model)
        last_norm = fn.norm_theta(theta_prev,theta_last)
        save_to_csv(last_norm.numpy(), path, f'RUN_{run}_thetas.csv')
        
        '''if deleting and t > 0:
            X_train, y_train = fn.remove_point_rho(copy_model, 
                                                   X_train, 
                                                   y_train, 
                                                   thresh, 
                                                   d,
                                                   n_classes,
                                                   loss_function(reduction=tf.keras.losses.Reduction.NONE))
        '''
        # Plot resulting model every 5 iterations
        if plot: #and t%5==0:   
            pl.plot_copy_model(copy_model, X_train, y_train, X_errors, y_errors, t, run, path, x_range=3.0)
        
        # reset values
        opt.lr = lr_
        n_subtraining = 0
        
        t += 1
        
def sequential_train_regression(d, 
                      n_classes,
                      lr_, 
                      n_sampling, 
                      original, 
                      X_test, 
                      y_test,
                      plot=True, 
                      max_iter=15, 
                      n_epochs=1000, 
                      batch_size=32,
                      max_subtraining=2, 
                      deleting=False, 
                      thresh = 0.001, 
                      path='',
                      dataset_name='',
                      run=0, 
                      sample_weights=False,
                      lmda = 0.0,
                      acc_thresh=0.85,
                      balancer = False,
                      loss = 'self',
                      optimizer='adam',
                      hidden_layers = [64,32,10],
                      activation='relu',
                      pure_online = False,
                      shiftting = False,
                      callbacks = False,
                      verbose=False):
  
    #if verbose:
    #    print("{} starting run {}...".format(tm.strftime("%y%m%d-%H:%M:%S", tm.gmtime()), run))
    
    # Initiliaze run
    t=0   
    n_subtraining = 0
    X_train_uniques = np.empty((0, d))
    X_train = np.empty((0, d))
    y_train = np.empty((0), dtype=int)
    
    if run>=50:
        print('Ey! we just have 50 different sampling files! so Im reusing one of them :)')
        run_=run//50
    else:
        run_=run
    
    with open(os.path.join('../sampling/'+dataset_name+'/RUN_'+str(run_)+'_data.npy'), 'rb') as f:
        X_new_full = np.load(f)
        y_new_full = np.load(f)
    
    rho_max = tf.constant(1.0, dtype = tf.float64)
    
    acc_train = 0
    acc_test_vec = []
    
    # Define copy model
    copy_model, opt, loss_function = create_copy_model(d, 
                                                       n_classes, 
                                                       lr_, 
                                                       optimizer=optimizer,
                                                       acc_thresh=acc_thresh,
                                                       input_shape=batch_size, 
                                                       hidden_layers=hidden_layers,
                                                       loss_name=loss,
                                                       activation=activation)
    
    
    while t < max_iter: 
        
        # create new synthetic data points
        #X_new, y_new = sp.sample(d, n_classes, original, n_sampling, balancer)
        
        X_new = X_new_full[n_sampling*t:n_sampling*(t+1)]
        y_new = y_new_full[n_sampling*t:n_sampling*(t+1)]
        
        X_train, y_train = sp.update_synthetic_set(X_new, y_new, X_train, y_train, pure_online)
        
        nN_prev = len(X_train)
        
        if deleting and t > 0:
            X_train, y_train = fn.remove_point_rho(copy_model, 
                                                   X_train, 
                                                   y_train, 
                                                   thresh, 
                                                   d,
                                                   n_classes,
                                                   loss_function(reduction=tf.keras.losses.Reduction.NONE))
        
        nN = len(X_train)
                
        
        if (nN_prev - nN)<n_sampling:
            lmda = lmda/2
            #lmda = lmda
        elif (nN_prev - nN)>=n_sampling: 
            lmda = lmda*1.5
            #lmda = lmda
        
        save_to_csv(lmda, path, f'RUN_{run}_lmda.csv')     
        
        X_train_uniques = np.vstack((X_train_uniques, X_train))
        X_train_uniques = np.unique(X_train_uniques, axis=0)
        
        save_to_csv(len(X_train_uniques), path, f'RUN_{run}_unique_points.csv')
        
        y_errors = y_train        
        
        loss = loss_function(reduction=tf.keras.losses.Reduction.AUTO)
        rho_mean = loss(tf.one_hot(y_train, n_classes), copy_model.predict(X_train, verbose=0)).numpy()
        
        #rho_mean = fn.rho_calculation(copy_model,X_train,y_train,d,n_classes)
        save_to_csv(rho_mean, path, f'RUN_{run}_rhoMatrix.csv')
        save_to_csv(nN, path, f'RUN_{run}_nN.csv')  
        
        theta_prev, copy_model.weights_dims = fn.params_to_vec(copy_model, return_dims = True) 
        
        while len(y_errors) !=0 and n_subtraining <= max_subtraining:
            
            if n_subtraining > 0:
                if verbose:
                    print("{} RUN {} ITER {} entering subtraining {}...".format(tm.strftime("%y%m%d-%H:%M:%S", tm.gmtime()), run, t, n_subtraining-1))
                
                # shuffle data
                idx = np.random.permutation(len(X_train))
                X_train, y_train = X_train[idx], y_train[idx]
                
                # update learning rate
                opt.lr=lr_*(n_subtraining-1+1)/(n_subtraining-1+0.5)

            #rho_mean = fn.rho_calculation(copy_model,X_train,y_train,d,n_classes)
            
            # Create a callback that saves the model's weights
            if callbacks:
                cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=0)
            else:
                cp_callback = []
            copy_model, rho_max = train_copy_model(copy_model, X_train, y_train, d, n_classes, opt, loss_function(reduction=tf.keras.losses.Reduction.AUTO), theta_prev, sample_weights, lmda, acc_train, acc_thresh, rho_max, n_epochs, batch_size,[cp_callback])
            n_subtraining +=1
            
            # Identify errors
            y_pred_ohe = copy_model.predict(X_train, verbose=0)
            y_pred = np.argmax(y_pred_ohe, axis=1)
            X_errors = X_train[y_pred!=y_train,:]
            y_errors = y_train[y_pred!=y_train]

            # Compute copy accuracy
            acc_train = fn.evaluation(copy_model, X_train,y_train,d,n_classes)
            acc_test = fn.evaluation(copy_model,X_test,y_test,d,n_classes)
            
            
            if verbose:
                print("{} RUN {} ITER {} N {} ERRORS {} acc train: {:.2f}, acc test: {:.2f}, lambda: {:.5f}".format(tm.strftime("%y%m%d-%H:%M:%S", tm.gmtime()), run, t, len(X_train), len(X_errors), acc_train, acc_test, lmda))
        
        save_to_csv(acc_train, path, f'RUN_{run}_acc_train.csv')
        acc_test_vec.append(acc_test)
        save_to_csv(acc_test, path, f'RUN_{run}_acc_test.csv')
        
        theta_last = fn.params_to_vec(copy_model)
        last_norm = fn.norm_theta(theta_prev,theta_last)
        save_to_csv(last_norm.numpy(), path, f'RUN_{run}_thetas.csv')
        
        '''if deleting and t > 0:
            X_train, y_train = fn.remove_point_rho(copy_model, 
                                                   X_train, 
                                                   y_train, 
                                                   thresh, 
                                                   d,
                                                   n_classes,
                                                   loss_function(reduction=tf.keras.losses.Reduction.NONE))
        '''
        # Plot resulting model every 5 iterations
        if plot: #and t%5==0:   
            pl.plot_copy_model(copy_model, X_train, y_train, X_errors, y_errors, t, run, path, x_range=3.0)
        
        # reset values
        opt.lr = lr_
        n_subtraining = 0
        
        t += 1
    weight = copy_model.get_weights()
    save_to_csv(weight, path, 'final_weights.csv')

    

def get_year_month(year, month):
    
    if month=='12':
        year=str(int(year)+1)
        month = '01'
        return year, month
    
    month = str(int(month)+1)
    
    if len(month)==1:
        month = '0'+month
    
    return year, month
    
    
def sequential_train_use_case(d, 
                      n_classes,
                      lr_, 
                      n_sampling, 
                      original, 
                      X_test, 
                      y_test,
                      plot=True, 
                      max_iter=15, 
                      n_epochs=1000, 
                      batch_size=32,
                      max_subtraining=2, 
                      deleting=False, 
                      thresh = 0.001, 
                      path='',
                      dataset_name='',
                      run=0, 
                      sample_weights=False,
                      lmda = 0.0,
                      acc_thresh=0.85,
                      balancer = False,
                      loss = 'self',
                      optimizer='adam',
                      hidden_layers = [64,32,10],
                      activation='relu',
                      pure_online = False,
                      shiftting = False,
                      callbacks = False,
                      verbose=False):
  
    #if verbose:
    #    print("{} starting run {}...".format(tm.strftime("%y%m%d-%H:%M:%S", tm.gmtime()), run))
    
    # Initiliaze run
    t=0   
    n_subtraining = 0
    X_train_uniques = np.empty((0, d))
    X_train = np.empty((0, d))
    X_test_accum = X_test
    y_train = np.empty((0), dtype=int)
    y_test_accum = y_test
    
    if run>=50:
        print('Ey! we just have 50 different sampling files! so Im reusing one of them :)')
        run_=run//50
    else:
        run_=run
    
    rho_max = tf.constant(1.0, dtype = tf.float64)
    
    acc_train = 0
    acc_test_vec = []
    
    # Define copy model
    copy_model, opt, loss_function = create_copy_model(d, 
                                                       n_classes, 
                                                       lr_, 
                                                       optimizer=optimizer,
                                                       acc_thresh=acc_thresh,
                                                       input_shape=batch_size, 
                                                       hidden_layers=hidden_layers,
                                                       loss_name=loss,
                                                       activation=activation)
    
    year = '2020'
    test_size = 0.2
    train_size = 1 - test_size
    
    while t < max_iter:      
        
        if t>0:
            year , month = get_year_month(year, month)
            data_filenames= '/home/nahuel.statuto/sampling/use_case/'+year+'_'+month

            X_new =pd.read_csv(data_filenames+'_X.csv', header=None).to_numpy()
            y_new =pd.read_csv(data_filenames+'_y.csv', header=None).to_numpy()
            
            try:
                idx = np.random.choice(np.arange(len(X_new)), n_sampling, replace=False)
            except:
                idx = np.random.choice(np.arange(len(X_new)))
            X_new = X_new[idx]
            y_new = y_new[idx]        

            X_new, X_test_t, y_new, y_test_t = train_test_split(X_new, y_new, test_size=test_size, random_state=42)
            X_test_accum = np.vstack((X_test_accum, X_test_t))
            y_test_accum = np.append(y_test_accum, y_test_t)
            y_test_t = tf.squeeze(y_test_t).numpy()
        else:
            months = ['01','02']

            xx = np.empty((0, 22))
            yy = np.empty((0), dtype=int)

            for month in months:
                data_filenames='/home/nahuel.statuto/sampling/use_case/'+year+'_'+month
                xx = np.vstack((   np.asarray(pd.read_csv(data_filenames+'_X.csv', header=None).to_numpy()), xx  ))
                yy = np.append(   np.asarray(pd.read_csv(data_filenames+'_y.csv', header=None).to_numpy()) , yy )

            try:
                idx = np.random.choice(np.arange(len(xx)), n_sampling, replace=False)
            except:
                idx = np.random.choice(np.arange(len(xx)))
            xx = xx[idx]
            yy = yy[idx]


            X_new, X_test_t, y_new, y_test_t = train_test_split(xx, yy, test_size=0.2, random_state=42)
            X_test_accum = np.vstack((X_test_accum, X_test_t))
            y_test_accum = np.append(y_test_accum, y_test_t)
            y_test_t = tf.squeeze(y_test_t).numpy()
        
        X_train, y_train = sp.update_synthetic_set(X_new, y_new, X_train, y_train, pure_online)
        print(data_filenames)
        
        nN_prev = len(X_train)
        
        if deleting and t > 0:
            X_train, y_train = fn.remove_point_rho(copy_model, 
                                                   X_train, 
                                                   y_train, 
                                                   thresh, 
                                                   d,
                                                   n_classes,
                                                   loss_function(reduction=tf.keras.losses.Reduction.NONE))
        
        nN = len(X_train)
                
        
        if (nN_prev - nN)<n_sampling*train_size:
            lmda = lmda/2
            #lmda = lmda
        elif (nN_prev - nN)>=n_sampling*train_size: 
            lmda = lmda*1.5
            #lmda = lmda
        
        save_to_csv(lmda, path, f'RUN_{run}_lmda.csv')     
        
        X_train_uniques = np.vstack((X_train_uniques, X_train))
        X_train_uniques = np.unique(X_train_uniques, axis=0)
        
        save_to_csv(len(X_train_uniques), path, f'RUN_{run}_unique_points.csv')
        
        y_errors = y_train        
        
        loss = loss_function(reduction=tf.keras.losses.Reduction.AUTO)
        rho_mean = loss(tf.one_hot(y_train, n_classes), copy_model.predict(X_train, verbose=0)).numpy()
        
        #rho_mean = fn.rho_calculation(copy_model,X_train,y_train,d,n_classes)
        save_to_csv(rho_mean, path, f'RUN_{run}_rhoMatrix.csv')
        save_to_csv(nN, path, f'RUN_{run}_nN.csv')  
        
        theta_prev, copy_model.weights_dims = fn.params_to_vec(copy_model, return_dims = True) 
        
        while len(y_errors) !=0 and n_subtraining <= max_subtraining:
            
            if n_subtraining > 0:
                if verbose:
                    print("{} RUN {} ITER {} entering subtraining {}...".format(tm.strftime("%y%m%d-%H:%M:%S", tm.gmtime()), run, t, n_subtraining-1))
                
                # shuffle data
                idx = np.random.permutation(len(X_train))
                X_train, y_train = X_train[idx], y_train[idx]
                
                # update learning rate
                opt.lr=lr_*(n_subtraining-1+1)/(n_subtraining-1+0.5)

            #rho_mean = fn.rho_calculation(copy_model,X_train,y_train,d,n_classes)
            
            # Create a callback that saves the model's weights
            if callbacks:
                cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=0)
            else:
                cp_callback = []
            copy_model, rho_max = train_copy_model(copy_model, X_train, y_train, d, n_classes, opt, loss_function(reduction=tf.keras.losses.Reduction.AUTO), theta_prev, sample_weights, lmda, acc_train, acc_thresh, rho_max, n_epochs, batch_size,[cp_callback])
            n_subtraining +=1
            
            # Identify errors
            y_pred_ohe = copy_model.predict(X_train, verbose=0)
            y_pred = np.argmax(y_pred_ohe, axis=1)
            X_errors = X_train[y_pred!=y_train,:]
            y_errors = y_train[y_pred!=y_train]

            # Compute copy accuracy
            acc_train = fn.evaluation(copy_model, X_train,y_train,d,n_classes)
            acc_test = fn.evaluation(copy_model,X_test,y_test,d,n_classes)
            acc_test_t = fn.evaluation(copy_model,X_test_t,y_test_t,d,n_classes)
            acc_test_accum = fn.evaluation(copy_model,X_test_accum,y_test_accum,d,n_classes)
            acc_test_original = fn.evaluation(original,X_test_accum,y_test_accum,d,n_classes)
            
            
            if verbose:
                print("{} RUN {} ITER {} N {} ERRORS {} acc train: {:.2f}, acc test: {:.2f}, lambda: {:.5f}".format(tm.strftime("%y%m%d-%H:%M:%S", tm.gmtime()), run, t, len(X_train), len(X_errors), acc_train, acc_test, lmda))
        
        save_to_csv(acc_train, path, f'RUN_{run}_acc_train.csv')
        acc_test_vec.append(acc_test)
        save_to_csv(acc_test, path, f'RUN_{run}_acc_test.csv')
        save_to_csv(acc_test_t, path, f'RUN_{run}_acc_test_t.csv')
        save_to_csv(acc_test_accum, path, f'RUN_{run}_acc_test_accum.csv')
        save_to_csv(acc_test_original, path, f'RUN_{run}_acc_original.csv')
        
        theta_last = fn.params_to_vec(copy_model)
        last_norm = fn.norm_theta(theta_prev,theta_last)
        save_to_csv(last_norm.numpy(), path, f'RUN_{run}_thetas.csv')
        
        '''if deleting and t > 0:
            X_train, y_train = fn.remove_point_rho(copy_model, 
                                                   X_train, 
                                                   y_train, 
                                                   thresh, 
                                                   d,
                                                   n_classes,
                                                   loss_function(reduction=tf.keras.losses.Reduction.NONE))
        '''
        # Plot resulting model every 5 iterations
        if plot: #and t%5==0:   
            pl.plot_copy_model(copy_model, X_train, y_train, X_errors, y_errors, t, run, path, x_range=3.0)
        
        # reset values
        opt.lr = lr_
        n_subtraining = 0
        
        t += 1
        