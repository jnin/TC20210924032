from sklearn import datasets
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd

def rho_calculation(copy_model, X_train, y_train,d):
    y_pred_ohe = copy_model.predict(X_train)
    y_train_ohe=tf.one_hot(y_train, 2)
    avv_= 0
    vec_rho = []
    for y_t, t_p in zip(y_train_ohe,y_pred_ohe):
        temp=0
        for dim in range(0,d):
            temp +=(y_t[dim]-t_p[dim])**2
        temp = np.sqrt(temp)
        vec_rho.append(temp)
        avv_+=temp
    return avv_/len(y_train_ohe), vec_rho

def remove_point_rho(X, y, threshold, vec_rho, d):
    X_train_ = np.empty((0, d))
    y_train_ = np.empty((0), dtype=int)

    for i, rho in enumerate(vec_rho):
        if rho>=threshold:
            X_train_=np.append(X_train_,[X[i]])
            y_train_=np.append(y_train_,y[i])

    X_train_=X_train_.reshape((int(len(X_train_)/2), 2))
    return X_train_ , y_train_


def define_compile_model(lr=0.01,seed=42):
    tf.random.set_seed(seed)

    copy_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, input_shape=(2,), activation='relu', kernel_initializer='he_normal'),
            tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_normal'),  
            tf.keras.layers.Dense(10, activation='relu', kernel_initializer='he_normal'),
            tf.keras.layers.Dense(2, activation='softmax')
          ])

    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    copy_model.compile(optimizer=opt, 
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    return copy_model

def save_params(main_path,data_path,params):
    try:
        with open(main_path +'/' +data_path+ '/'+'params.txt', "w") as text_file:
            print("saving params in =",text_file)
            for key in params:
                text_file.write(str(key)+' = '+ str(params[key]) + '\n')
    except:
        with open(data_path+ '/'+'params.txt', "w") as text_file:
            print("saving params in =",text_file)
            for key in params:
                text_file.write(str(key)+' = '+ str(params[key]) + '\n')


def fit_function_2(lr_, n_sampling, original, X_test, y_test, hot_start, plot=True, max_iter=15, n_epochs = 1000, max_subtraining = 2, deleting=False, threshold = 0.001, model_path='',step=0):
    d=2
    t=0
    X_train_ = np.empty((0, d))
    y_train_ = np.empty((0), dtype=int)
    y_errors = np.empty((0), dtype=int)
    acc = []
    avv = []
    rho = []
    nN = []
    rho_vec = []
    n_sub = []
    
    ##DEFINING MODEL-----------------------------------------------------------------------------------------
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if hot_start:
        r_seed = np.random.randint(1000)
    else:
        r_seed = 42
        
    tf.random.set_seed(r_seed)
    print('1')

    copy_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, input_shape=(2,), activation='relu', kernel_initializer='he_normal'),
            tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_normal'),  
            tf.keras.layers.Dense(10, activation='relu', kernel_initializer='he_normal'),
            tf.keras.layers.Dense(2, activation='softmax')
          ])

    opt = tf.keras.optimizers.Adam(learning_rate=lr_)

    copy_model.compile(optimizer=opt, 
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    ##\DEFINING MODEL-----------------------------------------------------------------------------------------
    print('2')
    n_subtraining = 0
    #print('Step: {}'.format(i),'-----------------------------')
    t_rand=np.random.randint(15)

    while t < max_iter:

        print('Iteration: {}'.format(t))
        #print('Subtraining: {}'.format(n_subtraining))
        if len(y_errors)==0:          
            # Generate new points and label them
            X_new = np.random.multivariate_normal(np.zeros((d,)), np.eye(d,d), size = n_sampling)
            y_new = original.predict(X_new)
            y_new_oh_mamma = tf.one_hot(y_new, 2)
            y_new = np.argmax(y_new_oh_mamma, axis=1)

            # Update synthetic trainign set with new points
            X_train_ = np.vstack((X_new, X_train_))
            y_train_ = np.append(y_new, y_train_)
            
            rho_mean, vec_rho = rho_calculation(copy_model,X_train_,y_train_,d)
            rho.append(rho_mean)
            rho_vec.append(vec_rho)
            
            if deleting:
                X_train_, y_train_ = remove_point_rho(X_train_, y_train_, threshold, vec_rho,d)
            
            nN.append(len(X_train_))
            
        else: # decrease of the lr
            weights = copy_model.get_weights()  # save the weights
            copy_model = define_compile_model(lr=lr_*(n_subtraining+1.5)/(n_subtraining+0.5),seed=r_seed) # 0.5 , 0.75, ...
            copy_model.set_weights(weights)  # set the weights

        X_train_, xx, y_train_, yy = train_test_split(X_train_, y_train_, test_size=1)
        X_train_ = np.vstack((xx, X_train_))
        y_train_ = np.append(yy, y_train_)

        # Update copy
        y_train_ohe = tf.one_hot(y_train_, 2)
        copy_model.fit(X_train_, y_train_ohe, epochs=n_epochs, batch_size=32, verbose=0)

        # Compute copy accuracy on original test data
        acc_=copy_model.evaluate(X_test, tf.one_hot(y_test, 2), verbose=0)[1]
        #print('Accuracy: {}'.format(acc_))

        # Identify errors
        y_pred_ohe = copy_model.predict(X_train_)
        #print(y_pred_ohe)
        y_pred_ = np.argmax(y_pred_ohe, axis=1)
        X_errors = X_train_[y_pred_!=y_train_,:]
        y_errors = y_train_[y_pred_!=y_train_]
        print('# Errors: {}'.format(len(y_errors)),'  # nN: {}'.format(nN[-1]), '  Accuracy: {}'.format(acc_))

        if len(y_errors)==0 or n_subtraining>max_subtraining:

            # Plot model-----------------------
            if plot and t==t_rand:
                x_range=3.0
                xx,yy = np.meshgrid(np.linspace(-x_range,x_range,200),np.linspace(-x_range,x_range,200))
                viz=np.c_[xx.ravel(),yy.ravel()]
                z = np.argmax(copy_model.predict(viz), axis=1)
                plt.scatter(X_train_[:, 0], X_train_[:, 1], c=y_train_,  alpha=0.7)
                plt.scatter(X_errors[:, 0], X_errors[:, 1], c='red',  alpha=0.2)
                plt.imshow(z.reshape((200,200)), origin='lower', extent=(-x_range,x_range,-x_range,x_range),alpha=0.3, vmin=0, vmax=1)
                plt.contour(xx,yy,z.reshape((200,200)),[0.5])
                plt.autoscale(False)
                plt.gcf().set_size_inches((6,6))
                plt.savefig(plot_path+'/iter='+str(t)+'_avg='+str(step)+'.pdf',bbox_inches='tight')
            #\ Plot model----------------------

            if n_subtraining !=0:
                weights = copy_model.get_weights()  # save the weights
                copy_model = define_compile_model(lr=lr_,seed=r_seed) # set original lr
                copy_model.set_weights(weights)  # set the weights
            t += 1
            n_sub.append(n_subtraining)
            n_subtraining = 0
            y_errors = np.empty((0), dtype=int)
            acc.append(acc_)

            if not hot_start:
                copy_model = define_compile_model(lr=lr_,seed=r_seed)
        else:
            n_subtraining +=1
    
    file_name=model_path+'/N=' + str(n_sampling) +'_step=' +str(step) +'_hot_start=' +str(hot_start) + '_acc.csv'
    with open(file_name, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        # write the data
        writer.writerow(np.asarray(acc))
    
    file_name=model_path+'/N=' + str(n_sampling) +'_step=' +str(step) +'_hot_start=' +str(hot_start) + '_nN.csv'
    with open(file_name, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        # write the data
        writer.writerow(np.asarray(nN)) 

    
def moons(n_samples):
    X, y =  datasets.make_moons(n_samples=n_samples, noise=0.05)
    return X,y,'moons'

def yin_yang(n_samples):
    """
     Returns the yin-yang dataset.
    """

    r_max = 1
    r = np.random.uniform(low=0, high=r_max**2, size=n_samples)
    theta = np.random.uniform(low=0, high=1, size=n_samples) * 2 * np.pi
    x = np.sqrt(r) * np.cos(theta)
    y = np.sqrt(r) * np.sin(theta)
    X = np.dstack([x, y])[0]
    y = np.empty((len(X),))

    # Upper circle
    center_x_u = 0
    center_y_u = 0.5
    radius_u = 0.5

    # Upper circle
    center_x_l = 0
    center_y_l = -0.5
    radius_l = 0.5

    i = 0
    for xi, yi in X:
        if ((xi > 0) & ((xi - center_x_u)**2 + (yi - center_y_u)**2 >= radius_u**2)) or ((xi < 0) & ((xi - center_x_l)**2 + (yi - center_y_l)**2 < radius_l**2)):
            y[i] = 1
        else:
            y[i] = 0

        if (xi - 0)**2 + (yi - 0.5)**2 < 0.15**2:
            y[i] = 1

        if (xi - 0)**2 + (yi - (-0.5))**2 < 0.15**2:
            y[i] = 0

        i += 1

    return X, y, 'yinyang'

def two_spirals(n_samples, noise=.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_samples,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_samples,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_samples,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))),
            np.hstack((np.zeros(n_samples),np.ones(n_samples))), 'spirals')