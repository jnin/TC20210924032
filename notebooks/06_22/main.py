import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import time as tm
import os
import csv 
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from math import sqrt

import functions as fn

main_path = ''

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#PARAMETERS--------------------------------------------------------
big_N = [200]

learning_rate=0.00004
n_averaging=50
max_iter=15
n_epochs = 1000
max_subtraining = 10
hot_start = True
deleting = True
threshold = 0.5e-4
dataset=1  #1- Spirals, 2- yinYang, 3- Moons

test=False

n_jobs = 20
#------------------------------------------------------------------


if dataset == 1:
      X,y,dataset_name=fn.two_spirals(500,noise = 1)
elif dataset == 2:
      X,y,dataset_name=fn.yin_yang(10000)
else:
      X,y,dataset_name=fn.moons(10000)


t = tm.strftime("%d%m%y-%H:%M:%S", tm.gmtime())

fd = {'dataset': dataset_name, 'big_N': big_N, 'lr': learning_rate, 'n_averaging': n_averaging, 'max_iter': max_iter,
      'n_epochs': n_epochs, 'max_subtraining': max_subtraining, 'hot_start': hot_start, 'deleting': deleting, 'threshold': threshold}
if test:
    model_path ='test/'+t+'-dataset_%(dataset)s_big_N_%(big_N)s_lr_%(lr)s_hot_start_%(hot_start)s_del_%(deleting)s_threshold_%(threshold)s'% fd
else: 
    model_path ='data/'+t+'-dataset_%(dataset)s_big_N_%(big_N)s_lr_%(lr)s_hot_start_%(hot_start)s_del_%(deleting)s_threshold_%(threshold)s'% fd
fd['model_path']=model_path

os.mkdir(model_path)
os.mkdir(model_path+'/'+'plots')

print("saving params")
fn.save_params(main_path,model_path,fd)


#Standardize and split to create training and test sets

#split your dataset into subsets that minimize the potential for bias in your evaluation and validation process.
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y.astype(int), test_size=test_size, random_state=42)

scaler = StandardScaler(copy=True)
scaler.fit(X_train)
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

### Original model
#Train and evaluate a gaussian kernel SVM

original = SVC(gamma=10)
original.fit(X_train,y_train)

x_range=3.0
xx,yy = np.meshgrid(np.linspace(-x_range,x_range,200),np.linspace(-x_range,x_range,200))
viz=np.c_[xx.ravel(),yy.ravel()]

z = original.predict(viz)

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
plt.savefig(model_path +'/plots/original_model:'+dataset_name+'.pdf',bbox_inches='tight')

xx,yy = np.meshgrid(np.linspace(-2.5,2.5,200),np.linspace(-2.5,2.5,200))
viz=np.c_[xx.ravel(),yy.ravel()]

y_test_pred = original.predict(X_test)
print('AO: {}'.format(np.average(np.where(y_test_pred==y_test,1.,0.))))

##COPY----------------------------------------

plots_path = model_path+'/plots/iterations'
os.mkdir(plots_path)

for n_sample in big_N:
    print('#N: {}'.format(n_sample),'--------------------------------------')    
    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(delayed(fn.fit_function_2)(learning_rate, n_sample, plot=True, max_iter=max_iter, n_epochs=1000,
                    max_subtraining=max_subtraining, original=original, X_test=X_test, y_test=y_test,
                    hot_start=hot_start, deleting=deleting, threshold=threshold, model_path=model_path,
                    step=i)  for i in range(0,n_averaging))
      
