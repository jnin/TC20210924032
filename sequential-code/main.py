import os
import warnings
import time as tm
import numpy as np
import joblib
from joblib import Parallel, delayed
from math import sqrt
import datasets as dt
import preprocessing as pr
import sequential_copy as sc

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

root = '..'
random_state = 42
test_size = 0.2
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

###---- DATASET
dic_datasets = dt.dic_for_datasets()

pendets = ['statlog-image','vertebral-column-2clases','statlog-heart','twonorm','statlog-german-credit','titanic', 'teaching','statlog-australian-credit','statlog-vehicle','spambase']

pendets = ['spirals']

mm = int(len(pendets)/4)
##########################################################
ii = 4 #range(5) para pasar por todas
## faltan: ii=
##########################################################

datasets = []
original_models=[]
#for dd in pendets[ii*mm:ii*mm+mm]:
for dd in pendets:
    print(dd)
    datasets.append(dd)
    original_models.append(dic_datasets[dd])


###---- COPY
# Training parameters
loss = 'self' # self, bce
sample_weights = False # sample weights NOT IMPLEMENTED YET
learning_rate=0.0005 # starting learning rate
n_epochs = 1000 # number of epochs per iteration
batch_size = 32
hidden_layers = [64,32,10]
activation='relu'
optimizer='adam'

# Sequential parameters
lmda = [0.0] # parámetro que multiplica la regularización 
pure_online = True # shows only N new points at each iteration
deleting = False # whether to remove points with rho below threshold or not
thresh_vec =[1e-10]#[5e-4,1e-4,5e-5,1e-5,5e-6,1e-6,5e-7,1e-7,5e-8,1e-8,1e-9,1e-10] # thershold for point deletion
acc_thresh = 0.40 # NOT USED YET - therehold in accuracy to add norms to lossfunction
max_iter = 35 # maximum number of iterations
n_samples_iter = [100] # number of new samples per iteration
balancer = False # whether to use or not balancer function to generate synthetic data
max_subtraining = 0 # number of subtraining allowed to get zero errors

###---- VALIDATION
n_runs = 20 #number independent runs
n_jobs = n_runs # number of jobs
test_mode = True

if __name__ == "__main__":
    
    # Timestamps
    t = tm.strftime("%y%m%d-%H%M%S", tm.gmtime())
    
    if test_mode:
            results_folder_name = 'results/test'
    else:
        results_folder_name = 'results/irene'
    
     # Read UCI dataset names
    if 'UCI' in datasets:
        names = joblib.load('UCI_names.pkl')
        datasets.remove('UCI')
        datasets.extend(names)

    for dataset, original_model in zip(datasets, original_models):
        
        # Create original dataset
        print("{} creating {} dataset...".format(tm.strftime("%y%m%d-%H:%M:%S", tm.gmtime()), dataset.upper()))
        X_train, y_train, X_test, y_test = dt.create_dataset(dataset, 
                                                             path=os.path.join(root, 'data'),
                                                             test_size=test_size, 
                                                             random_state=random_state) 
        d = X_test.shape[1]
        n_classes = len(np.unique(y_test))
        print("{} d: {}, n_classes: {}".format(tm.strftime("%y%m%d-%H:%M:%S", tm.gmtime()), d, n_classes))
        
        # Train original model
        print("{} creating {} model...".format(tm.strftime("%y%m%d-%H:%M:%S", tm.gmtime()), original_model))
        original = pr.create_original_model(dataset,
                                            original_model,
                                            X_train, 
                                            y_train, 
                                            X_test,
                                            y_test, 
                                            path=root,
                                            grid_search=True,
                                            random_state=random_state,
                                            plot=False)
        
        
        '''y_test_pred = original.predict(X_test)
        acc = np.average(np.where(y_test_pred==y_test,1.,0.))
        sc.save_to_csv([dataset,d,n_classes,acc],'../plots_paper/seq_vs_single','tt.cvs')
        #print('AO: {}'.format(np.average(np.where(y_test_pred==y_test,1.,0.))))
        
        '''
        # Create results path
        if not os.path.exists(os.path.join(root, results_folder_name, dataset)):
            os.mkdir(os.path.join(root, results_folder_name, dataset))
        
        
        # Run for the different values of the lmda parameter
        for l in lmda:
            for n in n_samples_iter:
                for thresh in thresh_vec:
                    params = {'dataset': dataset, 
                              'n_samples_iter': n,
                              'loss': loss,
                              'learning_rate': learning_rate, 
                              'n_runs': n_runs, 
                              'max_iter': max_iter,
                              'n_epochs': n_epochs, 
                              'batch_size': batch_size,
                              'max_subtraining': max_subtraining, 
                              'deleting': deleting, 
                              'thresh': thresh,
                              'sample_weights': sample_weights, 
                              'balancer': balancer,
                              'lmda': l,
                              'acc_thresh': acc_thresh,
                              'loss': loss,
                              'optimizer': optimizer,
                              'pure_online': pure_online}

                    path = 'N_SAMPLES_ITER_%(n_samples_iter)s_LR_%(learning_rate)s_'% params

                    # Create experiments path
                    if not os.path.exists(os.path.join(root, results_folder_name, dataset, path)):
                        os.mkdir(os.path.join(root, results_folder_name, dataset, path))
                    folder_name = '-LOSS_%(loss)s_PURE_%(pure_online)s_DEL_%(deleting)s_THRESH_%(thresh)s_LMDA_%(lmda)s_BALANCER_%(balancer)s_RUNS_%(n_runs)s'% params
                    os.mkdir(os.path.join(root, results_folder_name, dataset, path, t+folder_name))
                    os.mkdir(os.path.join(root, results_folder_name, dataset, path, t+folder_name, 'plots'))

                    # Save experiment parameters
                    print("{} saving params...".format(tm.strftime("%y%m%d-%H:%M:%S", tm.gmtime())))
                    pr.save_params(os.path.join(root, results_folder_name,dataset, path, t+folder_name), params)

                    # Copy
                    print("{} running for n={}, lmda={}...".format(tm.strftime("%y%m%d-%H:%M:%S", tm.gmtime()), n, l))


                    with Parallel(n_jobs=n_jobs) as parallel:
                        parallel(delayed(sc.sequential_train)(d, 
                                                               n_classes, 
                                                               learning_rate, 
                                                                 n, 
                                                                 plot=False, 
                                                                 max_iter=max_iter, 
                                                                 n_epochs=n_epochs,
                                                                 batch_size=batch_size,
                                                                 max_subtraining=max_subtraining, 
                                                                 original=original, 
                                                                 X_test=X_test, 
                                                                 y_test=y_test, 
                                                                 deleting=deleting,
                                                                 thresh=thresh, 
                                                                 path=os.path.join(root, results_folder_name, dataset, path, t+folder_name), 
                                                                 dataset_name=dataset,
                                                                 run=i, 
                                                                 sample_weights=sample_weights,                          
                                                                 lmda=l,
                                                                 acc_thresh=acc_thresh,
                                                                 balancer=balancer,
                                                                 loss=loss,
                                                                 optimizer=optimizer,
                                                                 hidden_layers=hidden_layers,
                                                                 activation=activation,
                                                                 pure_online=pure_online,
                                                                 verbose=True)  for i in range(0, n_runs))