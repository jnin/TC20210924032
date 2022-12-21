import os
import joblib
import pandas as pd
import numpy as np
from sklearn.utils import check_array, check_consistent_length
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

UCI_names = joblib.load('UCI_names.pkl')

def create_dataset(dataset, 
                   path='../data', 
                   test_size=0.2, 
                   random_state=42):
    
    if not os.path.exists(os.path.join(path, dataset)):
        os.mkdir(os.path.join(path, dataset))
    
    if not os.path.exists(os.path.join(path, dataset, '{}_data.pkl'.format(dataset))):
    
        if dataset == 'spirals':
            X, y = spirals(5000, noise=1) 
        elif dataset == 'yinyang':
            X, y = yinyang(10000)
        elif dataset == 'moons':
            X, y = moons(10000)
        elif dataset == 'iris':
            X, y = iris()
        elif dataset == 'wine':
            X, y = wine()
        elif dataset == 'covertype':
            X, y = covertype()
        elif dataset in UCI_names:
            X, y = uci()
        else:
            raise NameError("The value {} is not allowed for variable dataset. Please choose spirals, yinyang, moons, iris, wine, covertype or UCI".format(dataset))

        #Split dataset into subsets that minimize the potential for bias in your evaluation and validation process.
        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            y.astype(int), 
                                                            test_size=test_size, 
                                                            random_state=random_state,
                                                            stratify=y)

        scaler = StandardScaler(copy=True)
        scaler.fit(X_train)
        X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        data = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
        
        joblib.dump(data, os.path.join(path, dataset, '{}_data.pkl'.format(dataset)))
    
    else:
        data = joblib.load(os.path.join(path, dataset, '{}_data.pkl'.format(dataset)))
    
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']

def uci():
    
    # Read raw data
    data = pd.read_table(os.path.join(path, dataset, '{}_R.dat'.format(dataset)), index_col=0)
    dtype = dtypes[dataset]
    
    # Convert to matrix format
    X = data.drop('clase', axis=1).to_numpy()
    y = data['clase'].to_numpy()
    
    # Re-order columns
    idx = dtype.argsort()
    dtype = dtype[idx[::-1]]
    X = X[:, idx[::-1]]
        
    # Preprocessing.
    X = check_array(X, accept_sparse=True, ensure_min_samples=1, dtype=np.float64)
    y = check_array(y, ensure_2d=False, ensure_min_samples=1, dtype=None)
    dtype = check_array(dtype, ensure_2d=False, ensure_min_samples=1, dtype=None)
    check_consistent_length(X, y)
    
    return X, y

def covertype():
    """
     Returns the covertype dataset.
    """
    COVTYPE = datasets.fetch_covtype()
    X = COVTYPE.data
    y = COVTYPE.target

    return X, y

def wine():
    """
     Returns the IRIS dataset.
    """
    WINE = datasets.load_wine()
    X = WINE.data
    y = WINE.target
    return X,y

def iris():
    """
     Returns the IRIS dataset.
    """
    IRIS = datasets.load_iris()
    X = IRIS.data
    y = IRIS.target
    return X, y

def moons(n_samples):
    """
     Returns the make_moons dataset.
    """
    X, y =  datasets.make_moons(n_samples=n_samples, noise=0.05)
    return X,y,'moons'

def yinyang(n_samples):
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

    return X, y

def spirals(n_samples, noise=.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_samples,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_samples,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_samples,1) * noise
    return np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))),np.hstack((np.zeros(n_samples),np.ones(n_samples)))

def dic_for_datasets():
    dic_datasets ={'congressional-voting':'MLP',
        'credit-approval': 'MLP',
        'haberman-survival': 'RFC',
        'ionosphere': 'RFC',
        'magic': 'LinearSVM',
        'pima': 'LinearSVM',
        'synthetic-control': 'GaussianSVM',
        'ringnorm': 'GaussianSVM',
        'tic-tac-toe': 'XgBoost',
        'waveform': 'XgBoost',
        'breast-cancer': 'AdaBoost',
        'breast-cancer-wisc': 'AdaBoost',
        'breast-cancer-wisc-diag': 'AdaBoost',
        'breast-cancer-wisc-prog': 'AdaBoost',
        'bank': 'AdaBoost',
        'breast-tissue': 'AdaBoost',
        'chess-krvkp': 'MLP',
        'conn-bench-sonar-mines-rocks': 'MLP',
        'connect-4': 'MLP',
        'contrac': 'MLP',
        'cylinder-bands': 'MLP',
        'echocardiogram': 'MLP',
        'energy-y1': 'MLP',
        'energy-y2': 'MLP',
        'fertility': 'RFC',
        'heart-hungarian': 'RFC',
        'hepatitis': 'RFC',
        'ilpd-indian-liver': 'RFC',
        'iris': 'RFC',
        'mammographic': 'RFC',
        'miniboone': 'RFC',
        'molec-biol-splice': 'RFC',
        'mushroom': 'LinearSVM',
        'musk-1': 'LinearSVM',
        'musk-2': 'LinearSVM',
        'oocytes_merluccius_nucleus_4d': 'LinearSVM',
        'oocytes_trisopterus_nucleus_2f': 'LinearSVM',
        'parkinsons': 'LinearSVM',
        'pittsburg-bridges-MATERIAL': 'LinearSVM',
        'pittsburg-bridges-REL-L': 'LinearSVM',
        'pittsburg-bridges-T-OR-D': 'GaussianSVM',
        'planning': 'GaussianSVM',
        'seeds': 'GaussianSVM',
        'spambase': 'GaussianSVM',
        'statlog-australian-credit': 'GaussianSVM',
        'statlog-german-credit': 'GaussianSVM',
        'statlog-heart': 'GaussianSVM',
        'statlog-image': 'GaussianSVM',
        'statlog-vehicle': 'XgBoost',
        'teaching': 'XgBoost',
        'titanic': 'XgBoost',
        'twonorm': 'XgBoost',
        'vertebral-column-2clases': 'XgBoost',
        'vertebral-column-3clases': 'XgBoost',
        'waveform-noise': 'XgBoost',
        'wine': 'XgBoost',
        'spirals':'GaussianSVM'}
    return dic_datasets
