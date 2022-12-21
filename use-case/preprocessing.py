import os
import csv
import joblib
import warnings

warnings.simplefilter(action='ignore')

import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

import plot as pl

estimators = {'AdaBoost': (AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), random_state=42),
                         {"base_estimator__criterion" : ["gini", "entropy"], "base_estimator__splitter" : ["best", "random"],
                          "base_estimator__class_weight" : [None, 'balanced'], "n_estimators" : [50, 100, 500, 1000],
                          "learning_rate" : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]}), 
            'MLP': (MLPClassifier(random_state=42, hidden_layer_sizes=(42), learning_rate='adaptive', max_iter=100000),
                    {"activation" : ["identity", "logistic", "tanh", "relu"], "solver" : ["lbfgs", "sgd", "adam"], "alpha" :
                     [0.0001, 0.001, 0.01], "learning_rate_init" : [0.001, 0.01, 0.1]}),
            'RFC': (RandomForestClassifier(random_state=42, n_estimators=100, max_features=None, max_depth=None, n_jobs=-1),
                    {"criterion" : ["gini", "entropy"]}), 
            'LinearSVM': (SVC(random_state=42, kernel='linear', probability=True),
                         {"C" : [0.01, 0.1, 1, 10, 100], "class_weight": [None, 'balanced']}), 
            'GaussianSVM': (SVC(random_state=42, kernel='rbf', probability=True), 
                            {"C" : [0.01, 0.1, 1, 10], "gamma" : [0.001, 0.1, 1, 10], "class_weight": [None, 'balanced']}),  
            'XgBoost': (xgb.XGBClassifier(random_state=42, n_estimators=1000, max_depth= 5, n_jobs=-1, silent=True, verbose=0),
                        {"learning_rate" : [0.001, 0.01, 0.1, 1], "booster" : ["gbtree"]})}

def save_params(path, 
                params):
    
    with open(os.path.join(path, 'params.txt'), "w") as text_file:
        for key in params:
            text_file.write(str(key)+' = '+ str(params[key]) + '\n')

def create_original_model(dataset, 
                          model,
                          X_train,
                          y_train,
                          X_test,
                          y_test,
                          path, 
                          grid_search=False,
                          random_state=42,
                          plot=False):
    
    if not os.path.exists(os.path.join(path, 'data', dataset, '{}_model.pkl'.format(model))):
        
        estimator = estimators[model][0]
        
        if grid_search:
            
            grid = GridSearchCV(estimator, param_grid=estimators[model][1], scoring='accuracy', verbose=1)
            grid.fit(X_train, y_train)
            estimator = grid.best_estimator_
        
        original = estimator.fit(X_train, y_train)
            
        joblib.dump(original, os.path.join(path, 'data', dataset, '{}_model.pkl'.format(model)))
        
        if hasattr(original, 'get_params'):
            joblib.dump(original.get_params(), os.path.join(path, 'data', dataset, '{}_params.pkl'.format(model)))
        else:
            joblib.dump(original.get_config(), os.path.join(path, 'data', dataset, '{}_params.pkl'.format(model)))

    else:
        original = joblib.load(os.path.join(path, 'data', dataset, '{}_model.pkl'.format(model)))
        
    # Plot model
    if plot:
        pl.plot_original_model(dataset,
                               original, 
                               X_train, 
                               y_train, 
                               path)
    
    return original



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
