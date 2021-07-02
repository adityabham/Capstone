import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.ensemble as ske
import sklearn.model_selection as skms
import sklearn.metrics as skm
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


def hyperparameter_tuning_random(labeled_data):

    train_data = labeled_data[~labeled_data.param.isin()]

    # if any
    drop_cols = ['insignificant_data_parameters']
    predict_data = train_data.drop(drop_cols, axis=1).fillna(0)

    X_train, X_valid, y_train, y_valid = skms.train_test_split(predict_data, train_data.label)

    rf = ske.RandomForestClassifier(max_depth=2, class_weight='balanced')
    rf.fit(X_train, y_train)

    print('Parameters currently in use:\n')
    pprint(rf.get_params())

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=10, stop=100, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [int(x) for x in np.linspace(2, 10, num=8)]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [int(x) for x in np.linspace(2, 10, num=8)]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    print('\n')
    print('grid of random, test hyperparamters')
    pprint(random_grid)
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=7)
    rf_random.fit(X_train, y_train)

    print('\n')
    print('Best hyperparamters from random search\n')
    pprint(rf_random.best_params_)

    best_random = rf_random.best_estimator_
    predict_train_random_opt = best_random.predict(X_train)
    predict_valid_random_opt = best_random.predict(X_valid)

    print('------------------------------------------------')
    print('Random Search hyperparameter tuning evaluation')
    print('------------------------------------------------')

    print('training labels vs predictions')
    print(skm.confusion_matrix(y_train, predict_train_random_opt))
    x = skm.confusion_matrix(y_train, predict_train_random_opt)
    train_acc = ((x[1][1] + x[2][2])/(x[1][1]+x[1][2]+x[2][1]+x[2][2])) * 100
    print(train_acc)
    train_power = (x[2][2] / (x[1][2] + x[2][2])) * 100
    print(train_power)

    print('validation labels vs predictions')
    print(skm.confusion_matrix(y_valid, predict_valid_random_opt))
    y = skm.confusion_matrix(y_valid, predict_valid_random_opt)
    valid_acc = ((y[1][1] + y[2][2]) / (y[1][1] + y[1][2] + y[2][1] + y[2][2])) * 100
    print(valid_acc)
    valid_power = (y[2][2] / (y[1][2] + y[2][2])) * 100
    print(valid_power)

# Using the best parameters resulting from RandomizedSearchCV, GridSearchCV can be utilized for further tuning using
# the same process







