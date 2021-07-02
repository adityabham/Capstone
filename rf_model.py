# Aditya Bhamidipati

import numpy as np
import sklearn.ensemble as ske
import sklearn.model_selection as skms
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

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


def ml_arch(predictor_data):

    # Initializing DFs
    drop_cols = ['id', 'true_label', 'new_label']
    raw_data = predictor_data.drop(drop_cols, axis=1).fillna(0)

    # --------------------------------------------

    Xi_train, Xi_test, yi_train, yi_test = skms.train_test_split(raw_data, predictor_data.true_label, random_state=0)

    # random forest meta estimator initialization
    rf_intent = ske.RandomForestClassifier(max_depth=2, class_weight='balanced')
    rf_intent.fit(Xi_train, yi_train)

    rf_intent_random = RandomizedSearchCV(estimator=rf_intent, param_distributions=random_grid, n_iter=100, cv=5)
    rf_intent_random.fit(Xi_train, yi_train)
    rf_best_intent_random = rf_intent_random.best_estimator_

    predicti_test_opt = rf_best_intent_random.predict(Xi_test)

    # predictions

    confusion_i = confusion_matrix(yi_test, predicti_test_opt)
    print('Confusion Matrix (intent lables)')
    print(confusion_i)
    print('Accuracy: {:.2f}'.format(accuracy_score(yi_test, predicti_test_opt)))

    ml_intent_label = rf_best_intent_random.predict(raw_data)

    predictor_data['ml_intent_label'] = ml_intent_label

    # --------------------------------------------

    Xn_train, Xn_test, yn_train, yn_test = skms.train_test_split(raw_data, predictor_data.new_label, random_state=0)
    # random forest meta estimator initialization
    rf_pred_label = ske.RandomForestClassifier(max_depth=2, class_weight='balanced')
    rf_pred_label.fit(Xn_train, yn_train)

    rf_pred_random = RandomizedSearchCV(estimator=rf_pred_label, param_distributions=random_grid, n_iter=100, cv=5)
    rf_pred_random.fit(Xn_train, yn_train)
    rf_best_pred_random = rf_pred_random.best_estimator_

    predictn_test_opt = rf_best_pred_random.predict(Xn_test)

    # predictions

    confusion_n = confusion_matrix(yn_test, predictn_test_opt)
    print('Confusion Matrix (predictor lables)')
    print(confusion_n)
    print('Accuracy: {:.2f}'.format(accuracy_score(yn_test, predictn_test_opt)))

    ml_new_label = rf_best_pred_random.predict(raw_data)

    predictor_data['ml_new_label'] = ml_new_label

    # --------------------------------------------

    return predictor_data
