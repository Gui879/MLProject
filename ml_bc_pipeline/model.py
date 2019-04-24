import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, average_precision_score, precision_recall_curve, roc_curve, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import xgboost as xgb

def grid_search_MLP(training, param_grid, seed, cv=5):
    """ Multi-layer Perceptron classifier hyperparameter estimation using grid search with cross-validation.

    In this function, the MLP classifier is optimized by CV, implemented through GridSearchCV function from
    sklearn. Semantically, i.e., not technically, this is performed in the following way:
     1) several models are created with different hyper-parameters (according to param_grid);
     2) their performance is assessed by means of k-fold cross-validation (k=cv):
        2) 1) for cv times, the model is trained using k-1 folds of the training data;
        2) 2) each time, the resulting model is validated on the held out (kth) part of the data;
        2) 3) the final performance is computed as the average along cv iterations.


    From theory it is known that input standardization allows an ANN perform better. For this reason, this
    function automatically embeds input standardization within hyperparameter estimation procedure. This is
    done by arranging sklearn.preprocessing.StandardScaler and sklearn.neural_network.MLPClassifier into the
    same "pipeline". The tool which allows to do so is called sklearn.pipeline.Pipeline. More specifically,
    the preprocessing module further provides a utility class StandardScaler that implements the Transformer
    API to compute the mean and standard deviation on a training set so as to be able to later reapply the
    same transformation on the testing set.
    """
    #sklearn pipeline standardizes dummmie variables which we do not want so in here a custom scaler is used

    pipeline = Pipeline([("mlpc", MLPClassifier(random_state=seed))])

    clf_gscv = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring=make_scorer(average_precision_score))
    clf_gscv.fit(training.loc[:, (training.columns != "Response")].values, training["Response"].values)

    return clf_gscv



def decision_tree(training, param_grid, seed, cv=5):

    pipeline = Pipeline([("dt", DecisionTreeClassifier(random_state=seed))])

    clf_gscv = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring=make_scorer(average_precision_score))
    clf_gscv.fit(training.loc[:, training.columns != "Response"].values, training["Response"].values)

    return clf_gscv


def naive_bayes(training, param_grid, seed, cv=5):

    pipeline = Pipeline([("nb", ComplementNB())])

    clf_gscv = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring=make_scorer(average_precision_score))
    clf_gscv.fit(training.loc[:, training.columns != "Response"].values, training["Response"].values)

    return clf_gscv

def logistic_regression(training, param_grid, seed, cv=5):

    pipeline = Pipeline([ ("lr", LogisticRegression(random_state=seed))])

    clf_gscv = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring=make_scorer(average_precision_score))
    clf_gscv.fit(training.loc[:, training.columns != "Response"].values, training["Response"].values)
    print(type(clf_gscv))

    return clf_gscv


def ensemble(training, classifiers, seed, cv=5):

    clf = VotingClassifier(estimators=list(classifiers.items()), voting='soft')
    clf.fit(training.loc[:, training.columns != "Response"].values, training["Response"].values)
    return clf

def xgboost(training, unseen, seed, cv=5):

    xgb_model = xgb.XGBClassifier().fit(training.loc[:, training.columns != "Response"].values, training["Response"].values)

    return xgb_model

def assess_generalization_auroc(estimator, unseen, print_graph):

    y_score = estimator.predict_proba(unseen.loc[:, unseen.columns != "Response"].values)[:, 1]
    predicted = estimator.predict(unseen.loc[:, unseen.columns != "Response"].values)
    fpr, tpr, thresholds = roc_curve(unseen["Response"], y_score)

    stats = {}
    report = classification_report(unseen["Response"], predicted, output_dict=True)

    print(classification_report(unseen["Response"], predicted))

    recall_ = recall_score(unseen["Response"], predicted)
    f1_score_ = f1_score(unseen["Response"], predicted)
    precision_ = precision_score(unseen["Response"], predicted)

    best_threshold, best_profit = profit_curve(unseen["Response"], y_score, print_graph)
    stats['best_threshold'] = best_threshold
    stats['best_profit'] = best_profit

    for key in report.keys():
        for key2 in report[key].keys():
            stats[key+'_'+key2] = report[key][key2]

    stats['recall'] = recall_
    stats['precision'] = precision_
    stats['f1_score'] = f1_score_

    print('\n\nF1 Score >>> ', f1_score_)
    print('Recall >>> ', recall_)
    print('Precision >>> ', precision_)

    auc = roc_auc_score(unseen["Response"], y_score, average="weighted")

    if print_graph:
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, marker='.', label=" (AUROC (unseen) {:.2f}".format(auc) + ")")
        plt.plot([0, 1], [0.5, 0.5], 'k--')
        plt.xlabel('Recall (unseen)')
        plt.ylabel('Precision (unseen)')
        plt.title('PR curve on unseen data (' + estimator.__class__.__name__ + ')')
        plt.legend(loc='best', title="Models")
        plt.show()

    return auc, stats

def profit_curve(y_unseen, y_prob, print_graph):
    thresholds, c = np.arange(0, 1, 0.025), 1
    revenue_answer, expense_answer = 11, 3

    revenues = []
    dict_thresholds = {}

    plt.figure(figsize=(5, 5))
    i = 0
    for t in thresholds:
        y_pred = [0 if v < t else 1 for v in y_prob]
        cm = confusion_matrix(y_unseen, y_pred)
        revenue = cm[1][1] * revenue_answer
        expenses = cm[:, 1].sum() * expense_answer
        net_revenue = revenue - expenses
        revenues.append(net_revenue)

    if print_graph:
        plt.plot(thresholds, revenues, marker='.', label="mlp")
        plt.plot([0, 1], [0, 0], 'k--')
        plt.xlabel('\"Probability\" threshold')
        plt.ylabel("Net Revenue")
        plt.title('Profit curves on unseen data')
        plt.legend(loc='best', title="Models")
        plt.show()

    t = thresholds[np.argmax(revenues)]
    best_revenue = revenues[np.argmax(revenues)]
    return t, best_revenue

