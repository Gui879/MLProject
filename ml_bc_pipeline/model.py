import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, average_precision_score, precision_recall_curve, roc_curve, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from bayes_opt import BayesianOptimization
from gplearn.genetic import SymbolicRegressor,SymbolicClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from ml_bc_pipeline.data_preprocessing import Processor
from ml_bc_pipeline.feature_engineering import FeatureEngineer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events


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
    clf_gscv = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring=make_scorer(profit))
    clf_gscv.fit(training.loc[:, (training.columns != "Response")].values, training["Response"].values)
    return clf_gscv

def bayes_optimization_MLP(training, param_grid, seed, cv=5):
    #Bayes
    n_param_grid = {}
    for key, value in param_grid.items():
        key = key.replace("mlpc" + '__', '')
        n_param_grid[key] = (0, len(value)-1)

    def ob_function(hidden_layer_sizes, learning_rate_init):
        skf = StratifiedKFold(n_splits=cv, shuffle=True)
        for train_index, test_index in skf.split(training.loc[:, (training.columns != "Response")].values,training["Response"].values):
            train = training.iloc[train_index]
            test = training.iloc[test_index]

            hidden_layer_sizes = param_grid['mlpc__hidden_layer_sizes'][int(round(hidden_layer_sizes,0))]
            learning_rate_init = param_grid['mlpc__learning_rate_init'][int(round(learning_rate_init,0))]
            model  = MLPClassifier(random_state=seed, hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=learning_rate_init)
            model.fit(train.loc[:, (train.columns != "Response")].values, train["Response"].values)
            y_pred = model.predict(test.drop('Response',axis = 1))
            return profit(test['Response'],y_pred)


    b_optimizer = BayesianOptimization(f=ob_function, pbounds=n_param_grid, random_state=1)
    b_optimizer.maximize(n_iter = 300, init_points = 100)

def decision_tree(training, param_grid, seed, cv=5):

    pipeline = Pipeline([("dt", DecisionTreeClassifier(random_state=seed))])

    clf_gscv = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring=make_scorer(profit))
    print(training.loc[:, training.columns != "Response"].values.shape, training["Response"].values.shape)
    clf_gscv.fit(training.loc[:, training.columns != "Response"].values, training["Response"].values)

    return clf_gscv

def naive_bayes(training, param_grid, seed=None, cv=5):

    pipeline = Pipeline([("nb", ComplementNB())])

    clf_gscv = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring=make_scorer(profit))
    clf_gscv.fit(training.loc[:, training.columns != "Response"].values, training["Response"].values)

    return clf_gscv

def logistic_regression(training, param_grid, seed, cv=5):

    pipeline = Pipeline([ ("lr", LogisticRegression(random_state=seed))])

    clf_gscv = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring=make_scorer(profit))
    clf_gscv.fit(training.loc[:, training.columns != "Response"].values, training["Response"].values)
    print(type(clf_gscv))

    return clf_gscv

def bo_logistic_regression(training, param_grid, seed, cv= 5):
    logger = JSONLogger(path="./logs.json")


    n_param_grid = {}
    for key, value in param_grid.items():
        key = key.replace("lr" + '__', '')
        if type(value) == type(tuple([0,0])):
            n_param_grid[key] = value
            print(value)
        else:
            n_param_grid[key] = (0, 1)

    def ob_function(C):
        skf = StratifiedKFold(n_splits=cv, shuffle=True)
        for train_index, test_index in skf.split(training.loc[:, (training.columns != "Response")].values,training["Response"].values):
            train = training.iloc[train_index]
            test = training.iloc[test_index]
            pr = Processor(train, test, seed)
            fe = FeatureEngineer(pr.training, pr.unseen, seed)
            model  = LogisticRegression(random_state=seed, C = C, max_iter=200,)
            model.fit(fe.training.loc[:, (fe.training.columns != "Response")].values, fe.training["Response"].values)
            y_pred = model.predict(fe.unseen.drop('Response',axis = 1))
            return profit(test['Response'],y_pred)


    b_optimizer = BayesianOptimization(f=ob_function, pbounds=n_param_grid, random_state=1,)
    b_optimizer.subscribe(Events.OPTMIZATION_STEP, logger)
    b_optimizer.maximize(n_iter = 100, init_points = 50)
    return b_optimizer

def extraTreesClassifier(training, param_grid, seed, cv = 5):

    pipeline = Pipeline([ ("xtree",  ExtraTreesClassifier(random_state = seed))])

    clf_gscv = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring=make_scorer(profit))
    clf_gscv.fit(training.loc[:, training.columns != "Response"].values, training["Response"].values)

    return clf_gscv

def gp_grid_search(training, param_grid, seed, cv = 5):
    pipeline = Pipeline([("gp",SymbolicClassifier( random_state = seed))])

    clf_gscv = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring=make_scorer(profit))
    clf_gscv.fit(training.loc[:, training.columns != "Response"].values, training["Response"].values)

    return clf_gscv

def gp(training, param_grid, seed, cv=5):

    pipeline = Pipeline([ ("gp", SymbolicClassifier(random_state=seed))])
    print("gp>>>",training.shape)
    clf_gscv = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring=make_scorer(profit))
    clf_gscv.fit(training.loc[:, training.columns != "Response"].values, training["Response"].values)

    return clf_gscv

def adaboost(training,seed):
    clf = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=seed)
    clf.fit(training.loc[:, training.columns != "Response"].values, training["Response"].values)
    return  clf

def gradientBoosting(training, param_grid, seed,cv = 5):
    pipeline = Pipeline([("gr", GradientBoostingClassifier(max_features=None, max_depth = 2, random_state = seed))])

    clf_gscv = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring=make_scorer(profit))
    clf_gscv.fit(training.loc[:, training.columns != "Response"].values, training["Response"].values)

    return clf_gscv

def ensemble(training, classifiers):

    clf = VotingClassifier(estimators=list(classifiers.items()), voting='soft')
    clf.fit(training.loc[:, training.columns != "Response"].values, training["Response"].values)
    return clf

def xgboost(training,param_grid, seed, cv = 5):
    pipeline = Pipeline([ ("xg", XGBClassifier(random_state=seed))])

    xg_gscv = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring=make_scorer(profit))
    xg_gscv.fit(training.loc[:, training.columns != "Response"].values, training["Response"].values)
    return xg_gscv


def assess_generalization_auroc(estimator, unseen, print_graph):
    print('shape >>> ', unseen.shape)
    y_score = estimator.predict_proba(unseen.loc[:, unseen.columns != "Response"].values)[:, 1]
    fpr, tpr, thresholds = roc_curve(unseen["Response"], y_score)

    stats = {}

    best_threshold, best_profit_ratio, best_profit = profit_curve(unseen["Response"], y_score, print_graph)
    stats['best_threshold'] = best_threshold
    stats['best_profit_ratio'] = best_profit_ratio
    stats['best_profit'] = best_profit

    predicted = [0 if v < best_threshold else 1 for v in y_score]

    report = classification_report(unseen["Response"], predicted, output_dict=True)

    print(unseen["Response"],predicted)

    #print(classification_report(unseen["Response"], predicted))


    recall_ = recall_score(unseen["Response"], predicted)
    f1_score_ = f1_score(unseen["Response"], predicted)
    precision_ = precision_score(unseen["Response"], predicted)


    for key in report.keys():
        for key2 in report[key].keys():
            stats[key+'_'+key2] = report[key][key2]

    stats['recall'] = recall_
    stats['precision'] = precision_
    stats['f1_score'] = f1_score_

    #print('\n\nF1 Score >>> ', f1_score_)
    #print('Recall >>> ', recall_)
    #print('Precision >>> ', precision_)

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

def profit_curve(y_true, y_score, print_graph):
    thresholds, c = np.arange(0, 1, 0.025), 1
    revenue_answer, expense_answer = 11, 3

    revenues = []

    for t in thresholds:
        y_pred = [0 if v < t else 1 for v in y_score]
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.reshape(4)
        revenue = tp * revenue_answer
        expenses = (tp+fp) * expense_answer
        net_revenue = revenue - expenses
        revenues.append(net_revenue)

    if print_graph:
        plt.figure(figsize=(5, 5))
        plt.plot(thresholds, revenues, marker='.', label="mlp")
        plt.plot([0, 1], [0, 0], 'k--')
        plt.xlabel('\"Probability\" threshold')
        plt.ylabel("Net Revenue")
        plt.title('Profit curves on unseen data')
        plt.legend(loc='best', title="Models")
        plt.show()

    t = thresholds[np.argmax(revenues)]
    best_revenue = np.max(revenues)
    total_revenue = np.sum(y_true) * (revenue_answer - expense_answer)
    revenue_ratio = best_revenue/total_revenue
    return t, revenue_ratio, best_revenue

def profit(y_true, y_score):
    thresholds, c = np.arange(0, 1, 0.025), 1
    revenue_answer, expense_answer = 11, 3

    revenues = []

    for t in thresholds:
        y_pred = [0 if v < t else 1 for v in y_score]
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.reshape(4)
        revenue = tp * revenue_answer
        expenses = (tp+fp) * expense_answer
        net_revenue = revenue - expenses
        revenues.append(net_revenue)

    best_revenue = np.max(revenues)
    total_revenue = np.sum(y_true) * (revenue_answer - expense_answer)
    revenue_ratio = best_revenue/total_revenue
    return revenue_ratio

def svc(training, param_grid, seed, cv=5):
    pipeline = Pipeline([("svc", SVC(random_state=seed))])

    clf_gscv = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring=make_scorer(profit))
    clf_gscv.fit(training.loc[:, training.columns != "Response"].values, training["Response"].values)

def cluster_model(training, unseen, classifiers, seed, cv=5):

    Sum_of_squared_distances = []
    K = range(1, 15)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(training.loc[:, training.columns != "Response"].values)
        Sum_of_squared_distances.append(km.inertia_)

    #plt.plot(K, Sum_of_squared_distances, 'bx-')
    #plt.xlabel('k')
    #plt.ylabel('Sum_of_squared_distances')
    #plt.title('Elbow Method For Optimal k')
    #plt.show()
    points = list(zip(K,Sum_of_squared_distances))
    print(points)
    slopes = []
    for i in range(len(points)-2):
        slopea = (points[i][1] - points[i+1][1])/(points[i][0]-points[i+1][0])
        slopeb = (points[i+1][1] - points[i+2][1])/(points[i+1][0]-points[i+2][0])
        slopes.append(slopea-slopeb)
    n_clusters_ = np.argmax(slopes) +2
    #print('Best number of clusters: ')
    #n_clusters_ = int(n_clusters_)

    km = KMeans(n_clusters=n_clusters_)
    km = km.fit(training.loc[:, training.columns != "Response"].values)

    training['label'] = km.predict(training.loc[:, training.columns != "Response"].values)
    unseen['label'] = km.predict(unseen.loc[:, unseen.columns != "Response"].values)

    while np.sum(np.array([np.sum(training['label']==label) for label in range(n_clusters_)])<10)>0:
        n_clusters_ = n_clusters_-1
        km = KMeans(n_clusters=n_clusters_)
        km = km.fit(training.loc[:, training.columns != "Response"].values)

        training['label'] = km.predict(training.loc[:, training.columns != "Response"].values)
        unseen['label'] = km.predict(unseen.loc[:, unseen.columns != "Response"].values)


    clf_to_label = {}

    for label in range(n_clusters_):
        best_clf = None
        best_profit = 0
        print('n_clusters >>> ',label)
        for clf in classifiers.items():
            best_estimator = clf[1]['model'](training[training['label']==label].loc[:, training.columns!='label'], clf[1]['params'], seed)
            if assess_generalization_auroc(best_estimator.best_estimator_, unseen[unseen['label']==label].loc[:, unseen.columns!='label'], False)[1]['best_profit_ratio']>best_profit:
                best_profit = assess_generalization_auroc(best_estimator.best_estimator_, unseen[unseen['label']==label].loc[:, unseen.columns!='label'], False)[1]['best_profit_ratio']
                best_clf = best_estimator.best_estimator_

        clf_to_label[label] = best_clf

    return km, clf_to_label






