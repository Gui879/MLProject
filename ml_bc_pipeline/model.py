import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from utils import CustomScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, average_precision_score, precision_recall_curve, roc_curve, roc_auc_score

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
    dummies = list(training.select_dtypes(include=["category", "object"]).columns)
    if not dummies:
        pipeline = Pipeline([("std_scaler", StandardScaler()), ("mlpc", MLPClassifier(random_state=seed))])
    else:
        filt = ~ training.loc[:, training.columns != "Response"].columns.isin(dummies)
        continuous_idx = np.arange(0, len(filt))[filt]
        not_filt = [not i for i in filt]
        dummies_idx = np.arange(0, len(filt))[not_filt]
        pipeline = Pipeline([("std_scaler", CustomScaler(continuous_idx, dummies_idx)), ("mlpc", MLPClassifier(random_state=seed))])

    clf_gscv = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring=make_scorer(average_precision_score))
    clf_gscv.fit(training.loc[:, (training.columns != "Response")].values, training["Response"].values)
    
    return clf_gscv

def assess_generalization_auroc(estimator, unseen):
    y_score = estimator.predict_proba(unseen.loc[:, unseen.columns != "Response"].values)[:, 1]
    fpr, tpr, thresholds = roc_curve(unseen["Response"], y_score)
    auc = roc_auc_score(unseen["Response"], y_score, average="weighted")

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, marker='.', label=" (AUROC (unseen) {:.2f}".format(auc) + ")")
    plt.plot([0, 1], [0.5, 0.5], 'k--')
    plt.xlabel('Recall (unseen)')
    plt.ylabel('Precision (unseen)')
    plt.title('PR curve on unseen data')
    plt.legend(loc='best', title="Models")
    plt.show()

    return auc