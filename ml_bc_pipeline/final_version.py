import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from ml_bc_pipeline.data_loader import Dataset
from ml_bc_pipeline.data_preprocessing import Processor
from ml_bc_pipeline.feature_engineering import FeatureEngineer
from ml_bc_pipeline.model import gradientBoosting, grid_search_MLP, assess_generalization_auroc, decision_tree, \
    naive_bayes, logistic_regression, xgboost, ensemble, adaboost, extraTreesClassifier, gp_grid_search, gp, svc, \
    cluster_model,bo_logistic_regression

from datetime import datetime
from os import listdir
import json
from sklearn.utils import class_weight


def main():
    # ===========================
    # PARAMETERS
    # ===========================

    cv_splits = 5
    kfold_simple = False
    stratified_kfold = True

    test_version = len(listdir('Logs'))

    global log
    log = pd.DataFrame()

    global pipeline
    pipeline = {}

    def report(best_estimator_, test, best_params_=None, model_name="None", print_graph=False):
        auprc, stats = assess_generalization_auroc(best_estimator_, test, print_graph)
        stats['model_type'] = model_name
        stats['params'] = best_params_
        stats['auroc'] = auprc
        global log
        if log.shape[1] == 0:
            log = pd.DataFrame([stats.values()], columns=stats.keys())
        else:
            log = log.append(pd.DataFrame([stats.values()], columns=stats.keys()))
        print("AUROC: {:.2f}".format(auprc))

    def calculate_averages(variable_list):
        global log
        data = log
        models = data['model_type'].unique()
        columns_ = ['model'] + variable_list
        print(models)
        averages = pd.DataFrame()
        for model in models:
            var_averages = [model]
            for variable in variable_list:
                var_averages.append(np.average(data[data['model_type'] == model][variable]))
            if averages.shape[1] == 0:
                averages = pd.DataFrame([var_averages], columns=columns_)
            else:
                averages = averages.append(pd.DataFrame([var_averages], columns=columns_))
        return averages

    # +++++++++++++++++ 1) load and prepare the data
    ds = Dataset("ml_project1_data.xlsx").rm_df
    # +++++++++++++++++ 2) split into train and unseen
    DF_train, DF_unseen = train_test_split(ds.copy(), test_size=0.2, stratify=ds["Response"], random_state=0)

    #EVALUATION
    # +++++++++++++++++ 3) preprocess, based on train
    pr = Processor(DF_train.copy(), DF_unseen.copy(), 0)
    pipeline['preprocessing'] = pr.report

    # +++++++++++++++++ 4) feature engineering
    fe = FeatureEngineer(pr.training, pr.unseen, 0)
    pipeline['feature_engineering'] = fe.report


    #####INSERT MODELS TO BE EVALUATED - OPTIMIZE PARAMETERS

    # =====================================
    # NEURAL NETWORK
    # =====================================

    mlp_param_grid = {'mlpc__hidden_layer_sizes': [(3), (6), (3, 3), (5, 5)],
                      'mlpc__learning_rate_init': [0.001, 0.01]}

    mlp_gscv = grid_search_MLP(fe.training, mlp_param_grid, 0)
    print("Best parameter set: ", mlp_gscv.best_params_)
    report(mlp_gscv.best_estimator_, fe.unseen, mlp_gscv.best_params_, grid_search_MLP.__name__)

    #LOGISTIC REGRESSSION WITH BAYESIAN OPTIMIZATION

    logr_param_grid = {'lr__penalty': ['l2'],
                       'lr__C': np.logspace(-4, 4, 20),
                       'lr__solver': ['liblinear'],
                       'lr__max_iter':[200]}

    logr_gscv = logistic_regression(fe.training, logr_param_grid, 0)

    #XGBOOST
    xg_param_grid = {'xg__learning_rate':[0.1,0.5,1],
                     'xg__max_depth':[3,5],
                     'xg__n_estimators':[200]}

    xgb = xgboost(fe.training, xg_param_grid, 0)

    #GP
    gp_param_grid = {'gp__generations':[100],
                     'gp__tournament_size':[20],
                     'gp__population_size':[1000]}

    gp_gscv = gp(fe.training, gp_param_grid, 0)


    # Change partition
    if kfold_simple:
        skf = KFold(n_splits=cv_splits, shuffle=True)
    elif stratified_kfold:
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True)
    counter = 0
    for train_index, test_index in skf.split(DF_train.loc[:, DF_train.columns != "Response"], DF_train['Response']):
        counter = counter + 1
        print('###########################' + str(counter) + '###########################')
        train = DF_train.copy().iloc[train_index]
        test = DF_train.copy().iloc[test_index]
        print('split')
        print("After Split:", train.shape, test.shape)

        pr = Processor(train, test, 0)
        pipeline['preprocessing'] = pr.report
        print('processor')
        # +++++++++++++++++ 4) feature engineering

        fe = FeatureEngineer(pr.training, pr.unseen, 0)
        print("after fe:", fe.training.shape, fe.unseen.shape)
        pipeline['feature_engineering'] = fe.report
        print('feature_engineering')


        ##### TRAIN MODELS
        # =====================================
        # NEURAL NETWORK
        # =====================================

        mlp_gscv.best_estimator_ = mlp_gscv.best_estimator_.fit(fe.training.loc[:, fe.training.columns != "Response"].values, fe.training["Response"].values)
        #print("Best parameter set: ", mlp_gscv.best_params_)
        report(mlp_gscv.best_estimator_, fe.unseen, mlp_gscv.best_params_, grid_search_MLP.__name__)


        # =====================================
        # LOGISTIC REGRESSION
        # =====================================

        logr_gscv.best_estimator_ = logr_gscv.best_estimator_.fit(fe.training.loc[:, fe.training.columns != "Response"].values, fe.training["Response"].values)
        print("Best parameter set: ", logr_gscv.best_params_)
        report(logr_gscv.best_estimator_, fe.unseen, logr_gscv.best_params_,logistic_regression.__name__)

        # =====================================
        # XGBOOST
        # =====================================

        xgb.best_estimator_ = xgb.best_estimator_.fit(fe.training.loc[:, fe.training.columns != "Response"].values, fe.training["Response"].values)
        report(xgb.best_estimator_, fe.unseen, xgb.best_params_,'xgb')
        print("Best parameter set: ", xgb.best_estimator_)

        # =====================================
        # GENETIC PROGRAMMING
        # =====================================

        gp_gscv.best_estimator_ = gp_gscv.best_estimator_.fit(fe.training.loc[:, fe.training.columns != "Response"].values, fe.training["Response"].values)
        report(gp_gscv.best_estimator_, fe.unseen, gp_gscv.best_params_,'gp')
        print("Best parameter set: ", gp_gscv.best_params_)

        # =====================================
        # ENSEMBLE
        # =====================================

        classifiers = {
            'xg': xgb.best_estimator_,
            'mlp': mlp_gscv.best_estimator_,
            'gp': gp_gscv.best_estimator_,
            'lr': logr_gscv.best_estimator_,
        }

        en_gscv = ensemble(fe.training, classifiers)
        report(en_gscv.best_estimator_, fe.unseen, en_gscv.best_params_, 'en')
        print("Best parameter set: ", en_gscv.best_params_)

    log.to_csv('Logs/' + 'version' + str(test_version) + '_' + str(0) + '.csv')
    with open('Pipelines/version' + str(test_version) + '_' + str(0) + '.txt', 'w') as file:
        file.write(json.dumps(pipeline))

    averages = calculate_averages(
        ['auroc', 'precision', 'recall', 'f1_score', 'best_threshold', 'best_profit_ratio', 'best_profit'])
    averages.to_csv('Averages/version' + str(test_version) + '_' + str(0) + '.csv')


if __name__ == "__main__":
    main()