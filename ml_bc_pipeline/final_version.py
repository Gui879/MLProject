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
from sklearn.linear_model import LogisticRegression

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

    ds = Dataset("ml_project1_data.xlsx").rm_df
    students = Dataset('unseen_students.xlsx',True).rm_df


    #EVALUATION
    # +++++++++++++++++ 3) preprocess, based on train
    pr = Processor(ds, students, 1)
    pipeline['preprocessing'] = pr.report

    # +++++++++++++++++ 4) feature engineering
    fe = FeatureEngineer(pr.training, pr.unseen, 1)
    pipeline['feature_engineering'] = fe.report
    #Best Parameters
    C = 0.5571
    max_iter = 200
    final_model = LogisticRegression(random_state=1, C = C, max_iter=max_iter)
    final_model.fit(fe.training.loc[:, (fe.training.columns != "Response")].values, fe.training["Response"].values)
    y_pred = final_model.predict(students)
    students['Predicted'] = y_pred
    print(students)

if __name__ == "__main__":
    main()