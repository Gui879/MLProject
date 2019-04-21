import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ml_bc_pipeline.data_loader import Dataset
from ml_bc_pipeline.data_preprocessing import Processor
from ml_bc_pipeline.feature_engineering import FeatureEngineer
from ml_bc_pipeline.model import grid_search_MLP, assess_generalization_auroc, decision_tree, naive_bayes, logistic_regression, xgboost, ensemble
from datetime import datetime
from os import listdir

def main():

    #this is an important project :D

    # #+++++++++++++++++ 1) load and prepare the data
    # ds = Dataset("ml_project1_data.xlsx").rm_df
    # #+++++++++++++++++ 2) split into train and unseen
    # seed = 0
    # DF_train, DF_unseen = train_test_split(ds.copy(), test_size=0.2, stratify=ds["Response"], random_state=seed)
    # #+++++++++++++++++ 3) preprocess, based on train
    # pr = Processor(DF_train, DF_unseen)
    # #+++++++++++++++++ 4) feature engineering
    # fe = FeatureEngineer(pr.training, pr.unseen)


    # get top n features
    #criteria, n_top = "chisq", 9
    #DF_train_top, DF_unseen_top = fe.get_top(criteria="chisq", n_top=n_top)

    #+++++++++++++++++ 5) modelling
    #estimators = []

    #for estimator in estimators:

    #models = []


    def run_models(model, param_grid, n=5,pring_graph=False):
        log = []

        for seed in range(n):
            # +++++++++++++++++ 1) load and prepare the data
            ds = Dataset("ml_project1_data.xlsx").rm_df
            # +++++++++++++++++ 2) split into train and unseen

            DF_train, DF_unseen = train_test_split(ds.copy(), test_size=0.2, stratify=ds["Response"], random_state=seed)
            # +++++++++++++++++ 3) preprocess, based on train
            pr = Processor(DF_train, DF_unseen)
            # +++++++++++++++++ 4) feature engineering
            fe = FeatureEngineer(pr.training, pr.unseen)
            if param_grid:
                classifier = model(fe.training, param_grid, seed)
                print("Best parameter set: ", classifier.best_params_)
                estimator = classifier.best_estimator_
            else:
                estimator = model
            # estimators.append(nb_gscv.best_estimator_)
            auprc, stats = assess_generalization_auroc(estimator, fe.unseen, pring_graph)
            stats['model_type'] = model.__name__
            stats['params'] = classifier.best_params_
            stats['auroc'] = auprc
            log.append(stats.values())

            print("AUROC: {:.2f}".format(auprc))

        log = pd.DataFrame(log, columns=stats.keys())
        log.to_csv('Logs/'+model.__name__+'_'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")+'.csv')
    global log
    log = pd.DataFrame()
    def report(best_estimator_, best_params_ = None,model_name = "None", print_graph = False):
        auprc, stats = assess_generalization_auroc(best_estimator_, fe.unseen, print_graph)
        stats['model_type'] = model_name
        stats['params'] = best_params_
        stats['auroc'] = auprc
        global log
        if log.shape[1] == 0:
            log = pd.DataFrame([stats.values()], columns = stats.keys())
        else:
            log = log.append(pd.DataFrame([stats.values()], columns = stats.keys()))
        print("AUROC: {:.2f}".format(auprc))

    for seed in range(5):
        # +++++++++++++++++ 1) load and prepare the data
        ds = Dataset("ml_project1_data.xlsx").rm_df
        # +++++++++++++++++ 2) split into train and unseen

        DF_train, DF_unseen = train_test_split(ds.copy(), test_size=0.2, stratify=ds["Response"], random_state=seed)
        # +++++++++++++++++ 3) preprocess, based on train
        pr = Processor(DF_train, DF_unseen)
        # +++++++++++++++++ 4) feature engineering
        fe = FeatureEngineer(pr.training, pr.unseen)

        # =====================================
        # NEURAL NETWORK
        # =====================================

        mlp_param_grid = {'mlpc__hidden_layer_sizes': [(3), (6), (3, 3), (5, 5)],
                          'mlpc__learning_rate_init': [0.001, 0.01]}

        mlp_gscv = grid_search_MLP(fe.training, mlp_param_grid, seed)
        print("Best parameter set: ", mlp_gscv.best_params_)
        report(mlp_gscv.best_estimator_,mlp_gscv.best_params_, grid_search_MLP.__name__)

        # =====================================
        # DECISION TREE
        # =====================================

        dt_param_grid = {'dt__criterion': ["entropy", "gini"],
                         'dt__max_features': [3, 5, 7, None],
                         "dt__max_depth": [3, 4, 5, 6],
                         "dt__min_samples_split": [30, 50, 70]}
        dt_gscv = decision_tree(fe.training, dt_param_grid, seed)
        print("Best parameter set: ", dt_gscv.best_params_)
        report(dt_gscv.best_estimator_,dt_gscv.best_params_,decision_tree.__name__)

        # =====================================
        # NAIVE BAYES
        # =====================================

        nb_param_grid = {'nb__alpha': [0, 0.25, 0.5, 0.75, 1]}  # i'm not sure about this parameter
        nb_gscv = naive_bayes(fe.training, nb_param_grid, seed)
        print("Best parameter set: ", nb_gscv.best_params_)
        report(nb_gscv.best_estimator_,nb_gscv.best_params_,naive_bayes.__name__)

        # =====================================
        # LOGISTIC REGRESSION
        # =====================================

        logr_param_grid = {'lr__penalty': ['l1', 'l2'],
                           'lr__C': np.logspace(-4, 4, 20),
                           'lr__solver': ['liblinear']}
        logr_gscv = logistic_regression(fe.training, logr_param_grid, seed)
        print("Best parameter set: ", logr_gscv.best_params_)
        report(logr_gscv.best_estimator_, logr_gscv.best_params_,logistic_regression.__name__)

        # =====================================
        # XGBoost
        # =====================================

        xgb = xgboost(fe.training, fe.unseen, seed)
        report(xgb,model_name=xgboost.__name__)

        # =====================================
        # ENSEMBLE
        # =====================================

        classifiers = {
            'neural_net': mlp_gscv.best_estimator_,
            'dt': dt_gscv.best_estimator_,
            'nb': nb_gscv.best_estimator_,
            'logr': logr_gscv.best_estimator_,

        }

        ensemble_estimator = ensemble(fe.training, classifiers, seed)
        report(ensemble_estimator, classifiers.keys(), ensemble.__name__)

    test_version = len(listdir('Logs'))
    log.to_csv('Logs/' + 'version' + str(test_version)+ '_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + '.csv')

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
                averages = pd.DataFrame([var_averages], columns = columns_)
            else:
                averages = averages.append(pd.DataFrame([var_averages], columns = columns_))
        return averages

    averages = calculate_averages(['auroc','precision','recall','f1_score'])
    averages.to_csv('Averages/version' + str(test_version) + '_averages.csv')
if __name__ == "__main__":
    main()