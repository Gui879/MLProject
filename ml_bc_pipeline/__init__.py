import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ml_bc_pipeline.data_loader import Dataset
from ml_bc_pipeline.data_preprocessing import Processor
from ml_bc_pipeline.feature_engineering import FeatureEngineer
from ml_bc_pipeline.model import grid_search_MLP, assess_generalization_auroc, decision_tree, naive_bayes, logistic_regression, ensemble
from datetime import datetime

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

    '''
    # get top n features
    criteria, n_top = "chisq", 9
    DF_train_top, DF_unseen_top = fe.get_top(criteria="chisq", n_top=n_top)

    #+++++++++++++++++ 5) modelling
    estimators = []
    # =====================================
    # NEURAL NETWORK
    # =====================================

    mlp_param_grid = {'mlpc__hidden_layer_sizes': [(3), (6), (3, 3), (5, 5)],
                      'mlpc__learning_rate_init': [0.001, 0.01]}

    mlp_gscv = grid_search_MLP(DF_train, mlp_param_grid, seed)
    print("Best parameter set: ", mlp_gscv.best_params_)
    # pd.DataFrame.from_dict(mlp_gscv.cv_Response_).to_excel("D:\\PipeLines\\project_directory\\data\\mlp_gscv.xlsx")
    estimators.append(mlp_gscv.best_estimator_)
    # =====================================
    # DECISION TREE
    # =====================================

    dt_param_grid = {'dt__criterion': ["entropy", "gini"],
                  'dt__max_features': [3, 5, 7, None],
                  "dt__max_depth": [3, 4, 5, 6],
                  "dt__min_samples_split": [30, 50, 70]}
    dt_gscv = decision_tree(DF_train, dt_param_grid, seed)
    print("Best parameter set: ", dt_gscv.best_params_)
    estimators.append(dt_gscv.best_estimator_)
    '''
    # =====================================
    # NAIVE BAYES
    # =====================================

    # nb_param_grid = {'nb__alpha': [0, 0.25, 0.5, 0.75, 1]}  # i'm not sure about this parameter
    # nb_gscv = naive_bayes(fe.training, nb_param_grid, seed)
    # print("Best parameter set: ", nb_gscv.best_params_)
    # #estimators.append(nb_gscv.best_estimator_)


    '''
    # =====================================
    # LOGISTIC REGRESSION
    # =====================================
                                                
    logr_param_grid = {'lr__penalty': ['l1', 'l2'],
         'lr__C': np.logspace(-4, 4, 20),
         'lr__solver': ['liblinear']}
    logr_gscv = logistic_regression(DF_train, logr_param_grid, seed)
    print("Best parameter set: ", logr_gscv.best_params_)
    estimators.append(logr_gscv.best_estimator_)
    
    # =====================================
    # ENSEMBLE
    # =====================================

    classifiers = {
        'neural_net':  mlp_gscv.best_estimator_,
        'dt': dt_gscv.best_estimator_,
        'nb': nb_gscv.best_estimator_,
        'logr': logr_gscv.best_estimator_,

    }
    
    ensemble_estimator = ensemble(DF_train, classifiers, seed)
    #+++++++++++++++++ 6) retraining & assessment of generalization ability
    estimators.append(ensemble_estimator)
    '''
    #for estimator in estimators:

    def run_models(model, n=5, pring_graph=False):
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

            nb_param_grid = {'nb__alpha': [0, 0.25, 0.5, 0.75, 1]}  # i'm not sure about this parameter
            classifier = model(fe.training, nb_param_grid, seed)
            print("Best parameter set: ", classifier.best_params_)
            # estimators.append(nb_gscv.best_estimator_)
            auprc, stats = assess_generalization_auroc(classifier.best_estimator_, fe.unseen, pring_graph)
            stats['model_type'] = model.__name__
            stats['params'] = classifier.best_params_
            stats['auroc'] = auprc
            log.append(stats.values())

            print("AUROC: {:.2f}".format(auprc))

        log = pd.DataFrame(log, columns=stats.keys())
        log.to_csv('Logs/'+model.__name__+'_'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")+'.csv')

    run_models(naive_bayes)

if __name__ == "__main__":
    main()