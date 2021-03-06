import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from Code.ml_bc_pipeline.data_loader import Dataset
from Code.ml_bc_pipeline.data_preprocessing import Processor
from Code.ml_bc_pipeline.feature_engineering import FeatureEngineer
from Code.ml_bc_pipeline.model import grid_search_MLP, assess_generalization_auroc, bayes_optimization_MLP
from sklearn.datasets import make_classification, make_moons
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import make_scorer, average_precision_score, precision_recall_curve, roc_curve, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix, classification_report
from gplearn.genetic import SymbolicRegressor,SymbolicClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from model import profit
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from model import profit
from sklearn.model_selection import StratifiedKFold
def main():

    seed = 0
    np.random.seed(seed)
    df = Dataset('ml_project1_data.xlsx').rm_df
    y = df['Response']
    X = df.drop(columns='Response')
    training, testing, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)





    training['Response'] = y_train
    testing['Response'] = y_test
    pr = Processor(training,testing,seed = 0)
    fe = FeatureEngineer(pr.training,pr.unseen,seed = 0)
    training = fe.training
    testing = fe.unseen
    est = SymbolicClassifier(generations = 200, random_state = 0)
    est.fit(training.drop('Response',axis = 1), training['Response'])
    assess_generalization_auroc(est,testing,True)
    y_pred = est.predict_proba(testing.drop('Response',axis = 1))[:,1]
    y_true = testing['Response']
    print(profit(y_true, y_pred))

    #+++++++++++++++++ 5) modelling
    #Create Optimizer
    '''
    mlp_param_grid = {'mlpc__hidden_layer_sizes': [(3), (6), (3, 3), (5, 5)],
                      'mlpc__learning_rate_init': [0.001, 0.01]}
    mlp_gscv = bayes_optimization_MLP(fe.training,mlp_param_grid, cv = 5,seed = 0)
    #mlp_gscv.fit(training.loc[:, (training.columns != "Response")].values, training["Response"].values)
    print("Best parameter set: ", mlp_gscv.best_params_)
    # pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("D:\\PipeLines\\project_directory\\data\\mlp_gscv.xlsx")

    #+++++++++++++++++ 6) retraining & assessment of generalization ability
    #auprc,precision, recall = assess_generalization_auroc(mlp_gscv.best_estimator_, testing)
    #print("AUPRC: {:.2f}".format(auprc))
    '''

    plt.show()

if __name__ == "__main__":
    main()