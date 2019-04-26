import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from Code.ml_bc_pipeline.data_loader import Dataset
from Code.ml_bc_pipeline.data_preprocessing import Processor
from Code.ml_bc_pipeline.feature_engineering import FeatureEngineer
from Code.ml_bc_pipeline.model import grid_search_MLP, assess_generalization_auroc
from sklearn.datasets import make_classification, make_moons
from mlxtend.plotting import plot_decision_regions

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

    '''
    training = pd.read_csv('train_data.csv')
    testing = pd.read_csv('test_data.csv')

    #+++++++++++++++++ 5) modelling
    mlp_param_grid = {'mlpc__hidden_layer_sizes': [(3), (6), (3, 3), (5, 5)],
                      'mlpc__learning_rate_init': [0.001, 0.01]}

    mlp_gscv = grid_search_MLP(training, mlp_param_grid, seed)
    print("Best parameter set: ", mlp_gscv.best_params_)
    # pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("D:\\PipeLines\\project_directory\\data\\mlp_gscv.xlsx")

    #+++++++++++++++++ 6) retraining & assessment of generalization ability
    auprc,precision, recall = assess_generalization_auroc(mlp_gscv.best_estimator_, testing)
    print("AUPRC: {:.2f}".format(auprc))


    plt.show()
    '''
if __name__ == "__main__":
    main()