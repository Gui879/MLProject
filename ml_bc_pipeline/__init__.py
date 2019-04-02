import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from ml_bc_pipeline.data_loader import Dataset
from ml_bc_pipeline.data_preprocessing import Processor
from ml_bc_pipeline.feature_engineering import FeatureEngineer
from ml_bc_pipeline.model import grid_search_MLP, assess_generalization_auroc


def main():

    #this is an important project :D

    #+++++++++++++++++ 1) load and prepare the data
    file_path = "ml_project1_data.xlsx"
    ds = pd.read_excel(file_path)
    ds.drop(["Dt_Customer","Education", "Marital_Status", "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Complain"],axis  = 1,inplace = True)
    #+++++++++++++++++ 2) split into train and unseen
    seed = 0
    DF_train, DF_unseen = train_test_split(ds.copy(), test_size=0.2, stratify=ds["Response"], random_state=seed)

    #+++++++++++++++++ 3) preprocess, based on train
    pr = Processor(DF_train, DF_unseen)
    '''
    #+++++++++++++++++ 4) feature engineering
    fe = FeatureEngineer(pr.training, pr.unseen)

    # apply Box-Cox transformations
    num_features = fe.training._get_numeric_data().drop(['Response'], axis=1).columns
    fe.box_cox_transformations(num_features, target="Response")

    # rank input features according to Chi-Squared
    continuous_flist = fe.box_cox_features
    categorical_flist = ['DT_MS_Divorced', 'DT_MS_Widow', 'DT_E_Master', 'DT_R_5', 'DT_R_6', "Gender"]
    fe.rank_features_chi_square(continuous_flist, categorical_flist)
    print("Ranked input features:\n", fe._rank)

    # get top n features
    criteria, n_top = "chisq", 9
    DF_train_top, DF_unseen_top = fe.get_top(criteria="chisq", n_top=n_top)
    '''
    #+++++++++++++++++ 5) modelling
    mlp_param_grid = {'mlpc__hidden_layer_sizes': [(3), (6), (3, 3), (5, 5)],
                      'mlpc__learning_rate_init': [0.001, 0.01]}

    mlp_gscv = grid_search_MLP(DF_train, mlp_param_grid, seed)
    print("Best parameter set: ", mlp_gscv.best_params_)
    # pd.DataFrame.from_dict(mlp_gscv.cv_Response_).to_excel("D:\\PipeLines\\project_directory\\data\\mlp_gscv.xlsx")

    #+++++++++++++++++ 6) retraining & assessment of generalization ability
    auprc = assess_generalization_auroc(mlp_gscv.best_estimator_, DF_unseen)
    print("AUROC: {:.2f}".format(auprc))


if __name__ == "__main__":
    main()