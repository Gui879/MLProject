import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer


class FeatureEngineer:
    def __init__(self, training, unseen):
        self._rank = {}
        self.training = training
        self.unseen = unseen

        self._extract_business_features()
        self._merge_categories()
        self._generate_dummies()

    def _extract_business_features(self):
        self.dict_bt = {"BT_MntIncome": lambda df: df["Mnt"].divide(df["Income"], fill_value=0).multiply(100),
                        "BT_MntFrq": lambda df: df["Mnt"].divide(df["Frq"], fill_value=0).multiply(100)}

        for key, value in self.dict_bt.items():
            self.training[key] = value(self.training)
            self.unseen[key] = value(self.unseen)

    def _merge_categories(self):
        self.dict_merge_cat = {"Marital_Status": lambda x: 2 if x == "Widow" else (1 if x == "Divorced" else 0),
                        "Education": lambda x: 1 if x == "Master" else 0,
                        "Recomendation": lambda x: 2 if x == 6 else (1 if x == 5 else 0)}

        for key, value in self.dict_merge_cat.items():
            self.training["MC_"+key] = self.training[key].apply(value).astype('category')
            self.unseen["MC_" + key] = self.unseen[key].apply(value).astype('category')

    def _generate_dummies(self):
        features_to_enconde = ['MC_Marital_Status', 'MC_Education', 'MC_Recomendation']
        columns = ["DT_MS_Divorced", "DT_MS_Widow", "DT_E_Master", "DT_R_5", "DT_R_6"]
        idxs = [1, 2, 4, 6, 7]
        # encode categorical features from training data as a one-hot numeric array.
        enc = OneHotEncoder(handle_unknown='ignore')
        Xtr_enc = enc.fit_transform(self.training[features_to_enconde]).toarray()
        # update training data
        df_temp = pd.DataFrame(Xtr_enc[:, idxs], index=self.training.index, columns=columns)
        self.training = pd.concat([self.training, df_temp], axis=1)
        for c in columns:
            self.training[c] = self.training[c].astype('category')
        # use the same encoder to transform unseen data
        Xun_enc = enc.transform(self.unseen[features_to_enconde]).toarray()
        # update unseen data
        df_temp = pd.DataFrame(Xun_enc[:, idxs], index=self.unseen.index, columns=columns)
        self.unseen = pd.concat([self.unseen, df_temp], axis=1)
        for c in columns:
            self.unseen[c] = self.unseen[c].astype('category')

    def box_cox_transformations(self, num_features, target):
        # 1) perform feature scaling, using MinMaxScaler from sklearn
        bx_cx_scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
        X_tr_01 = bx_cx_scaler.fit_transform(self.training[num_features].values)
        X_un_01 = bx_cx_scaler.transform(self.unseen[num_features].values)
        num_features_BxCx = ["BxCxT_" + s for s in num_features]
        self.training = pd.concat([self.training.loc[:, self.training.columns != target],
                                   pd.DataFrame(X_tr_01, index=self.training.index, columns=num_features_BxCx),
                                   self.training[target]], axis=1)
        self.unseen = pd.concat([self.unseen.loc[:, self.unseen.columns != target],
                                   pd.DataFrame(X_un_01, index=self.unseen.index, columns=num_features_BxCx),
                                   self.unseen[target]], axis=1)
        # 2) define a set of transformations
        self._bx_cx_trans_dict = {"x": lambda x: x, "log": np.log, "sqrt": np.sqrt,
                      "exp": np.exp, "**1/4": lambda x: np.power(x, 0.25),
                      "**2": lambda x: np.power(x, 2), "**4": lambda x: np.power(x, 4)}
        # 3) perform power transformations on scaled features and select the best
        self.best_bx_cx_dict = {}
        for feature in num_features_BxCx:
            best_test_value, best_trans_label, best_power_trans = 0, "", None
            for trans_key, trans_value in self._bx_cx_trans_dict.items():
                # 3) 1) 1) apply transformation on training data
                feature_trans = np.round(trans_value(self.training[feature]), 4)
                if trans_key == "log":
                    feature_trans.loc[np.isfinite(feature_trans) == False] = -50
                # 3) 1) 2) bin transformed feature (required to perform Chi-Squared test)
                bindisc = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="uniform")
                feature_bin = bindisc.fit_transform(feature_trans.values.reshape(-1, 1))
                feature_bin = pd.Series(feature_bin[:, 0], index=self.training.index)
                # 3) 1) 3) obtain contingency table
                cont_tab = pd.crosstab(feature_bin, self.training[target], margins=False)
                # 3) 1) 4) compute Chi-Squared test
                chi_test_value = stats.chi2_contingency(cont_tab)[0]
                # 3) 1) 5) choose the best so far Box-Cox transformation based on Chi-Squared test
                if chi_test_value > best_test_value:
                    best_test_value, best_trans_label, best_power_trans = chi_test_value, trans_key, feature_trans
            self.best_bx_cx_dict[feature] = (best_trans_label, best_power_trans)
            # 3) 2) append transformed feature to the data frame
            self.training[feature] = best_power_trans
            # 3) 3) apply the best Box-Cox transformation, determined on training data, on unseen data
            self.unseen[feature] = np.round(self._bx_cx_trans_dict[best_trans_label](self.unseen[feature]), 4)
        self.box_cox_features = num_features_BxCx

    def rank_features_chi_square(self, continuous_flist, categorical_flist):
        chisq_dict = {}
        if continuous_flist:
            bindisc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy="uniform")
            for feature in continuous_flist:
                feature_bin = bindisc.fit_transform(self.training[feature].values[:, np.newaxis])
                feature_bin = pd.Series(feature_bin[:, 0], index=self.training.index)
                cont_tab = pd.crosstab(feature_bin, self.training["DepVar"], margins=False)
                chisq_dict[feature] = stats.chi2_contingency(cont_tab.values)[0:2]
        if categorical_flist:
            for feature in categorical_flist:
                cont_tab = pd.crosstab(self.training[feature], self.training["DepVar"], margins=False)
                chisq_dict[feature] = stats.chi2_contingency(cont_tab.values)[0:2]

        df_chisq_rank = pd.DataFrame(chisq_dict, index=["Chi-Squared", "p-value"]).transpose()
        df_chisq_rank.sort_values("Chi-Squared", ascending=False, inplace=True)
        df_chisq_rank["valid"] = df_chisq_rank["p-value"] <= 0.05
        self._rank["chisq"] = df_chisq_rank

    def print_top(self, n):
        print(self._rank.index[0:8])

    def get_top(self, criteria="chisq", n_top=10):
        input_features = list(self._rank[criteria].index[0:n_top])
        input_features.append("DepVar")
        return self.training[input_features], self.unseen[input_features]