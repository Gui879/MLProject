import sys
import numpy as np
import pandas as pd
import datetime




from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA

class FeatureEngineer:
    def __init__(self, training, unseen):
        self._rank = {}
        self.training = training
        self.unseen = unseen
        print("First:",self.training.shape)
        self._extract_business_features()
        #self.LDA_extraction()
        print("Feature Engeneering Completed!")


    def _extract_business_features(self):

        print('BEFORE')
        print(len(self.training.columns))
        print(len(self.unseen.columns))

        for dataset in [self.training, self.unseen]:
            # TER CUIDADO, CONFIRMAR SE O NUM WEB PURCHASES TB É
            dataset["Web_Purchases_Per_Visit"] = dataset["NumWebPurchases"] / dataset["NumWebVisitsMonth"]
            dataset["Total_Purchases"] = dataset["NumWebPurchases"] + dataset["NumCatalogPurchases"] + dataset["NumStorePurchases"]
            dataset["RatioWebPurchases"] = dataset["NumWebPurchases"] / dataset["Total_Purchases"]
            dataset["RatioCatalogPurchases"] = dataset["NumCatalogPurchases"] / dataset["Total_Purchases"]
            dataset["RatioStorePurchases"] = dataset["NumStorePurchases"] / dataset["Total_Purchases"]

            dataset["Age"] = datetime.datetime.now().year - dataset["Year_Birth"]
            dataset["TotalAcceptedCampaigns"] = dataset["AcceptedCmp1"]+dataset["AcceptedCmp2"]+dataset["AcceptedCmp3"]+dataset["AcceptedCmp4"]+dataset["AcceptedCmp5"]
            # Total amount spent
            dataset["TotalMoneySpent"] = dataset["MntWines"] + dataset["MntFruits"] + dataset["MntMeatProducts"] + dataset["MntFishProducts"] + dataset["MntSweetProducts"] + dataset["MntGoldProds"]
            # Calculating the ratios of money spent per category
            dataset["RatioWines"] = dataset["MntWines"] / dataset["TotalMoneySpent"]
            dataset["RatioFruits"] = dataset["MntFruits"] / dataset["TotalMoneySpent"]
            dataset["RatioMeatProducts"] = dataset["MntMeatProducts"] / dataset["TotalMoneySpent"]
            dataset["RatioFishProducts"] = dataset["MntFishProducts"] / dataset["TotalMoneySpent"]
            dataset["RatioSweetProducts"] = dataset["MntSweetProducts"] / dataset["TotalMoneySpent"]
            dataset["RatioGoldProdataset"] = dataset["MntGoldProds"] / dataset["TotalMoneySpent"]

            # Changing income to 2 years
            dataset["Income2Years"] = dataset["Income"] * 2

            # Calculating Effort Rate
            dataset["EffortRate"] = dataset["TotalMoneySpent"] / dataset["Income2Years"]

            # All kidataset
            dataset["TotalKids"] = dataset["Teenhome"] + dataset["Kidhome"]

            # People per Household
            status = ["Together", "Married"]
            dataset["Count_Household"] = 0
            dataset["Count_Household"].loc[(dataset["Marital_Status"].isin(status))] = 2 + dataset["TotalKids"]
            dataset["Count_Household"].loc[(~dataset["Marital_Status"].isin(status))] = 1 + dataset["TotalKids"]

            # Income per person in household
            dataset["Income_Per_Person"] = dataset["Income2Years"] / dataset["Count_Household"]
            features_to_enconde = ['Education', 'Marital_Status']
            dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
            dataset.fillna(0, inplace=True)
            dataset.drop(features_to_enconde, axis=1, inplace=True)

        print('AFTER')
        print(len(self.training.columns))
        print(len(self.unseen.columns))

    def LDA_extraction(self):
        ds = self.training
        y = self.training['Response']
        clf = LinearDiscriminantAnalysis(solver="eigen")
        lda = clf.fit(ds, y)
        lda_ds = lda.transform(ds)
        lda_coef = lda.coef_
        lda_means = lda.means_
        exp_var = lda.explained_variance_ratio_
        print(lda.explained_variance_ratio_)
        return lda_ds, exp_var

    def factor_analysis_extraction(self):
        ds = self.training
        colunas = list(ds)
        fact = FactorAnalysis().fit(ds)
        fact_ds = fact.transform(ds)
        factor_analysis = pd.DataFrame(fact.components_, columns=colunas)
        return fact_ds, factor_analysis

    def ICA(self):
        ds = self.training
        colunas = list(ds)
        ica = FastICA().fit(ds)
        indep_comp = pd.DataFrame(ica.components_, columns=colunas)
        indep_ds = ica.transform(ds)
        return indep_comp, indep_ds

    def PCA_extraction(self):
        ds = self.training
        pca = PCA()
        pca.fit(ds)
        components = pd.Series(pca.explained_variance_, index=range(1, ds.shape[1] + 1))
        components = components * 100
        return components

    def _drop_constant_features(self):
        num_df = self.training._get_numeric_data().drop(['Response'], axis=1)
        const = num_df.columns[num_df.std() < 0.01]
        self.training.drop(labels=const, axis=1, inplace=True)
        self.unseen.drop(labels=const, axis=1, inplace=True)

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

    def linear_regression_selection(self, vd):
        df = self.training
        reg_results = pd.DataFrame(columns=['variable', 'Coef', 'std_err', 'adj_R2', 'pvalue'])

        X = df[vd]
        for var in scaled_df.drop(columns=vd).columns:
            Y = df[var]
            model = sm.OLS(X, Y).fit()
            predictions = model.predict(X)
            reg_results = reg_results.append({'variable': var, 'Coef': model.params[0], 'std_err': model.bse[0],
                                              'pvalue': model.pvalues[0], 'adj_R2': model.rsquared_adj},
                                             ignore_index=True)
        return reg_results

    def fisher_score(self, vd):
        ds = self.training
        f_score = []
        num = []
        den = []
        for class_ in np.unique(ds[vd]):
            nj = ds[ds[vd] == i].shape[0]
            uij = ds[ds[vd] == i].mean()
            ui = ds.mean()
            pij = ds[ds[vd] == i].var()
            num.append(nj * (uij - ui) ** 2)
            den.append(nj * pij)
        return sum(num) / sum(den)

    def entropy(self):
        ds = self.training
        if len(ds.unique()) > 1:
            p_c1 = ds.mean()
            p_c0 = 1 - p_c1
            return np.sum([-p_c0 * np.log2(p_c0), -p_c1 * np.log2(p_c1)])
        else:
            return 0

    def all_inf_gain(self, vd):
        ds = self.training
        vars_inf_gain = []
        for var in ds:
            v = np.sort(ds[var].unique())
            dict_gains = {}
            for i in range(0, len(v) - 1):
                fbin_i = ds[var] <= v[i]
                _0 = ds[vd][~fbin_i]
                _1 = ds[vd][fbin_i]
                dict_gains[var + "_" + str(v[i])] = self.entropy(ds[vd]) - (
                        (~fbin_i).mean() * self.entropy(_0) + fbin_i.mean() * self.entropy(_1))
            vars_inf_gain.append(dict_gains)
        return vars_inf_gain

    def ind_inf_gain(self, var, vd):
        ds = self.training
        '''para ser usado no FCBF'''
        v = np.sort(ds[var].unique())
        dict_gains = {}
        for i in range(0, len(v) - 1):
            fbin_i = ds[var] <= v[i]
            _0 = ds[vd][~fbin_i]
            _1 = ds[vd][fbin_i]
            dict_gains[var + "_" + str(v[i])] = self.entropy(ds[vd]) - (
                        (~fbin_i).mean() * self.entropy(_0) + fbin_i.mean() * self.entropy(_1))
        return dict_gains

    def recursive_feature_elimination(self, vd):
        ds = self.training
        'atencao que ha um parametro que ]e o numero de features que queremos ter... o default ]e metade!'
        X = ds.drop(columns=vd)
        Y = ds[vd]
        model = LogisticRegression()
        rfe = RFE(model)
        fit = rfe.fit(X, Y)
        return fit.support_

    def Anova_F_selection(ds, vd):
        '''again.. aqui ha um k, que ]e o numero de features que queremos ter. Default ele da metade.'''
        X = ds.drop(columns=vd)
        Y = ds[vd]
        sel = SelectKBest(score_func=f_classif)
        sel.fit(X, Y)
        return sel.transform(X), sel.pvalues_

    def extra_Trees_Classifier(ds, vd):
        ''' choosing number of features based on their importance'''
        X = ds.drop(columns=vd)
        Y = ds[vd]
        model = ExtraTreesClassifier()
        model.fit(X, Y)
        return model.feature_importances_