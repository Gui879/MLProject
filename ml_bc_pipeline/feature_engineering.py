import sys
import numpy as np
import pandas as pd
import datetime
import re

from scipy.stats import stats, chi2_contingency
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler,PowerTransformer
from imblearn.over_sampling import SMOTENC,SMOTE,ADASYN
from ga_feature_selection.feature_selection_ga import FeatureSelectionGA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

from prince import MFA


class FeatureEngineer:

    def __init__(self, training, unseen, seed):
        self._rank = {}
        self.report=[]
        self.training = training
        self.unseen = unseen
        self.seed = seed
        self._extract_business_features()

        #self.linear_regression_selection('Response',10)
        #self.lda_extraction()
        #self.linear_regression_selection('Response',10)
        #components = self.multi_factor_analysis(self.training.shape[1],10)
        #self.training = pd.concat([self.training, components], axis=1)
        #self.correlation_based_feature_selection(self.correlation_feature_ordering)
        #self.rank_features_chi_square()
        self.feature_selection_rank(0.3, self.ga_feature_selection(LogisticRegression(solver='lbfgs')),
                                    self.recursive_feature_elimination('Response', 10),
                                    self.anova_F_selection('Response', 10))
        #self.feature_selection_rank(0.5,self.anova_F_selection('Response',20),self.extra_Trees_Classifier(20))
        #print("Feature Engeneering Completed!")
        #self.ga_feature_selection(LogisticRegression(solver='lbfgs'))
        #self.correlation_based_feature_selection(self.correlation_feature_ordering)

        #self.box_cox_transformations()
        #This Works
        self.feature_selection_rank(0.3,self.ga_feature_selection(LogisticRegression(solver = 'lbfgs')), self.recursive_feature_elimination('Response',10), self.anova_F_selection('Response',10))
        self.multi_factor_analysis(20, 100)
        #self.box_cox_transformations()
        #This Works
        self.SMOTE_NC()
        #self.rank_features_chi_square(self.training.select_dtypes(exclude='category').columns ,self.training.select_dtypes(include='category').columns)


        print("Feature Engeneering Completed!")
        #self.ga_feature_selection(LogisticRegression(solver = 'lbfgs'))


    def _extract_business_features(self):
        self.report.append('_extract_business_features')

        for dataset in [self.training, self.unseen]:

            # TER CUIDADO, CONFIRMAR SE O NUM WEB PURCHASES TB É
            a = dataset["NumWebPurchases"]
            b = dataset["NumWebVisitsMonth"]
            dataset["Web_Purchases_Per_Visit"] = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
            dataset["Total_Purchases"] = dataset["NumWebPurchases"] + dataset["NumCatalogPurchases"] + dataset["NumStorePurchases"]

            b =  dataset["Total_Purchases"]
            dataset["RatioWebPurchases"] = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            a = dataset["NumCatalogPurchases"]
            dataset["RatioCatalogPurchases"] = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            a = dataset["NumStorePurchases"]
            dataset["RatioStorePurchases"] = np.divide(a, b, out=np.zeros_like(a), where=b != 0)

            dataset["Age"] = datetime.datetime.now().year - dataset["Year_Birth"]
            dataset["TotalAcceptedCampaigns"] = dataset["AcceptedCmp1"].astype(int)+dataset["AcceptedCmp2"].astype(int)+ dataset["AcceptedCmp3"].astype(int)+dataset["AcceptedCmp4"].astype(int)+dataset["AcceptedCmp5"].astype(int)
            # Total amount spent
            dataset["TotalMoneySpent"] = dataset["MntWines"] + dataset["MntFruits"] + dataset["MntMeatProducts"] + dataset["MntFishProducts"] + dataset["MntSweetProducts"] + dataset["MntGoldProds"]
            # Calculating the ratios of money spent per category
            dataset["RatioWines"] = dataset["MntWines"] / dataset["TotalMoneySpent"]
            dataset["RatioFruits"] = dataset["MntFruits"] / dataset["TotalMoneySpent"]
            dataset["RatioMeatProducts"] = dataset["MntMeatProducts"] / dataset["TotalMoneySpent"]
            dataset["RatioFishProducts"] = dataset["MntFishProducts"] / dataset["TotalMoneySpent"]
            dataset["RatioSweetProducts"] = dataset["MntSweetProducts"] / dataset["TotalMoneySpent"]
            dataset["RatioGoldProdataset"] = dataset["MntGoldProds"] / dataset["TotalMoneySpent"]
            a = dataset["TotalMoneySpent"]
            dataset["MoneyPerPurchase"] = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            # Changing income to 2 years
            dataset["Income2Years"] = dataset["Income"] * 2

            # Calculating Effort Rate
            a = dataset["TotalMoneySpent"]
            b = dataset["Income2Years"]
            dataset["EffortRate"] = np.divide(a, b, out=np.zeros_like(a), where=b != 0)

            # All kidataset
            dataset["TotalKids"] = dataset["Teenhome"] + dataset["Kidhome"]

            # People per Household
            dataset["Count_Household"] = 0
            dataset["Count_Household"] = 2*pd.to_numeric(dataset['Marital_Status_Married']).values + 2*pd.to_numeric(dataset['Marital_Status_Together']) + dataset["TotalKids"]
            dataset["Count_Household"] = pd.to_numeric(dataset['Marital_Status_Divorced']) + pd.to_numeric(dataset['Marital_Status_Single']) + dataset["TotalKids"]


            # Income per person in household
            a = dataset["Income2Years"]
            b = dataset["Count_Household"]
            dataset["Income_Per_Person"] = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            features_to_enconde = ['Education', 'Marital_Status']
            dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
            dataset.fillna(0, inplace=True)

    def lda_extraction(self):
        self.report.append('lda_extraction')
        ds = self.training
        y = self.training['Response']
        clf = LinearDiscriminantAnalysis(solver="svd")
        lda = clf.fit(ds, y)
        lda_ds = lda.transform(ds)
        lda_coef = lda.coef_
        lda_means = lda.means_
        exp_var = lda.explained_variance_ratio_
        return lda_ds, exp_var

    def factor_analysis_extraction(self):
        self.report.append('factor_analysis_extraction')
        ds = self.training
        colunas = list(ds)
        fact = FactorAnalysis().fit(ds)
        fact_ds = fact.transform(ds)
        factor_analysis = pd.DataFrame(fact.components_, columns=colunas)
        return fact_ds, factor_analysis

    def ica(self):
        self.report.append('ica')
        ds = self.training
        colunas = list(ds)
        ica = FastICA().fit(ds)
        indep_comp = pd.DataFrame(ica.components_, columns=colunas)
        indep_ds = ica.transform(ds)
        return indep_comp, indep_ds

    def pca_extraction(self,threshold = 0.8):
        self.report.append('pca_extraction')
        ds_training = self.training.copy().loc[:, self.training.dtypes != 'category'].drop('Response',axis = 1)
        ds_testing = self.unseen.copy().loc[:, self.unseen.dtypes != 'category'].drop('Response', axis=1)
        pca = PCA(random_state = self.seed)
        train_components = pca.fit_transform(ds_training.values,10)
        explained = 0
        final_components = 0
        for component in pca.explained_variance_ratio_:
            explained = explained + component
            final_components = final_components + 1
            if explained >= threshold:
                break
        pca_components = train_components[:,:final_components]
        training_components = pd.DataFrame(pca_components, columns=['C_' + str(col) for col in range(final_components)],index=self.training.index)
        test_components = pca.transform(ds_testing.values)
        print(final_components)
        pca_components = test_components[:,:final_components]
        testing_components = pd.DataFrame(pca_components, columns=['C_' + str(col) for col in range(final_components)],index=self.unseen.index)
        print(testing_components.shape)
        training_components, testing_components = self._normalize(training_components, testing_components)
        training_components['Response'] = self.training['Response']
        testing_components['Response'] = self.unseen['Response']
        self.training = training_components
        self.unseen = testing_components
        print(self.training.columns, self.unseen.columns)

    def _drop_constant_features(self):
        self.report.append('_drop_constant_features')
        num_df = self.training._get_numeric_data().drop(['Response'], axis=1)
        const = num_df.columns[num_df.std() < 0.01]
        self.training.drop(labels=const, axis=1, inplace=True)
        self.unseen.drop(labels=const, axis=1, inplace=True)

    def box_cox_transformations(self):
        self.report.append('box_cox_transformations')
        num_features = self.training.copy().loc[:, self.training.dtypes != 'category'].drop('Response', axis=1).columns
        target = 'Response'
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
                chi_test_value = chi2_contingency(cont_tab)[0]
                # 3) 1) 5) choose the best so far Box-Cox transformation based on Chi-Squared test
                if chi_test_value > best_test_value:
                    best_test_value, best_trans_label, best_power_trans = chi_test_value, trans_key, feature_trans
            self.best_bx_cx_dict[feature] = (best_trans_label, best_power_trans)
            # 3) 2) append transformed feature to the data frame
            self.training[feature] = best_power_trans
            # 3) 3) apply the best Box-Cox transformation, determined on training data, on unseen data
            self.unseen[feature] = np.round(self._bx_cx_trans_dict[best_trans_label](self.unseen[feature]), 4)
        print(self.training.columns)
        self.box_cox_features = num_features_BxCx

    def multi_factor_analysis(self, n_components, n_iterations):
        X = self.training.drop(columns = ['Response'])
        groups = {'categorical': X.loc[:, self.training.dtypes != 'category'].columns,'numerical': X.loc[:, self.training.dtypes == 'category'].columns}
        mfa = MFA(
        groups = groups,
        n_components = n_components,
        n_iter = n_iterations,
        copy = True,
        check_input = True,
        engine = 'auto',
        random_state = self.seed)
        components = mfa.fit_transform(X,self.training['Response']).values
        inertia = mfa.explained_inertia_
        t = 0
        n_components = 0
        for col in range(len(inertia)):
            t += inertia[col]
            n_components = n_components + 1
            if t >= 0.8:
                break
        un_components = mfa.transform(self.unseen.drop('Response', axis = 1)).values
        transformed = pd.DataFrame(components[:,:n_components], columns = ['C_' + str(i) for i in range(n_components)])
        transformed_un = pd.DataFrame(un_components[:,:n_components], columns = ['C_' + str(i) for i in range(n_components)])
        training_components, testing_components = self._normalize(transformed, transformed_un)
        self.training['Response'].index = training_components.index
        training_components['Response'] = self.training['Response']
        self.unseen['Response'].index = testing_components.index
        testing_components['Response'] = self.unseen['Response']
        self.training = training_components
        self.unseen = testing_components

    ########FEATURE SELECTION################################

    def rank_features_chi_square(self):
        self.report.append('rank_features_chi_square')
        chisq_dict = {}
        continuous_flist = self.training.drop('Response',axis = 1).loc[:,self.training.dtypes != 'category'].columns
        categorical_flist = self.training.drop('Response',axis = 1).loc[:, self.training.dtypes == 'category'].columns
        if len(continuous_flist)>0:
            bindisc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy="uniform")
            for feature in continuous_flist:
                feature_bin = bindisc.fit_transform(self.training[feature].values[:, np.newaxis])
                feature_bin = pd.Series(feature_bin[:, 0], index=self.training.index)
                cont_tab = pd.crosstab(feature_bin, self.training["Response"], margins=False)
                chisq_dict[feature] = chi2_contingency(cont_tab.values)[0:2]
        if len(categorical_flist)>0:
            for feature in categorical_flist:
                cont_tab = pd.crosstab(self.training[feature], self.training["Response"], margins=False)
                chisq_dict[feature] = chi2_contingency(cont_tab.values)[0:2]

        df_chisq_rank = pd.DataFrame(chisq_dict, index=["Chi-Squared", "p-value"]).transpose()
        df_chisq_rank.sort_values("Chi-Squared", ascending=False, inplace=True)
        df_chisq_rank["valid"] = df_chisq_rank["p-value"] <= 0.05
        self._rank["chisq"] = df_chisq_rank

    def print_top(self, n):
        print(self._rank.index[0:n])

    def get_top(self, criteria="chisq", n_top=10):
        input_features = list(self._rank[criteria].index[0:n_top])
        input_features.append("Response")
        return self.training[input_features], self.unseen[input_features]

    def linear_regression_selection(self, vd, n):
        self.report.append('linear_regression_selection')
        reg_results = pd.DataFrame(columns=['variable', 'Coef', 'std_err', 'adj_R2', 'pvalue'])
        X = self.training[vd]
        for var in self.training.drop(columns=vd).columns:
            Y = self.training[var]
            model = sm.OLS(X, Y).fit()
            predictions = model.predict(X)
            reg_results = reg_results.append({'variable': var, 'Coef': model.params[0], 'std_err': model.bse[0],
                                              'pvalue': model.pvalues[0], 'adj_R2': model.rsquared_adj},
                                             ignore_index=True)
            reg_results.sort_values(by='adj_R2', inplace=True, ascending=False)

        return np.array(reg_results.head(n)['variable'].values)

    def fisher_score(self, vd, n):
        self.report.append('fisher_score')
        ds = self.training
        f_score = []
        num = []
        den = []
        for class_ in np.unique(ds[vd]):
            nj = ds[ds[vd] == class_].shape[0]
            uij = ds[ds[vd] == class_].mean()
            ui = ds.mean()
            pij = ds[ds[vd] == class_].var()
            num.append(nj * (uij - ui) ** 2)
            den.append(nj * pij)
        results = sum(num) / sum(den)
        results.drop(labels='Response', inplace=True)
        results.sort_values(ascending=False, inplace=True)
        return np.array(results.head(n).index)

    def entropy(self):
        ds = self.training
        if len(ds.unique()) > 1:
            p_c1 = ds.mean()
            p_c0 = 1 - p_c1
            return np.sum([-p_c0 * np.log2(p_c0), -p_c1 * np.log2(p_c1)])
        else:
            return 0

    def all_inf_gain(self, vd, n):
        self.report.append('all_inf_gain')
        ds = self.training
        vars_inf_gain = []
        best_dict = {}
        for var in ds:
            v = np.sort(ds[var].unique())
            dict_gains = {}
            for i in range(0, len(v) - 1):
                fbin_i = ds[var] <= v[i]
                _0 = ds[vd][~fbin_i]
                _1 = ds[vd][fbin_i]
                res = self.entropy(ds[vd]) - ((~fbin_i).mean() * self.entropy(_0) + fbin_i.mean() * self.entropy(_1))
                dict_gains[var + "_" + str(v[i])] = res
            vars_inf_gain.append(dict_gains)

        vars_inf_gain = {k: v for d in vars_inf_gain for k, v in d.items()}

        best_dict = {}
        keys = [key for key in vars_inf_gain.keys()]
        for var in ds.drop(columns=vd):
            r = re.compile(var + '_.*')
            vals = [vars_inf_gain[x] for x in list(filter(r.match, keys))]
            if vals:
                best_dict[list(vars_inf_gain.keys())[list(vars_inf_gain.values()).index(max(vals))]] = max(vals)

        best_dict = dict(sorted(best_dict.items(), key=lambda kv: kv[1], reverse=True))
        ks = [k.split('_')[0] for k in best_dict.keys()]
        return np.array(pd.DataFrame(best_dict, index=[0]).T.head(n).index), ks[:n]

    def ind_inf_gain(self, var, vd):
        self.report.append('ind_inf_gain')
        ds = self.training
        v = np.sort(ds[var].unique())
        dict_gains = {}

        for i in range(0, len(v) - 1):
            fbin_i = ds[var] <= v[i]
            _0 = ds[vd][~fbin_i]
            _1 = ds[vd][fbin_i]
            dict_gains[var + "_" + str(v[i])] = self.entropy(ds[vd]) - (
                        (~fbin_i).mean() * self.entropy(_0) + fbin_i.mean() * self.entropy(_1))

        return dict_gains

    def recursive_feature_elimination(self, vd, n):
        self.report.append('recursive_feature_elimination')
        ds = self.training
        'atencao que ha um parametro que ]e o numero de features que queremos ter... o default ]e metade!'
        X = ds.drop(columns=vd)
        Y = ds[vd]
        model = LogisticRegression()
        rfe = RFE(model, n_features_to_select=n)
        fit = rfe.fit(X, Y)
        return np.array(ds.drop(columns=vd).columns)[np.array(fit.support_)]

    def anova_F_selection(self, vd, n):
        self.report.append('anova_F_selection')
        '''again.. aqui ha um k, que ]e o numero de features que queremos ter. Default ele da metade.'''
        ds = self.training
        X = ds.drop(columns=vd)
        Y = ds[vd]
        sel = SelectKBest(score_func=f_classif)
        sel.fit(X, Y)
        res = dict(zip(ds.drop(columns=vd).columns, sel.scores_))
        res = dict(sorted(res.items(), key=lambda kv: kv[1], reverse=True))
        return np.array(pd.DataFrame(res, index=[0]).T.head(n).index)

    def extra_Trees_Classifier(self, n):
        self.report.append('extra_Trees_Classifier')
        ''' choosing number of features based on their importance'''
        ds = self.training
        X = ds.drop(columns='Response')
        Y = ds['Response']
        model = ExtraTreesClassifier()
        model.fit(X, Y)
        res = dict(zip(ds.drop(columns='Response').columns, model.feature_importances_))
        res = dict(sorted(res.items(), key=lambda kv: kv[1], reverse=True))
        return np.array(pd.DataFrame(res, index=[0]).T.head(n).index)

    def feature_selection_rank(self,treshold, *arg):
        print('before\n')
        print(self.training.head())
        VARS = []
        for array in arg:
            VARS.append(array)
        VARS = [id_ for sublist in VARS for id_ in sublist]
        ratios = [VARS.count(i) / len(arg) for i in VARS]
        ratios = dict(sorted(dict(zip(VARS, ratios)).items(), key=lambda x: x[1], reverse=True))
        kept_vars = list({k for (k, v) in ratios.items() if v > treshold})
        temp = self.training[kept_vars]
        temp_un = self.unseen[kept_vars]
        temp['Response'] = self.training['Response']
        temp_un['Response'] = self.unseen['Response']
        self.training = temp
        self.unseen = temp_un

    def correlation_feature_ordering(self):
        self.report.append('Correlation_Feature_ordering')
        vars_corr = {}
        feature_order = self.training.drop('Response',axis=1).columns
        for var in range(len(feature_order)):
            try:
                correlation = np.abs(self.training[[feature_order[var],'Response']].astype(float).corr().iloc[0,1])
                vars_corr[feature_order[var]] = correlation
            except:
                pass

        return dict(sorted(vars_corr.items(), key=lambda kv: kv[1], reverse=True))

    def correlation_based_feature_selection(self,feature_importance_function):
        self.report.append('Correlation_Based_Feature_selection')
        #Returns variables sorted from most importance to least important
        variables_list = feature_importance_function()
        keys = list(variables_list.keys())
        to_delete = []
        iter_ = iter(range(1,len(variables_list)-1))
        corrs = np.abs(self.training[keys].astype('float64').corr())
        for i in iter_:
            key = keys[i]
            if key in to_delete:
                next(iter_,None)
            for j in range(0,i):
                key2 = keys[j]
                if key2 not in to_delete:
                    correlation = corrs[key].ix[key2]
                    if correlation > 0.8:
                        to_delete.append(key2)
        for key in to_delete:
            del variables_list[key]
        print(len(variables_list.keys()))
        self.training = self.training[list(variables_list.keys())+['Response']]
        self.unseen = self.unseen[list(variables_list.keys()) + ['Response']]

    def ga_feature_selection(self,model):

        feature_selection = FeatureSelectionGA(model,
                                               self.training.loc[:, self.training.columns != "Response"].values,
                                               self.training["Response"].values)
        feature_selection.generate(n_pop=50, ngen=4)
        print(self.training.columns)
        print(self.training.columns[np.where(np.array(feature_selection.best_ind)==1)])
        #Print above works, make return of dic with coolumns
        return self.training.columns[np.where(np.array(feature_selection.best_ind)==1)]



    #SAMPLING

    def SMOTE_NC(self):
        categories = self.training.dtypes
        self.report.append('SMOTE_NC_sampling')
        Y = self.training["Response"]
        X = self.training.drop(columns=["Response"])
        x_cols = X.columns
        cat_cols = X.loc[:, self.training.dtypes == 'category'].columns
        if len(cat_cols) > 0:
            sm = SMOTENC(random_state=self.seed, categorical_features=[cat_cols.get_loc(col) for col in cat_cols])
        else:
            sm = SMOTE(random_state=self.seed)
        X_res, Y_res = sm.fit_resample(X.values, Y.values)
        sampled_ds = pd.DataFrame(X_res, columns=x_cols)
        sampled_ds['Response'] = Y_res
        # sampled_ds.index=ds.index
        self.training = sampled_ds

    def SMOTE_sampling(self, ds):
        self.report.append('SMOTE_sampling')
        Y = ds["Response"]
        X = ds.drop(columns=["Response"])
        sm = SMOTE(random_state=self.seed)
        X_res, Y_res = sm.fit_resample(X, Y)
        sampled_ds = pd.DataFrame(X_res)
        sampled_ds['Response'] = Y_res
        # sampled_ds.index=ds.index
        sampled_ds.columns = ds.columns
        return round(sampled_ds, 2)

    def Adasyn_sampling(self, ds):
        self.report.append('Adasyn_sampling')
        Y = ds["Response"]
        X = ds.drop(columns=["Response"])
        ada = ADASYN(random_state=self.seed)
        X_res, Y_res = ada.fit_resample(X, Y)
        sampled_ds = pd.DataFrame(X_res)
        sampled_ds['Response'] = Y_res
        # sampled_ds.index=ds.index
        sampled_ds.columns = ds.columns
        return round(sampled_ds, 2)
    
    def _normalize(self,training,unseen):
        dummies = list(training.select_dtypes(include=["category", "object"]).columns)
        dummies.append('Response')
        scaler = MinMaxScaler()
        scaler.fit(training.values)
        training = pd.DataFrame(scaler.transform(training.values), columns=training.columns, index=training.index)
        print(unseen.shape)
        unseen = pd.DataFrame(scaler.transform(unseen.values), columns=unseen.columns, index=unseen.index)
        return training, unseen

    def decision_tree_forward(self, n_selected_features):
        """
        This function implements the forward feature selection algorithm based on decision tree

        Input
        -----
        X: {numpy array}, shape (n_samples, n_features)
            input data
        y: {numpy array}, shape (n_samples, )
            input class labels
        n_selected_features: {int}
            number of selected features

        Output
        ------
        F: {numpy array}, shape (n_features,)
            index of selected features
        """

        x_train = self.training.loc[:, self.training.columns!='Response'].values
        y = self.training['Response'].values
        n_samples, n_features = x_train.shape
        # using 10 fold cross validation
        cv = KFold(n_splits=10, shuffle=True)
        # choose decision tree as the classifier
        clf = DecisionTreeClassifier()

        # selected feature set, initialized to be empty
        F = []
        count = 0
        while count < n_selected_features:
            max_acc = 0
            for i in range(n_features):
                if i not in F:
                    F.append(i)
                    X_tmp = x_train[:, F]
                    acc = 0
                    for train, test in cv.split(x_train,y):
                        clf.fit(X_tmp[train], y[train])
                        y_predict = clf.predict(X_tmp[test])
                        acc_tmp = accuracy_score(y[test], y_predict)
                        acc += acc_tmp
                    acc = float(acc) / 10
                    F.pop()
                    # record the feature which results in the largest accuracy
                    if acc > max_acc:
                        max_acc = acc
                        idx = i
            # add the feature which results in the largest accuracy
            F.append(idx)
            count += 1
        return self.training.loc[:, self.training.columns!='Response'].columns[np.array(F)].values

