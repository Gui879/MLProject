import sys
import numpy as np
from sklearn.impute import SimpleImputer


class Processor:
    """ Performs data preprocessing

        The objective of this class is to preprocess the data based on training subset. The
        preprocessing steps focus on constant features removal, missing values treatment and
        outliers removal and imputation.

    """

    def __init__(self, training, unseen):
        """ Constructor

            It is worth to notice that both training and unseen are nothing more nothing less
            than pointers, i.e., pr.training is DF_train and pr.unseen is DF_unseen yields True.
            If you want them to be copies of respective objects, use .copy() on each parameter.

        """
        self.training = training #.copy() to mantain a copy of the object
        self.unseen = unseen #.copy() to mantain a copy of the object

        self._drop_constant_features()
        #self._drop_categorical_missing_values()

        num_features = self._filter_df_by_std()
        self._impute_num_missings_mean(num_features)

    def _drop_constant_features(self):
        num_df = self.training._get_numeric_data().drop(['Response'], axis=1)
        const = num_df.columns[num_df.std() < 0.01]
        self.training.drop(labels=const, axis=1, inplace=True)
        self.unseen.drop(labels=const, axis=1, inplace=True)

    def _drop_categorical_missing_values(self):
        features = ["Education", "Marital_Status", "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Complain"]
        self.training.dropna(subset=features, inplace=True)
        self.unseen.dropna(subset=features, inplace=True)

    def _filter_df_by_std(self):
        def _filter_ser_by_std(series_, n_stdev=3.0):
            mean_, stdev_ = series_.mean(), series_.std()
            cutoff = stdev_ * n_stdev
            lower_bound, upper_bound = mean_ - cutoff, mean_ + cutoff
            return [True if i < lower_bound or i > upper_bound else False for i in series_]

        training_num = self.training._get_numeric_data().drop(["Response"], axis=1)
        mask = training_num.apply(axis=0, func=_filter_ser_by_std, n_stdev=3.0)
        training_num[mask] = np.NaN
        self.training[training_num.columns] = training_num

        return list(training_num.columns)

    def _impute_num_missings_mean(self, num_features):
        self._imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        X_train_imputed = self._imputer.fit_transform(self.training[num_features].values)
        X_unseen_imputed = self._imputer.transform(self.unseen[num_features].values)

        self.training[num_features] = X_train_imputed
        self.unseen[num_features] = X_unseen_imputed