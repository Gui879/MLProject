import sys
import pandas as pd
from datetime import datetime, date


class Dataset:
    """ Loads and prepares the data

        The objective of this class is load the dataset and execute basic data
        preparation before effectively moving into the cross validation workflow.

    """

    def __init__(self, full_path):
        self.rm_df = pd.read_excel(full_path, index_col=0)
        self._drop_metadata_features()
        self._drop_doubleback_features()
        self._drop_unusual_classes()
        self._label_encoder()
        self._as_category()
        self._days_since_customer()

    def _drop_metadata_features(self):
        print(self.rm_df.columns)
        metadata_features = ['CostPerContact', 'RevenuePerPositiveAnswer']
        self.rm_df.drop(labels=metadata_features, axis=1, inplace=True)

    def _drop_doubleback_features(self):
        """ Drops perfectly correlated feature

            From metadata we know that there are two purchase channels: by Catalogue
            or by Internet. One is the opposite of another, reason why we will remove
            one of them, for example, the NetPurchase.

        """

        self.rm_df.drop(["NetPurchase"], axis=1, inplace=True)

    def _drop_unusual_classes(self):
        """ Drops absurd categories

            One of data quality issues is related with the integrity of input features.
            From metadata and posterior analysis of the dataset we know the only possible
            categories for each categorical feature. For this reason we will remove
            everything but in those categories.

        """

        errors_dict = {"Gender": "?", "Education": "OldSchool", "Marital_Status": "BigConfusion"}
        for key, value in errors_dict.items():
            self.rm_df = self.rm_df[self.rm_df[key] != value]

    def _label_encoder(self):
        """ Manually encodes categories (labels) in the categorical features

            You could use automatic label encoder from sklearn (sklearn.preprocessing.LabelEncoder), however,
            when it is possible, I prefer to use a manual encoder such that I have a control on the code of
            each label. This makes things easier to interpret when analyzing the outcomes of our ML algorithms.

        """

        cleanup_nums = {"Gender": {'M': 1, 'F': 0},
                        #"Education": {'Basic': 0, '2n Cycle': 1, 'Graduation': 2, 'Master': 3, 'PhD': 4},
                        #"Marital_Status": {'Married': 0, 'Together': 1, 'Divorced': 2, 'Widow': 3, 'Single': 4}
                        }
        self.rm_df.replace(cleanup_nums, inplace=True)

    def _as_category(self):
        """ Encodes Recomendation and Dependents as categories

            Explicitly encodes Recomendation and Dependents as categorical features.

        """

        self.rm_df["Gender"] = self.rm_df["Gender"].astype('category')
        self.rm_df["Dependents"] = self.rm_df["Dependents"].astype('category')
        self.rm_df["Recomendation"] = self.rm_df["Recomendation"].astype('category')
        # self.rm_df["DepVar"] = self.rm_df["DepVar"].astype('category')
        #self.rm_df["Education"] = self.rm_df["Education"].astype('category')
        #self.rm_df["Marital_Status"] = self.rm_df["Marital_Status"].astype('category')

    def _days_since_customer(self):
        """ Encodes Dt_Customer (nº days since customer)

            Similarly to the label encoder, we have to transform the Dt_Customer in order to feed numerical
            quantities into our ML algorithms. Here we encode Dt_Customer into number the of days since, for
            example, the date when the data was extracted from the source - assume it was on 18/02/1993.

        """

        ref_date = date(2019, 2, 18)
        ser = self.rm_df['Dt_Customer'].apply(func=datetime.date)
        self.rm_df["Dt_Customer"] = ser.apply(func=lambda x: (ref_date - x).days)