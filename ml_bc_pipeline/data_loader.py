import sys
import pandas as pd
from datetime import datetime, date
from sklearn.preprocessing import OneHotEncoder

class Dataset:
    """ Loads and prepares the data

        The objective of this class is load the dataset and execute basic data
        preparation before effectively moving into the cross validation workflow.

    """

    def __init__(self, full_path, unseen = False):
        self.rm_df = pd.read_excel(full_path)
        self.rm_df.set_index('ID',inplace=True)
        self._drop_duplicates(full_path)
        self._drop_metadata_features(unseen = unseen)
        #self._drop_doubleback_features()
        self._drop_unusual_classes()
        #self._label_encoder()
        #self._as_category()
        self._days_since_customer(unseen = unseen)
        self._generate_dummies()
        print("Finnished loading data!")

    def _generate_dummies(self):
        features_to_enconde = ['Education', 'Marital_Status']
        columns = []
        idxs = []
        control = 0
        for column in features_to_enconde:
            for index in range(len(self.rm_df[column].unique()) - 1):
                columns.append(column + '_' + self.rm_df[column].unique()[index])
                idxs.append(control)
                control = control + 1
            control = control + 1
        # encode categorical features from training data as a one-hot numeric array.
        enc = OneHotEncoder(handle_unknown='ignore')

        Xtr_enc = enc.fit_transform(self.rm_df[features_to_enconde].values).toarray()
        # update training data
        df_temp = pd.DataFrame(Xtr_enc[:,idxs], index=self.rm_df.index, columns=columns)
        self.rm_df = pd.concat([self.rm_df, df_temp], axis=1)
        for c in columns:
            self.rm_df[c] = self.rm_df[c].astype('category')
        self.rm_df.drop(features_to_enconde,axis=1,inplace = True)

    def _drop_duplicates(self,full_path):
        print(self.rm_df.shape)
        self.rm_df.drop_duplicates(inplace = True)
        self.rm_df.drop_duplicates(subset = list(set(self.rm_df.columns) - set('Response')), keep = False)

    def _drop_metadata_features(self,unseen = True):
        #To be used for profit calculations
        if unseen:
            pass
        else:
            metadata_features = ['Z_CostContact','Z_Revenue']
            self.rm_df.drop(labels=metadata_features, axis=1, inplace=True)

    def _drop_doubleback_features(self):
        """ Drops perfectly correlated feature

            From metadata we know that there are two purchase channels: by Catalogue
            or by Internet. One is the opposite of another, reason why we will remove
            one of them, for example, the NetPurchase.
        """

        #In our problem we have no doubleback features
        pass

    def _drop_unusual_classes(self):
        """ Drops absurd categories

            One of data quality issues is related with the integrity of input features.
            From metadata and posterior analysis of the dataset we know the only possible
            categories for each categorical feature. For this reason we will remove
            everything but in those categories.

        """

        errors_dict = {"Marital_Status": ['YOLO','Absurd','Alone']}
        for key, value in errors_dict.items():
            self.rm_df = self.rm_df[~self.rm_df[key].isin(value)]

    def _label_encoder(self):
        """ Manually encodes categories (labels) in the categorical features

            You could use automatic label encoder from sklearn (sklearn.preprocessing.LabelEncoder), however,
            when it is possible, I prefer to use a manual encoder such that I have a control on the code of
            each label. This makes things easier to interpret when analyzing the outcomes of our ML algorithms.

        """

        pass

    def _as_category(self, unseen):
        """ Encodes Recomendation and Dependents as categories

            Explicitly encodes Recomendation and Dependents as categorical features.

        """

        self.rm_df["Education"] = self.rm_df["Education"].astype('category')
        self.rm_df["Marital_Status"] = self.rm_df["Marital_Status"].astype('category')
        self.rm_df["AcceptedCmp1"] = self.rm_df["AcceptedCmp1"].astype('category')
        self.rm_df["AcceptedCmp2"] = self.rm_df["AcceptedCmp2"].astype('category')
        self.rm_df["AcceptedCmp3"] = self.rm_df["AcceptedCmp3"].astype('category')
        self.rm_df["AcceptedCmp4"] = self.rm_df["AcceptedCmp4"].astype('category')
        self.rm_df["AcceptedCmp5"] = self.rm_df["AcceptedCmp5"].astype('category')
        self.rm_df["Complain"] = self.rm_df["Complain"].astype('category')
        if unseen:
            pass
        else:
            self.rm_df["Response"] = self.rm_df["Response"].astype('category')

    def _days_since_customer(self, unseen):
        """ Encodes Dt_Customer (n days since customer)

            Similarly to the label encoder, we have to transform the Dt_Customer in order to feed numerical
            quantities into our ML algorithms. Here we encode Dt_Customer into number the of days since, for
            example, the date when the data was extracted from the source - assume it was on 18/02/1993.

        """
        if unseen:
            self.rm_df['Dt_Customer'] = self.rm_df.Dt_Customer.apply(str)
        self.rm_df.Dt_Customer = pd.to_datetime(self.rm_df['Dt_Customer'].str.replace('-', ''), format='%Y%m%d', errors='ignore')
        ref_date = datetime.date(datetime.now())
        if unseen:
            self.rm_df['Dt_Customer'] = pd.to_datetime(self.rm_df.Dt_Customer)
        ser = self.rm_df['Dt_Customer'].apply(func=datetime.date)
        self.rm_df["Dt_Customer"] = ser.apply(func=lambda x: (ref_date - x).days)

