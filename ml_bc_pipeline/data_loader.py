import sys
import pandas as pd
from datetime import datetime, date
from sklearn.preprocessing import OneHotEncoder

class Dataset:
    """ Loads and prepares the data

        The objective of this class is load the dataset and execute basic data
        preparation before effectively moving into the cross validation workflow.

    """

    def __init__(self, full_path):
        self.rm_df = pd.read_excel(full_path)
        self._drop_duplicates(full_path)
        self._drop_metadata_features()
        #self._drop_doubleback_features()
        self._drop_unusual_classes()
        #self._label_encoder()
        #self._as_category()
        self._days_since_customer()
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

    def _drop_duplicates(self,full_path):
        """Nós temos dois tipos de dados repetidos. Dados repetidos com o Response diferente e dados repetidos com o Response
        igual. O que significa que temos de ter cuidado, porque têm de ser tratados de maneira diferente
        Primeiramente vamos eliminar os dados que estão repetidos com Response diferente. No máximo temos 3 casos repetidos,
        pelo que tendo 2 para 1, não representa uma diferença significativamente grande para ficar com 1 individuo do grupo que
        tem mais. Depois, com os dados que ficamos, vamos ver os que têm duplicados, e neste caso vamos ficar sempre com
        um deles. Por default, o primeiro que aparece."""

        ds = self.rm_df
        ds = ds.drop(columns=["ID", "Response"])
        da = pd.read_excel(full_path)

        # Colunas do dataset numa lista e retirar a primeira, aka ID
        colunas = list(da)
        colunas.pop(0)

        # Obter o count e a lista com os ID's dos repetidos com o target.
        target_count = da.groupby(colunas)['ID'].count()
        target_list = da.groupby(colunas)['ID'].apply(list)
        target = pd.concat([target_count, target_list], axis=1)
        target.columns = ['count', 'lista']

        # Obter o count e a lista com os ID's dos repetidos sem o target.
        no_target_count = da.groupby(list(ds))['ID'].count()
        no_target_list = da.groupby(list(ds))['ID'].apply(list)
        no_target = pd.concat([no_target_count, no_target_list], axis=1)
        no_target.columns = ['count', 'lista']

        ## Comparar os resultados do "com target" e do "sem target" e fazer a intersecção dos mesmos. Porque se estão iguais
        ## nos dois lados, significa que nunca há casos em que os Response são diferentes.
        no_target_set = set(map(tuple, no_target.lista))
        target_set = set(map(tuple, target.lista))
        id_intercept = no_target_set.intersection(target_set)
        id_intercept = list(id_intercept)

        # Juntar os ID's numa lista para depois ficar apenas com estes casos
        ids = []
        for i in id_intercept:
            for j in i:
                ids.append(j)

        # Fiz isto para evitar perder os IDS, usando um merge atraves do index
        frame = pd.read_excel("ml_project1_data.xlsx")
        frame = frame.loc[frame['ID'].isin(ids)]
        sem_ID = frame.drop(columns=["ID"])
        sem_ID = sem_ID.drop_duplicates(keep="first")
        frame_ID = frame["ID"]
        merged = sem_ID.merge(frame_ID.to_frame(), left_index=True, right_index=True, how='inner')
        merged.index = merged['ID']
        ds = merged.drop(columns='ID').copy()
        del merged

    def _drop_metadata_features(self):
        #To be used for profit calculations
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

    def _as_category(self):
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
        self.rm_df["Response"] = self.rm_df["Response"].astype('category')

    def _days_since_customer(self):
        """ Encodes Dt_Customer (nº days since customer)

            Similarly to the label encoder, we have to transform the Dt_Customer in order to feed numerical
            quantities into our ML algorithms. Here we encode Dt_Customer into number the of days since, for
            example, the date when the data was extracted from the source - assume it was on 18/02/1993.

        """
        self.rm_df.Dt_Customer = pd.to_datetime(self.rm_df['Dt_Customer'].str.replace('-', ''), format='%Y%m%d', errors='ignore')
        ref_date = datetime.date(datetime.now())
        ser = self.rm_df['Dt_Customer'].apply(func=datetime.date)
        self.rm_df["Dt_Customer"] = ser.apply(func=lambda x: (ref_date - x).days)

