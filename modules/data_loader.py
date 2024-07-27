import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.tree import export_graphviz
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
from pyvis.network import Network
import os

class DataLoader:
    def __init__(self, dataset_filename, target_variable, n_samples=None):
        self.dataset_filename = dataset_filename
        self.target_variable = target_variable
        self.n_samples = n_samples
        self.original_dataframe = None
        self.df_with_categorical = None
        self.df_without_categorical = None
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_smote = None
        self.y_train_smote = None
        self.feature_names = None

    def load_dataset(self):
        self.df = pd.read_csv(self.dataset_filename)
        
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_', regex=False)
        
        print("Target variable (y) values:\n", self.df.label.unique())

        label_encoder = LabelEncoder()
        self.df[self.target_variable] = label_encoder.fit_transform(self.df[self.target_variable])
        self.df = self.df.dropna(subset=[self.target_variable])
        print("Target variable (y) values:\n", self.df.label.unique())

        if self.n_samples is not None:
            self.apply_stratified_sampling()

        self.original_dataframe = self.df.copy(deep=True)
        print("Columns in original_dataframe immediately after loading: ", self.original_dataframe.columns.tolist())

    def impute_data(self):

        self.df_with_categorical = self.impute_dataframe(self.original_dataframe, include_categorical=True)
        self.df_without_categorical = self.impute_dataframe(self.original_dataframe, include_categorical=False)
    
    def impute_dataframe(self, dataframe, include_categorical):
        df_for_imputation = dataframe.copy(deep=True)
        numeric_cols = df_for_imputation.select_dtypes(include=[np.number]).columns
        numeric_imputer = SimpleImputer(strategy='mean')
        df_for_imputation[numeric_cols] = numeric_imputer.fit_transform(df_for_imputation[numeric_cols])
        
        print("Numeric columns imputed:", numeric_cols.tolist())

        if include_categorical:
            categorical_cols = df_for_imputation.select_dtypes(exclude=[np.number]).columns
            print("Categorical columns identified for imputation:", categorical_cols.tolist())
            
            if not categorical_cols.empty:
                categorical_imputer = SimpleImputer(strategy='most_frequent')
                df_for_imputation[categorical_cols] = categorical_imputer.fit_transform(df_for_imputation[categorical_cols])
            else:
                print("No categorical columns found for imputation.")
        else:
            df_for_imputation = df_for_imputation.select_dtypes(include=[np.number])

        return df_for_imputation

    def apply_stratified_sampling(self):
        min_samples_per_class = 30
        
        sampled_df_list = []
        
        for class_value, group in self.df.groupby(self.target_variable):
            if len(group) <= min_samples_per_class:
                sampled_df_list.append(group)
            else:
                sample_size = int(np.rint(self.n_samples * len(group) / len(self.df)))
                sample_size = max(min_samples_per_class, sample_size)
                sampled_group = group.sample(n=sample_size, random_state=42)
                sampled_df_list.append(sampled_group)
        
        self.df = pd.concat(sampled_df_list).sample(frac=1, random_state=42).reset_index(drop=True)
        
        print("Class distribution after sampling:")
        print(self.df[self.target_variable].value_counts())

    def preprocess_data(self):
        df_for_preprocessing = self.df.copy(deep=True)
        
        imputer = SimpleImputer(strategy='mean')
        df_numeric = df_for_preprocessing.select_dtypes(include=[np.number]).copy()
        df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)
        df_for_preprocessing[df_numeric.columns] = df_numeric_imputed

        self.X = df_for_preprocessing.drop(columns=[self.target_variable]).select_dtypes(include=[np.number])
        self.y = df_for_preprocessing[self.target_variable]


    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.5, random_state=42)
        self.feature_names = self.X_train.columns.tolist()

    def balance_data(self):
        smote = SMOTE(random_state=42)
        self.X_train_smote, self.y_train_smote = smote.fit_resample(self.X_train, self.y_train)

        if self.n_samples is not None and self.n_samples < len(self.X_train_smote):
            indices = np.random.choice(self.X_train_smote.index, size=self.n_samples, replace=False)
            self.X_train_smote = self.X_train_smote.loc[indices]
            self.y_train_smote = self.y_train_smote.loc[indices]
