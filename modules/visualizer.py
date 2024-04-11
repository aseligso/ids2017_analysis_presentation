import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
import os
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
import modules.utils

class Visualizer:
    def __init__(self, original_dataframe, target_variable, n_samples, feature_names):

        self.target_variable = target_variable
        self.feature_names = feature_names
        self.n_samples = n_samples
        self.original_dataframe = original_dataframe
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = feature_names
        self.tsne_plot = None
        self.umap_plot = None
        self.correlation_plot = None

    def visualize_feature_importance(self, model):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances Derived from the Model")
        bars = plt.bar(range(len(self.feature_names)), importances[indices], color='skyblue', align="center")
        plt.xticks(range(len(self.feature_names)), [self.feature_names[i] for i in indices], rotation=45, ha="right")
        plt.xlabel('Feature Names')
        plt.ylabel('Importance Score')
        plt.xlim([-1, len(self.feature_names)])
        plt.tight_layout()

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval, round(yval, 2), va='bottom')

        plt.show()

    def visualize_with_tsne(self, n_samples):

        tsne = TSNE(n_components=2, perplexity=30.0, random_state=42)
        tsne_results = tsne.fit_transform(self.original_dataframe.select_dtypes(include=[np.number]).drop(columns=[self.target_variable]))

        fig, ax = plt.subplots(figsize=(10, 8)) 
        scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=self.original_dataframe[self.target_variable], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Classes')
        plt.title('t-SNE Visualization')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.show()

        self.tsne_plot = fig 
        return fig

    def visualize_with_umap(self, n_samples):

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(self.original_dataframe.select_dtypes(include=[np.number]).drop(columns=[self.target_variable]))

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=self.original_dataframe[self.target_variable], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Classes')
        plt.title('UMAP Visualization')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.show()

        self.umap_plot = fig 
        return fig

    def visualize_correlations(self, n_samples):

        df_numeric = self.original_dataframe.select_dtypes(include=[np.number])

        corr_matrix = df_numeric.corr()
        features_to_visualize = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.7: 
                    features_to_visualize.add(corr_matrix.columns[i])
                    features_to_visualize.add(corr_matrix.columns[j])

        G = nx.Graph()

        for column in features_to_visualize:
            G.add_node(column, type='feature')

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.7: 
                    G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j])

        fig, ax = plt.subplots(figsize=(14, 12))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=1500, node_color='lightblue', font_size=10)
        plt.title('Network Visualization of High Correlation Features')
        plt.show()

        self.correlation_plot = fig 

        return fig

    def visualize_network_graph(self, x_feature, y_feature, z_feature):

        print(self.original_dataframe)

        G = nx.Graph()

        for _, row in self.original_dataframe.iterrows():
            source_ip = row['source_ip']
            dest_ip = row['destination_ip']
            label = row['label']
            connection_strength = row['total_fwd_packets']
            if not G.has_edge(source_ip, dest_ip):
                G.add_edge(source_ip, dest_ip, label=label, connection_strength=connection_strength)

        net = Network(notebook=True, height='500px', width='100%')

        for edge in G.edges():
            src, dst = edge
            label = "Label: " + str(G.edges[edge]['label'])
            color = 'red' if G.edges[edge]['label'] == 1 else 'blue'
            net.add_node(src, title=src, color=color)
            net.add_node(dst, title=dst, color=color)
            net.add_edge(src, dst, title=label, value=G.edges[edge]['connection_strength'], color=color)

        net.show("network_graph.html")