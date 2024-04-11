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

class ModelEvaluator:
    def __init__(self, dataset_filename, target_variable, include_pairplot=False, include_decision_tree=False, n_samples=None):
        self.dataset_filename = dataset_filename
        self.target_variable = target_variable
        self.include_pairplot = include_pairplot
        self.include_decision_tree = include_decision_tree
        self.n_samples = n_samples
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.tsne_plot = None
        self.umap_plot = None
        self.correlation_plot = None

    def load_dataset(self):
        self.df = pd.read_csv(self.dataset_filename)
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

        self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_', regex=False)

    def preprocess_data(self):
        label_encoder = LabelEncoder()
        self.df[self.target_variable] = label_encoder.fit_transform(self.df[self.target_variable])

        imputer = SimpleImputer(strategy='mean')
        df_numeric = self.df.select_dtypes(include=[np.number]).copy()
        df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)
        self.df[df_numeric.columns] = df_numeric_imputed

        self.X = self.df.drop(columns=[self.target_variable]).select_dtypes(include=[np.number])
        self.y = self.df[self.target_variable]

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def balance_data(self):
        smote = SMOTE(random_state=42)
        self.X_train_smote, self.y_train_smote = smote.fit_resample(self.X_train, self.y_train)

        if self.n_samples is not None and self.n_samples < len(self.X_train_smote):
            indices = np.random.choice(self.X_train_smote.index, size=self.n_samples, replace=False)
            self.X_train_smote = self.X_train_smote.loc[indices]
            self.y_train_smote = self.y_train_smote.loc[indices]

    def train_models(self):
        self.metrics = []
        self.feature_names = self.X_train.columns.tolist()
        models = [('Random Forest', RandomForestClassifier(), {'n_estimators': [50, 100, 150, 200]})]

        for name, model, param_grid in models:
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(self.X_train_smote, self.y_train_smote)
            best_model = grid_search.best_estimator_

            self.visualize_feature_importance(best_model)

            if self.include_decision_tree and name == 'Decision Tree':
                dot_data = export_graphviz(best_model, out_file=None, filled=True, rounded=True,
                                           special_characters=True, feature_names=self.X.columns,
                                           class_names=np.unique(self.y).astype(str))
                tree_visualizer = graphviz.Source(dot_data)
                tree_visualizer.render("decision_tree")

            y_pred = best_model.predict(self.X_test)

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='macro', zero_division=0)
        
            self.metrics.append((name, accuracy, precision, recall))
            self.display_metrics(name, accuracy, precision, recall)
            self.save_results()

    def display_metrics(self, name, accuracy, precision, recall):
        print(f"Model: {name}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print()

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

    def visualize_with_tsne(self, n_samples=None):
        if n_samples:
            df_sampled = self.df.sample(n=min(n_samples, len(self.df)), random_state=42)
        else:
            df_sampled = self.df

        tsne = TSNE(n_components=2, perplexity=30.0, random_state=42)
        tsne_results = tsne.fit_transform(df_sampled.select_dtypes(include=[np.number]).drop(columns=[self.target_variable]))

        fig, ax = plt.subplots(figsize=(10, 8))  # Create a new figure and axes
        scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=df_sampled[self.target_variable], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Classes')
        plt.title('t-SNE Visualization')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.show()

        self.tsne_plot = fig  # Store the figure object

    def visualize_with_umap(self, n_samples=None):
        if n_samples:
            df_sampled = self.df.sample(n=min(n_samples, len(self.df)), random_state=42)
        else:
            df_sampled = self.df

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(df_sampled.select_dtypes(include=[np.number]).drop(columns=[self.target_variable]))

        fig, ax = plt.subplots(figsize=(10, 8))  # Create a new figure and axes
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=df_sampled[self.target_variable], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Classes')
        plt.title('UMAP Visualization')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.show()

        self.umap_plot = fig  # Store the figure object

    def visualize_correlations(self, n_samples=None):
        if n_samples:
            df_sampled = self.df.sample(n=min(n_samples, len(self.df)), random_state=42)
        else:
            df_sampled = self.df

        df_numeric = df_sampled.select_dtypes(include=[np.number])

        corr_matrix = df_numeric.corr()
        features_to_visualize = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.7:  # Define a threshold for high correlation
                    features_to_visualize.add(corr_matrix.columns[i])
                    features_to_visualize.add(corr_matrix.columns[j])

        # Create a graph
        G = nx.Graph()

        # Add nodes for features
        for column in features_to_visualize:
            G.add_node(column, type='feature')

        # Add edges between features
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.7:  # Define a threshold for high correlation
                    G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j])

        # Visualize the graph
        fig, ax = plt.subplots(figsize=(14, 12))  # Create a new figure and axes
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=1500, node_color='lightblue', font_size=10)
        plt.title('Network Visualization of High Correlation Features')
        plt.show()

        self.correlation_plot = fig  # Store the figure object

    def visualize_network_graph(self, n_samples, x_feature, y_feature, z_feature):
        if self.n_samples:
            df_sampled = self.df.sample(n=min(self.n_samples, len(self.df)), random_state=42)
        else:
            df_sampled = self.df

        G = nx.Graph()

        # Add edges between unique source and destination_ip pairs
        for _, row in df_sampled.iterrows():
            source_ip = row['source_ip']
            dest_ip = row['destination_ip']
            label = row['label']
            connection_strength = row['total_fwd_packets']  # Using Total Fwd Packets as connection strength
            if not G.has_edge(source_ip, dest_ip):
                G.add_edge(source_ip, dest_ip, label=label, connection_strength=connection_strength)

        # Create a pyvis network object
        net = Network(notebook=True, height='500px', width='100%')

        # Add nodes and edges to the network
        for edge in G.edges():
            src, dst = edge
            label = "Label: " + str(G.edges[edge]['label'])
            color = 'red' if G.edges[edge]['label'] == 1 else 'blue'  # Red for label 1, blue for label 0
            net.add_node(src, title=src, color=color)
            net.add_node(dst, title=dst, color=color)
            net.add_edge(src, dst, title=label, value=G.edges[edge]['connection_strength'], color=color)

        # Save the interactive visualization to an HTML file
        net.show("network_graph.html")

    def save_metrics(self, name, accuracy, precision, recall):
        with open(f"{name}_metrics.txt", "w") as f:
            f.write(f"Model: {name}\n")
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")

    def save_decision_tree(self, dot_data):
        with open("decision_tree.dot", "w") as f:
            f.write(dot_data)

    def save_visualization(self, plt, filename):
        plt.savefig(filename)
        # plt.close()

    def save_results(self):
        if not os.path.exists("results"):
            os.makedirs("results")

        # Save metrics
        for name, accuracy, precision, recall in self.metrics:
            self.save_metrics(name, accuracy, precision, recall)

        # Save visualizations if they are not None
        if self.tsne_plot is not None:
            self.save_visualization(self.tsne_plot, "results/tsne_plot.png")
        if self.umap_plot is not None:
            self.save_visualization(self.umap_plot, "results/umap_plot.png")
        if self.correlation_plot is not None:
            self.save_visualization(self.correlation_plot, "results/correlation_plot.png")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Model Evaluation Script")
    parser.add_argument("--dataset_filename", type=str, help="Path to the dataset file")
    parser.add_argument("--target_variable", type=str, help="Name of the target variable")
    parser.add_argument("--include_decision_tree", action="store_true", help="Include decision tree visualization")
    parser.add_argument("--n_samples", type=int, help="Number of samples to use")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    evaluator = ModelEvaluator(dataset_filename=args.dataset_filename, 
                               target_variable=args.target_variable, 
                               include_decision_tree=args.include_decision_tree, 
                               n_samples=args.n_samples)
    
    evaluator.load_dataset()
    evaluator.preprocess_data()
    evaluator.split_data()
    evaluator.balance_data()
    evaluator.train_models()
    evaluator.visualize_with_tsne(n_samples=args.n_samples)
    evaluator.visualize_with_umap(n_samples=args.n_samples)
    evaluator.visualize_correlations(n_samples=args.n_samples)
    evaluator.visualize_network_graph(n_samples=args.n_samples, x_feature='flow_duration', y_feature='total_fwd_packets', z_feature='total_backward_packets')
    evaluator.save_results()