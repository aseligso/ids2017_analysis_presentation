from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz

class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test, feature_names):
        self.df = None
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = None
        self.best_model = None

    def train_models(self):
        self.metrics = []
        self.feature_names = self.X_train.columns.tolist()
        
        models = [('Random Forest', RandomForestClassifier(), {'n_estimators': [50, 100, 150, 200]})]

        for name, model, param_grid in models:
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(self.X_train, self.y_train)
            self.best_model = grid_search.best_estimator_

            y_pred = self.best_model.predict(self.X_test)

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='macro', zero_division=0)
        
            self.metrics.append((name, accuracy, precision, recall))

    def save_decision_tree(self, dot_data):
        with open("decision_tree.dot", "w") as f:
            f.write(dot_data)

