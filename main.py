import argparse
from modules.data_loader import DataLoader
from modules.model_trainer import ModelTrainer
from modules.visualizer import Visualizer
from modules.utils import save_visualization

def parse_arguments():
    parser = argparse.ArgumentParser(description="Model Evaluation Script")
    parser.add_argument("--dataset_filename", type=str, help="Path to the dataset file")
    parser.add_argument("--target_variable", type=str, help="Name of the target variable")
    parser.add_argument("--n_samples", type=int, help="Number of samples to use")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Load and preprocess data
    data_loader = DataLoader(args.dataset_filename, args.target_variable, args.n_samples)
    data_loader.load_dataset()
    data_loader.impute_data()
    data_loader.preprocess_data()
    data_loader.split_data()
    data_loader.balance_data()
    
    model_trainer = ModelTrainer(data_loader.X_train_smote, data_loader.X_test, data_loader.y_train_smote, data_loader.y_test, data_loader.feature_names)
    metrics = model_trainer.train_models()

    visualizer = Visualizer(data_loader.df_with_categorical, args.target_variable, args.n_samples, data_loader.feature_names)
    
    tsne_fig = visualizer.visualize_with_tsne(args.n_samples)
    save_visualization(tsne_fig, "tsne_plot.png")

    umap_fig = visualizer.visualize_with_umap(args.n_samples)
    save_visualization(umap_fig, "umap_plot.png") 

    corr_fig = visualizer.visualize_correlations(args.n_samples)
    save_visualization(corr_fig, "correlation_plot.png")