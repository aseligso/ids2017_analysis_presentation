# Model Evaluator

This Python script evaluates machine learning models using a provided dataset. It includes functionalities such as data preprocessing, model training, evaluation metrics computation, and visualization.

## Requirements

- Python 3
- Required Python packages: `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `umap-learn`, `matplotlib`, `seaborn`, `networkx`, `pyvis`

## Usage

1. **Clone the Repository**: Clone this repository to your local machine.

2. **Install Dependencies**: Install the required Python packages. You can do this using pip:

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Dataset**: Place your dataset file in the same directory as the script or provide the path to the dataset file using the `--dataset_filename` argument.

4. **Run the Script**: Run the script with the desired arguments:

   ```bash
   python data_analyzer.py --dataset_filename dataset.csv --target_variable target_column --n_samples 1000
   ```

   Replace `dataset.csv` with your dataset filename, `target_column` with the name of the target variable, and adjust other arguments as needed.

## Arguments

- `--dataset_filename`: Path to the dataset file.
- `--target_variable`: Name of the target variable in the dataset.
- `--n_samples`: Number of samples to use for visualization (optional).

## Output

The script generates the following output:

- Metrics files for each model trained.
- Visualizations:
  - t-SNE plot (`tsne_plot.png`)
  - UMAP plot (`umap_plot.png`)
  - Correlation plot (`correlation_plot.png`)
  - Network graph visualization (`network_graph.html`)

## Author

[Andrew Seligson]
