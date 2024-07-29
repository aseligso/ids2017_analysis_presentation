# IDS2017 Dataset Analysis and Model Evaluation Tool

## Overview
This project provides a comprehensive tool for analyzing the IDS2017 dataset and evaluating machine learning models for network intrusion detection. It combines the power of the IDS2017 dataset with advanced data analysis and visualization techniques to enhance network security analysis.

## IDS2017 Dataset and the Problem of Cyber Security Analysis
The IDS2017 dataset, provided by the Canadian Institute for Cybersecurity (CIC), is an industry standard benchmark and resource for developing and evaluating network intrusion detection systems (IDS). Key features include:
- Diverse attack types (DoS, DDoS, brute force, web attacks, etc.)
- Realistic network traffic data
- Extensive feature set (flow-based, packet-based, and time-based attributes)

## Capabilities
This tool leverages the IDS2017 dataset to:
1. Perform anomaly detection in network traffic
2. Classify different types of cyber attacks
3. Conduct feature selection for optimized IDS performance
4. Evaluate and compare various machine learning models for network security

## High Dimensionality in Cyber Network Datasets
Cyber network datasets, including IDS2017, are characterized by their high dimensionality. This presents unique challenges and opportunities:
- Feature richness: Numerous attributes capture various aspects of network behavior.
- Curse of dimensionality: High-dimensional spaces can lead to sparsity and computational complexity.
- Feature selection importance: Identifying the most relevant features becomes crucial for model performance.
- Visualization challenges: Techniques like t-SNE and UMAP become essential for understanding data structure, especially the distribution of the data itself. 

This tool is specifically designed to handle and leverage this high dimensionality for improved intrusion detection.

### Addressing Dynamic Inputs and Specialized Log Analysis
A key challenge in cyber security analysis is dealing with dynamic and diverse input data. This project suggests a way forward in dealing with this diverse and varied input data. 

1. Log Type Specialization:
   - Developing separate analysis pipelines for different log types (e.g., TCP logs, DNS logs, application logs)
   - Creating specialized models optimized for each log type's unique characteristics

2. Dynamic Input Handling:
   - Implementing adaptive preprocessing techniques to handle varying log formats and structures
   - Developing real-time data ingestion and analysis capabilities to process streaming log data

3. Temporal Analysis:
   - Incorporating time-series analysis to detect evolving patterns and anomalies across different log types
   - Developing models that can correlate events across different time scales and log sources

4. Scalability and Performance:
   - Optimizing data processing and model inference for high-volume, high-velocity log data
   - Exploring distributed computing solutions for handling large-scale log analysis

5. Contextual Integration:
   - Developing methods to integrate context from different log types to provide a holistic view of system security
   - Creating visualization tools that can represent relationships and patterns across diverse log data

By addressing these challenges, the aim is to create a more robust, flexible, and comprehensive cyber security analysis framework that can adapt to the dynamic nature of modern network environments and diverse log sources.

## Project Structure

This project is organized to analyze the IDS2017 dataset for network security, with a focus on modularity and command-line interface for flexibility.

project_root/
├── modules/
│   ├── data_loader.py
│   ├── model_trainer.py
│   ├── utils.py
│   └── visualizer.py
├── results/
│   ├── correlation_plot.png
│   ├── tsne_plot.png
│   └── umap_plot.png
├── main.py
├── network_graph.html
├── README.md
└── requirements.txt


## Modules

### data_loader.py
Handles data loading and preprocessing. This module is responsible for:
- Loading the IDS2017 dataset
- Preprocessing steps including imputation and feature engineering
- Splitting data into training and testing sets

### model_trainer.py
Manages model training and evaluation. Key functionalities include:
- Training various machine learning models
- Performing hyperparameter tuning
- Evaluating model performance

### utils.py
Contains utility functions for:
- Saving metrics
- Saving visualizations
- Other helper functions used across the project

### visualizer.py
Handles various data visualization techniques:
- t-SNE visualization
- UMAP projection
- Correlation plots
- Network graph visualization

## Main Execution

The `main.py` file serves as the entry point for the application. It uses command-line arguments to control the flow of the program, allowing for flexible configuration of analysis parameters.

## Results

The `results/` directory stores output visualizations, including:
- Correlation plots
- t-SNE plots
- UMAP plots

Additionally, `network_graph.html` provides an interactive network visualization.

## Configuration

The project uses command-line arguments for configuration, allowing easy adjustment of parameters without modifying the code.

## Dependencies

Required dependencies are listed in `requirements.txt`. Install them using:

## Requirements
- Python 3
- Required packages: `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `umap-learn`, `matplotlib`, `seaborn`, `networkx`, `pyvis`
- 
## Installation

1. Clone this repository:
git clone https://github.com/yourusername/ids2017-analysis-tool.git

2. Install dependencies:
pip install -r requirements.txt

## Usage

Run the script with the IDS2017 dataset:

```bash
python data_analyzer.py --dataset_filename IDS2017.csv --target_variable Label --n_samples 10000

Adjust parameters as needed:

--dataset_filename: Path to the IDS2017 dataset file
--target_variable: Name of the target variable (typically "Label" for IDS2017)
--n_samples: Number of samples to use for visualization
```

## Output
The tool generates:

Evaluation metrics for each trained model and the following visualizations:

1. t-SNE plot of the IDS2017 data (tsne_plot.png)
2. UMAP projection of network traffic (umap_plot.png)
2. Correlation plot of IDS2017 features (correlation_plot.png)
3. Interactive network graph of attack patterns (network_graph.html)

These outputs provide insights into the IDS2017 dataset and the performance of different machine learning models for network intrusion detection.

## Applications in Network Security
This tool enhances network security analysis by:

1. Identifying unusual patterns in IDS2017 traffic that may indicate security threats
2. Categorizing different types of attacks to tailor specific defensive measures
3. Analyzing which features in the IDS2017 dataset contribute most to detecting intrusions
4. Comparing machine learning models to determine the most effective approach for network security

By leveraging the IDS2017 dataset and analysis tool, researchers and practitioners can develop more robust and effective network intrusion detection systems.Future Work: Multimodality in Cyber Security and LLM Integration

## Future Work: Multimodality in Cyber Security and LLM Integration

### Multimodal Cyber Security Analysis
As a future project, we aim to explore multimodal approaches in cyber security:
- Integrating network traffic data with system logs and user behavior data
- Developing models that can process and correlate information from multiple data sources
- Enhancing detection capabilities by leveraging diverse data modalities

### Leveraging LLMs for Log Analysis
We plan to investigate the potential of Large Language Models (LLMs) in cyber security:
- Using LLMs to analyze and interpret complex system and network logs
- Developing natural language interfaces for querying and understanding security events
- Exploring the potential of LLMs in generating human-readable reports from technical log data
