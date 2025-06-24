# Machine Learning-Based Estimation of Superdroplet Growth Rates Using DNS Data

This repository houses the complete workflow for estimating superdroplet growth rates using machine learning techniques applied to Direct Numerical Simulation (DNS) data.

- **post-processing**: This directory includes scripts that ingest Lagrangian and Eulerian data from DNS simulations and transform it into a structured format (`training.txt` files) suitable for machine learning model training.

- **training**: This directory features scripts that process the `training.txt` files to extract features and labels for the machine learning model. It explores two distinct scaling methods, each implemented in separate scripts. Additionally, it includes code for training and validating the model, along with generating relevant data outputs and visualizations.

- **aposteriori**: This directory contains scripts designed to perform a posteriori analysis on unseen data, evaluating the model's performance and generalization capabilities.

For more detailed information, please refer to each directory's own README file.
