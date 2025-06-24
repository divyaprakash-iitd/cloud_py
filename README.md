# Machine Learning-Based Estimation of Superdroplet Growth Rates Using DNS Data

This repository contains code of the entire workflow of the this project. 

The `post-processing` directory contains codes which reads the Lagrangian and Eulerian data generated from the DNS at each time-step and processes it to generate superdroplets' data that can be used for training in the form of `training.txt` files. 

The `training` directory contains the codes for processing the `training.txt` files in order to generate features and labels that can be fed into the Machine Learning (ML) model. There are two types of scaling which has been explored, each in a separate script. It also contains the script that trains and validates the model and generates the relevant data and plots.  

The `aposteriori` directory contains the code to carry out the a posteriori analysis on the unseen data.

More details can be found in each repository's own README files.
