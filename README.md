# Nearest Neighbors Brute Algorithm From Scratch
## Introduction
This project implements the Nearest Neighbors Brute Algorithm from scratch. The algorithm is a simple yet powerful method used for classification and regression tasks. This markdown document provides an overview of the mathematical concepts behind the algorithm, its pseudocode, and the modular structure of the code.

## Mathematics Behind Nearest Neighbors Brute Algorithm
The Nearest Neighbors Brute Algorithm relies on distance metrics to find the closest neighbors. The commonly used distance functions include Euclidean and Manhattan distances. These distances are essential for determining the similarity between data points.

## Pseudocode of Nearest Neighbors Brute Algorithm
The pseudocode outlines the key steps of the Nearest Neighbors Brute Algorithm. It involves iterating through each test data point, calculating distances to all training points, and selecting the nearest neighbor based on the chosen distance metric.

## Modules
The code is organized into several modules for clarity and maintainability:

### Module 1: datasets.py
Handles dataset generation functions, such as make_classification, make_blobs, and make_regression.

### Module 2: plots.py
Contains functions for visualizing data, including 2D scatter plots and line graphs.

### Module 3: metrics.py
Defines evaluation metrics such as accuracy and R-squared.

### Module 4: splitters.py
Implements dataset splitting functions, like splitXy.

### Module 5: neighbors.py
Implements the core functionality of the Nearest Neighbors Brute Algorithm for both classification and regression tasks.

## Examples
The code includes examples demonstrating how to use the Nearest Neighbors Brute Algorithm for classification and regression problems.

### Example 1: Nearest Neighbors For Classification Problem
Illustrates the application of the algorithm for a classification task.

### Example 2: Nearest Neighbors For Regression Problem
Demonstrates the use of the algorithm for a regression task.

## Usage
To use the project, make sure to install necessary dependencies by running pip install numpy matplotlib before executing the code in the notebook.

## Contributing
The project welcomes contributions from other users. They can open an issue or submit a pull request with their ideas or changes.

## License
The project is licensed under the terms of the MIT license.





