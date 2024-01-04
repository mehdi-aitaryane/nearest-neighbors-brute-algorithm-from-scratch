import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


# This function creates a 2D scatter plot of the input data.
# Parameters:
# title: str, Title of the plot
# xlabel: str, Label for the x-axis
# ylabel: str, Label for the y-axis
# X: np.ndarray, Input array of features
# y: np.ndarray, Input array of target values
# Returns: None

def scatter2D(title, xlabel, ylabel, X, y):
    plt.figure(figsize=(14, 8))
    plt.scatter(X[:, 0], X[:, 1], c = y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# This function plots a 2D line graph of X and y
# title: the title of the graph
# xlabel: the label of the x-axis
# ylabel: the label of the y-axis
# X: a 2D array of features
# y: a 1D array of targets

def plot2D(title, xlabel, ylabel, X, y):
    plt.figure(figsize=(20, 8))
    plt.plot(X[:, 0], y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_bf_2d(model, X, y, title = "Title", xlabel = "Feature 1", ylabel = "Target 1", degree=2):
    plt.figure(figsize=(14, 8))

    model.fit(X, y)

    y_true = y
    y_pred = model.predict(X)

    # Scatter plot for data points
    plt.scatter(X[:, 0], y_true, color='blue', label='Data')

    # Line of best fit for predictions
    c_pred = np.polyfit(X[:, 0], y_pred, degree)
    p_pred = np.poly1d(c_pred)
    x_pred = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    plt.plot(x_pred, p_pred(x_pred), color='red', label='Predicted')

    # Line of best fit for actual values
    c_actual = np.polyfit(X[:, 0], y_true, degree)
    p_actual = np.poly1d(c_actual)
    x_actual = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    plt.plot(x_actual, p_actual(x_actual), color='gray', label='Actual')

    # Set labels and titles
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Set legend with titles
    plt.legend()

    plt.show()


def plot_db_2d(model, X, y, title = "Title", xlabel = "Feature 1", ylabel = "Feature 2", step=100):
    plt.figure(figsize=(14, 8))
    
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, step), np.linspace(y_min, y_max, step))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Reshape the predictions to the shape of the meshgrid
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=cm.jet, alpha = 0.3)

    # Plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y)

    # Set plot labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.show()
