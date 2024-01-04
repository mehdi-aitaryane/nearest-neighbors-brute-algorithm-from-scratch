import numpy as np

def splitXy(X, y, test_size = 0.2, shuffle = True, random_state = None):
    dataset = np.concatenate((X, y.reshape(len(y), 1)), axis=1)
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(dataset)
    ind = int((1 - test_size) * len(dataset))
    train = dataset[0: ind]
    test = dataset[ind:]
    X_train = train[:, :dataset.shape[1] - 1].astype(X.dtype)
    X_test = test[:, :dataset.shape[1] - 1].astype(X.dtype)
    y_train = train[:, dataset.shape[1] - 1].astype(y.dtype)
    y_test = test[:, dataset.shape[1] - 1].astype(y.dtype)
    return X_train, X_test, y_train, y_test