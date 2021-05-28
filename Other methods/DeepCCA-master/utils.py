from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import pandas as pd


def load_data(data_file1, data_file2):
    """loads the data from the csv files, and converts to numpy arrays"""
    print('loading data ...', data_file1)
    X = pd.read_csv(data_file1)
    # f1.drop(0, axis=1, inplace = True)
    # print(f1.shape)
    # print(f1.head(10))
    
    X = X.transpose().reset_index(drop=True)
    X.drop(0, axis=0, inplace = True)
    X = pd.DataFrame(X)
    #X.reset_index()
    # print(X.shape)
    # print(X.head(10))
    y = pd.read_csv(data_file2)
    y = pd.DataFrame(y, columns=['label'])
    # print(y.shape)
    # print(y.head(10))
    print(X.shape)
    print(y.shape)
    
    
    X = X.values
    X = X.astype(float)
    print(X)
    # for x in X:
    #     x = x.flatten()
    #     print (x.dtype)
    y = np.array(y.values).flatten().tolist()
    print('length of labels is: ', len(y))
    
    
    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2) 
        )
    X_train, X_val, y_train, y_val = (
        train_test_split(X_train, y_train, 
        test_size=0.25) 
        # 0.25 x 0.8 = 0.2
        )


    train_set_x, train_set_y    = make_tensor(X_train, y_train)
    valid_set_x, valid_set_y    = make_tensor(X_val, y_val)
    test_set_x, test_set_y      = make_tensor(X_test, y_test)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]


def make_tensor(data_x, data_y):
    """converts the input to numpy arrays"""
    data_x = torch.tensor(data_x)
    data_mean = torch.mean(data_x)
    data_var = torch.var(data_x)
    data_normalized = (data_x - data_mean) / torch.sqrt(data_var)
    data_y = np.asarray(data_y, dtype='int32')
    return data_normalized, data_y


def svm_classify(data, C):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """
    train_data, _, train_label = data[0]
    valid_data, _, valid_label = data[1]
    test_data, _, test_label = data[2]

    print('training SVM...')
    clf = svm.LinearSVC(C=C, dual=False)
    clf.fit(train_data, train_label.ravel())

    p = clf.predict(test_data)
    test_acc = accuracy_score(test_label, p)
    p = clf.predict(valid_data)
    valid_acc = accuracy_score(valid_label, p)

    return [test_acc, valid_acc]