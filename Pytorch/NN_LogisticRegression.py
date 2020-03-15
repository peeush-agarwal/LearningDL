import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from NN_LinearRegression import get_device, visualize_data, TabularDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# create dummies for species column
def create_dummies(data, col_name):
    dummies = pd.get_dummies(data[col_name], prefix=col_name)
    data = pd.concat([data, dummies], axis=1)
    return data

def visualize_before_after_scaling(unscaled_features, X):
    # Before & After Mean normalization
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))

    ax1.set_title('Before Scaling')
    sns.kdeplot(unscaled_features['sepal_length'], ax=ax1)
    sns.kdeplot(unscaled_features['sepal_width'], ax=ax1)
    sns.kdeplot(unscaled_features['petal_length'], ax=ax1)
    sns.kdeplot(unscaled_features['petal_width'], ax=ax1)

    ax2.set_title('After Scaling')
    sns.kdeplot(X['sepal_length'], ax=ax2)
    sns.kdeplot(X['sepal_width'], ax=ax2)
    sns.kdeplot(X['petal_length'], ax=ax2)
    sns.kdeplot(X['petal_width'], ax=ax2)

    plt.show()

def prepare_data_for_logistic_regression(data):
    # Encode category in species column to numerical values
    labelEncoder = LabelEncoder()
    data['species'] = labelEncoder.fit_transform(data['species'])

    data = create_dummies(data, 'species')

    data = data.drop('species', axis=1)

    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    label_col = ['species_0','species_1','species_2']

    X = data[feature_cols]
    y = data[label_col]

    unscaled_features = X

    sc = StandardScaler()
    # calculate mean and std dev (fit) and apply the transformation (transform)
    X_array = sc.fit_transform(X.values)
    X = pd.DataFrame(X_array, index = X.index, columns = X.columns)
    #visualize_before_after_scaling(unscaled_features, X)

    return X, y

def main():
    data = pd.read_csv('Data/iris.csv')
    # visualize_data(data)

    X, y = prepare_data_for_logistic_regression(data)

    df = pd.DataFrame(data=np.concatenate(X, y, axis=1),index=X.index, columns=[X.columns,y.columns])
    print(df.head())
    print(df.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 43)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

if __name__ == "__main__":
    main()