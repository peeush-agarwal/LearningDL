import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable 
from NN_LinearRegression import get_device, visualize_data, TabularDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
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

    # data = create_dummies(data, 'species')

    # data = data.drop('species', axis=1)

    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    label_col = ['species']

    return data[data.columns[0:4]].values, data.species.values

    X = data[feature_cols]
    y = data[label_col]

    # unscaled_features = X

    # sc = StandardScaler()
    # # calculate mean and std dev (fit) and apply the transformation (transform)
    # X_array = sc.fit_transform(X.values)
    # X = pd.DataFrame(X_array, index = X.index, columns = X.columns)
    # #visualize_before_after_scaling(unscaled_features, X)

    return X.values, y.values

class FeedForwardNN(nn.Module):
    def __init__(self, n_features, n_labels = 1):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(n_features, 10)
        self.fc2 = nn.Linear(10, n_labels)
        self.softmax = nn.Softmax()
    
    def forward(self, X):
        z1 = F.relu(self.fc1(X))
        z2 = self.fc2(z1)
        output = self.softmax(z2)

        return output

def train(model, X_train, y_train, criterion, device):
  optimizer = optim.SGD(model.parameters(), lr=0.01)
  no_of_epochs = 10
  for epoch in range(no_of_epochs):
      X_train = X_train.to(device)
      y_train  = y_train.to(device)

      # set the parameter gradients to zero
      optimizer.zero_grad()

      # Forward Pass
      preds = model(X_train)
      
      preds = preds.squeeze()
      print(y_train[:5])
      print(preds.data[:5])

      loss = criterion(preds, y_train)
        
      # Backward Pass
      loss.backward()

      # Update the gradients
      optimizer.step()
      print('[Epoch %d] loss:%.3f' % (epoch+1, loss.data[0]))
  print('Training completed')

def test(model, X_test, test_y, criterion, device):
  with torch.no_grad():
    preds = model(X_test)
    _, predict_y = torch.max(preds, 1)
        
    print ('prediction accuracy', accuracy_score(test_y.data, predict_y.data))

    print ('macro precision', precision_score(test_y.data, predict_y.data, average='macro'))
    print ('micro precision', precision_score(test_y.data, predict_y.data, average='micro'))
    print ('macro recall', recall_score(test_y.data, predict_y.data, average='macro'))
    print ('micro recall', recall_score(test_y.data, predict_y.data, average='micro'))

def main():
    data = pd.read_csv('Data/iris.csv')
    # visualize_data(data)

    # # transform species to numerics
    # data.loc[data.species=='setosa', 'species'] = 0
    # data.loc[data.species=='versicolor', 'species'] = 1
    # data.loc[data.species=='virginica', 'species'] = 2

    # X = data[data.columns[0:4]].values
    # y = data.species.values

    X, y = prepare_data_for_logistic_regression(data)
    # print(X[:5])
    # print(y[:5])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 43)
    
    # wrap up with Variable in pytorch
    X_train = Variable(torch.Tensor(X_train).float())
    X_test = Variable(torch.Tensor(X_test).float())
    train_y = Variable(torch.Tensor(y_train).long())
    test_y = Variable(torch.Tensor(y_test).long())

    # train_dataset = TabularDataset(X = X_train, y = y_train)
    # test_dataset = TabularDataset(X = X_test, y = y_test)

    # trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True,num_workers=0)
    # testloader = DataLoader(test_dataset, batch_size=4, shuffle=False,num_workers=0)

    device = get_device()

    model = FeedForwardNN(n_features=4,n_labels=1).to(device)

    criterion = nn.CrossEntropyLoss()

    train(model, X_train, train_y, criterion, device)
    test(model, X_test, test_y, criterion, device)

if __name__ == "__main__":
    main()