""" Basic Neural network from scratch """

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt

def get_device():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(device)
  device = 'cpu' # 'cuda' gives error "RuntimeError: CUDA error: no kernel image is available for execution on the device"
  return device

# DataLoader - It will load data from disk to CPU or GPU for training
class TabularDataset(Dataset):
  def __init__(self, data, feature_cols=None, label_col=None):
    """
    Characterizes a Dataset for PyTorch

    Parameters
    ----------

    data: pandas data frame
      The data frame object for the input data. It must
      contain all the continuous, categorical and the
      output columns to be used.

    feature_cols: List of strings
      The names of the columns in the data.
      These columns will be passed through the embedding
      layers in the model.

    label_col: string
      The name of the output variable column in the data
      provided.
    """

    self.n = data.shape[0]

    if label_col:
      self.y = data[label_col].astype(np.float32).values.reshape(-1, 1)
    else:
      self.y =  np.zeros((self.n, 1))

    if feature_cols:
      self.X = data[feature_cols].astype(np.float32).values
    else:
      self.X = np.zeros((self.n, 1))

  def __len__(self):
    """
    Denotes the total number of samples.
    """
    return self.n

  def __getitem__(self, idx):
    """
    Generates one sample of data.
    """
    return [self.X[idx], self.y[idx]]

class FeedForwardNN(nn.Module):
    def __init__(self, n_features, n_labels = 1):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 10)
        self.fc2 = nn.Linear(10, n_labels)
    
    def forward(self, X):
        z1 = F.relu(self.fc1(X))
        output = self.fc2(z1)

        return output

def visualize_data(data):
  """
  Exploratory Data Analysis (EDA)

  describe() => Continuous variables description
  value_counts() => Gives distribution of category in species
  pairplot() => Plots the Bivariate relationship from data
  """
  print('Data description:')
  print(data.describe())
  print('Species value counts:')
  print(data['species'].value_counts())
  print('Bivariate pairwise plots:')
  sns.pairplot(data, hue='species', diag_kind='kde')
  plt.show()

def train(model, trainloader, criterion, device):
  optimizer = optim.SGD(model.parameters(), lr=0.01)
  no_of_epochs = 10
  for epoch in range(no_of_epochs):
      running_loss = 0
      for features, y in trainloader:
          features = features.to(device)
          y  = y.to(device)

          # set the parameter gradients to zero
          optimizer.zero_grad()

          # Forward Pass
          preds = model(features)

          loss = criterion(preds, y)
          
          # Backward Pass
          loss.backward()

          # Update the gradients
          optimizer.step()
      
          running_loss += loss.item()
      print('[Epoch %d] loss:%.3f' % (epoch+1, running_loss/len(trainloader)))
  print('Training completed')

def test(model, testloader, criterion, device):
  with torch.no_grad():
    running_loss = 0
    for features, y in testloader:
      features = features.to(device)
      y  = y.to(device)
      
      preds = model(features)
      
      loss = criterion(preds, y)
      running_loss += loss.item()
    print('Test-data Loss:%.3f' % (running_loss/len(testloader)))

def main():
  data = pd.read_csv('Data/iris.csv')

  # visualize_data(data)

  data_train, data_test = train_test_split(data, test_size=0.3, shuffle=True, random_state = 43)
  print(data_train.shape)
  print(data_test.shape)

  feature_cols = ['sepal_length', 'sepal_width', 'petal_length']
  label_col = ['petal_width']

  train_dataset = TabularDataset(data = data_train, feature_cols=feature_cols, label_col=label_col)
  test_dataset = TabularDataset(data = data_test, feature_cols=feature_cols, label_col=label_col)

  trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True,num_workers=0)
  testloader = DataLoader(test_dataset, batch_size=4, shuffle=False,num_workers=0)

  device = get_device()

  model = FeedForwardNN(n_features=3,n_labels=1).to(device)

  criterion = nn.MSELoss()

  train(model, trainloader, criterion, device)
  test(model, testloader, criterion, device)

if __name__ == "__main__":
  main()