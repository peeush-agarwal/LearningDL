""" Basic Neural network from scratch """

import sys
import pandas as pd
import torch
#import os

#print(os.getcwd())
# Display all of the files found in your current working directory
#print(os.listdir(os.getcwd()))

def main(argv):
    """ Basic neural network from scratch """
    data = pd.read_csv('Data/iris.csv')
    print(data.head())

    features = data[['sepal_length', 'sepal_width', 'petal_length']]
    target = data['petal_width']

    print(features.head())
    print(target.head())
    print(type(features))
    print(type(features.values))

    features_tensor = torch.from_numpy(features.values)
    print(features_tensor)

    target_tensor = torch.from_numpy(target.values)
    print(target_tensor)



if __name__ == "__main__":
    main(sys.argv[1:])
