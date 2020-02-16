#!/usr/bin/python

import sys, getopt
import numpy as np
#import matplotlib.pyplot as plt

def init(n):
    X = np.linspace(0,1,n)
    Y = np.sin(2*np.pi*X)
    return X,Y

def display_data(X, Y):
   print(f'X: {X}')
   print(f'Y: {Y}')

# def plot(X, Y):
#     plt.plot(X, Y)

# def model(m):
#     if m == 0:
#         return lambda X, W: np.full(X.size, W[0])
#     return lambda X, W: model(m-1)(X,W) + W[m]*np.power(X,m)

# def compute(m, X, W):
#     return model(m)(X,W)

# def compute_loss(Y, Y_pred):
#    return 1/2 * np.sum((Y_pred - Y)**2)

# def display_loss(m, loss):
#    print(f'Loss at m={m} : {loss}')

def main(argv):
   n_obs = 10
   try:
      opts, args = getopt.getopt(argv,"hn:",["nobservations="])
   except getopt.GetoptError:
      print ('polynomial_curve_fitting.py -n <observations>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print ('polynomial_curve_fitting.py -n <observations>\nDefault value of n (observations) = 10')
         sys.exit()
      elif opt in ("-n", "--nobservations"):
         n_obs = int(arg)
   print ('Num of observations are :', n_obs)

   X, Y = init(n_obs)
   display_data(X, Y)
   #plot(X, Y)
   
   # m_values = [0, 1, 2, 3, 5, 10]
   # for m in m_values:
   #    W = np.zeros(m+1)
   #    Y_0 = compute(m, X, W)
   #    loss = compute_loss(Y, Y_0)
   #    display_loss(m, loss)
   
   print ('End of program')

if __name__ == "__main__":
   main(sys.argv[1:])