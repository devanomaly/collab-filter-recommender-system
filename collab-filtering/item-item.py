# Here we implement a item-based collaborative filtering, trained with a subset of data (see "preprocessing_final.ipynb") 

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from sortedcontainers import SortedList
import os

# We now load the data (takes about 10s)
with open('user2movie.json', 'rb') as f:
  user2movie = pickle.load(f)
with open('movie2user.json', 'rb') as f:
  movie2user = pickle.load(f)
with open('usermovie2rating.json', 'rb') as f:
  usermovie2rating = pickle.load(f)
with open('usermovie2rating_test.json', 'rb') as f:
  usermovie2rating_test = pickle.load(f)

N = np.max(list(user2movie.keys())) + 1 # N = number of users
# test set may contain movies that train set doesn't have data on
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u,m),r in usermovie2rating_test.items()])
M = max(m1,m2) + 1
print("N =",N)
print("M =",M)

# the calculation of user similarities is now O(N*M^2)
# in the "real-world",  we'd parallelize this (say, with PySpark)
# since the weights are symmetric (w_ij = w_ji) we really only need to do half the calculations... but the price to be paid would be memory =/ 

K = 20 # number of neighbors we'd like to consider
limit = 5 # minimum number of common movies users must have in common in order to the weights be considered
neighbors = [] # we'll store neighbors in this list
averages = [] # each item's average rating for later use
deviations = [] # idem above, now for item's deviation

# This loop ("training") takes a lot of time...
for j in range(M):
  # find the K closest items to item j
  users_j = movie2user[j]
  users_j_set = set(users_j)
  
  # average and deviation...
  ratings_j = {user:usermovie2rating[(user, j)] for user in users_j}
  avg_j = np.mean(list(ratings_j.values()))
  dev_j = {user: (rating - avg_j) for user,rating in ratings_j.items()}
  dev_j_values = np.array(list(dev_j.values()))
  sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))
  
  averages.append(avg_j)
  deviations.append(dev_j)
  
  sl = SortedList()
  for k in range(M):
    # don't include j itself!
    if k != j:
      users_k = movie2user[k]
      users_k_set = set(users_k)
      common_users = (users_j_set & users_k_set) # find their intersection
      if len(common_users) > limit:
        # obtain the average and deviation, now for k
        ratings_k = {user:usermovie2rating[(user, k)] for user in users_k}
        avg_k = np.mean(list(ratings_k.values()))
        dev_k = {user: (rating - avg_k) for user,rating in ratings_k.items()}
        dev_k_values = np.array(list(dev_k.values()))
        sigma_k = np.sqrt(dev_k_values.dot(dev_k_values))
        
        # now, compute the correlation
        numerator = sum(dev_j[m]*dev_k[m] for m in common_users)
        w_jk = numerator/(sigma_j*sigma_k)
        
        # insert into sorted list and truncate
        # negate weight, since list is sorted ascending
        # max (1) is "closest"
        sl.add((-w_jk, k))
        if len(sl) > K:
          del sl[-1]
  # then, we store the neighbors
  neighbors.append(sl)
  if j % 1 == 0:
    print(j)


# now, we calculate train and test MSE
def predict(m,u):
  numerator=0
  denominator=0
  for neg_w, j in neighbors[m]:
    # weight is stored as its negative, remember that!
    try:
      numerator += -neg_w*deviations[j][u]
      denominator += abs(neg_w)
    except KeyError:
      # neighbor didn't rate same movie
      # to avoid 2 lookups, we just throw the exception and pass
      pass
  if denominator == 0:
    prediction = averages[m]
  else:
    prediction = (numerator/denominator) + averages[m]
  prediction = min(5, prediction)
  prediction = max(0.5, prediction)
  return prediction

# let us now train and test!
train_preds = []
train_targets = []
for (u,m), target in usermovie2rating.items():
  prediction = predict(m,u)
  train_preds.append(prediction)
  train_targets.append(target)

test_preds = []
test_targets = []

for (u,m), target in usermovie2rating.items():
  prediction = predict(m,u)
  test_preds.append(prediction)
  test_targets.append(target)
  
# finally, we calculate the RMSE accuracy
def RMSE(p,t):
  p = np.array(p)
  t = np.array(t)
  return np.sqrt(np.mean((p-t)**2))

print("Train RMSE:", RMSE(train_preds, train_targets))
print("Test RMSE:", RMSE(test_preds, test_targets))