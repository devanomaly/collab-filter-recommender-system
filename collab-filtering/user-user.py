# Here we implement a user-based collaborative filtering, trained with a subset of data (see "preprocessing_final.ipynb")

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from time import time
from sortedcontainers import SortedList
import os

# We now load the data (takes about 10s)
with open("user2movie.json", "rb") as f:
    user2movie = pickle.load(f)
with open("movie2user.json", "rb") as f:
    movie2user = pickle.load(f)
with open("usermovie2rating.json", "rb") as f:
    usermovie2rating = pickle.load(f)
with open("usermovie2rating_test.json", "rb") as f:
    usermovie2rating_test = pickle.load(f)

N = np.max(list(user2movie.keys())) + 1  # N = number of users
# test set may contain movies that train set doesn't have data on
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1
print("N =", N)
print("M =", M)

# the calculation of user similarities is O(M*N^2)
# in the "real-world",  we'd parallelize this (say, with PySpark)
# since the weights are symmetric (w_ij = w_ji) we really only need to do half the calculations... but the price to be paid would be memory =/

K = 25  # number of neighbors we'd like to consider
limit = 5  # minimum number of common movies users must have in common in order to the weights be relevant
neighbors = []  # we'll store neighbors in this list
averages = []  # each user's average rating for later use
deviations = []  # idem above, now for user's deviation

# This takes even more time than "item-item.py" training loop...
print("starting loops now")
beginning = time()
for j in range(N):
    # find the K closest users to user j
    movies_j = user2movie[j]
    movies_j_set = set(movies_j)

    # average and deviation...
    ratings_j = {movie: usermovie2rating[(j, movie)] for movie in movies_j}
    avg_j = np.mean(list(ratings_j.values()))
    dev_j = {movie: (rating - avg_j) for movie, rating in ratings_j.items()}
    dev_j_values = np.array(list(dev_j.values()))
    sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))
    averages.append(avg_j)
    deviations.append(dev_j)

    sl = SortedList()
    for k in range(N):
        # don't include j itself!
        if k != j:
            movies_k = user2movie[k]
            movies_k_set = set(movies_k)
            common_movies = movies_j_set & movies_k_set  # find their intersection
            if len(common_movies) > limit:
                # obtain the average and deviation, now for k
                ratings_k = {movie: usermovie2rating[(k, movie)] for movie in movies_k}
                avg_k = np.mean(list(ratings_k.values()))
                dev_k = {movie: (rating - avg_k) for movie, rating in ratings_k.items()}
                dev_k_values = np.array(list(dev_k.values()))
                sigma_k = np.sqrt(dev_k_values.dot(dev_k_values))

                # now, compute the correlation
                numerator = sum(dev_j[m] * dev_k[m] for m in common_movies)
                w_jk = numerator / (sigma_j * sigma_k)

                # insert into sorted list and truncate
                # negate weight, since list is sorted ascending
                # max (1) is "closest"
                sl.add((-w_jk, k))
                if len(sl) > K:
                    del sl[-1]
    # then, we store the neighbors
    neighbors.append(sl)
    print("j =", j)
print("done with the loops")
# in my computer, it took around 14.8h
print("it took about " + str(time() - beginning) + "s")

# now, we calculate train and test MSE
def predict(n, m):
    numerator = 0
    denominator = 0
    for neg_w, j in neighbors[n]:
        # weight is stored as its negative, remember that!
        try:
            numerator += -neg_w * deviations[j][m]
            denominator += abs(neg_w)
        except KeyError:
            # neighbor didn't rate same movie
            # to avoid 2 lookups, we just throw the exception and pass
            pass
    if denominator == 0:
        prediction = averages[n]
    else:
        prediction = (numerator / denominator) + averages[n]
    prediction = min(5, prediction)
    prediction = max(0.5, prediction)
    return prediction


# let us now train and test!
train_preds = []
train_targets = []
for (n, m), target in usermovie2rating.items():
    prediction = predict(n, m)
    train_preds.append(prediction)
    train_targets.append(target)

test_preds = []
test_targets = []

for (n, m), target in usermovie2rating_test.items():
    prediction = predict(n, m)
    test_preds.append(prediction)
    test_targets.append(target)

# finally, we calculate the RMSE accuracy
def RMSE(p, t):
    p = np.array(p)
    t = np.array(t)
    return np.sqrt(np.mean((p - t) ** 2))


print("Train RMSE:", RMSE(train_preds, train_targets))
print("Test RMSE:", RMSE(test_preds, test_targets))
