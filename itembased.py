import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from sortedcontainers import SortedList

# load in the data
import os
if not os.path.exists('venv/include/user2category.json') or \
   not os.path.exists('venv/include/category2user.json') or \
   not os.path.exists('venv/include/usercategory2rating.json'):
   import preprocess2dict


with open('venv/include/user2category.json', 'rb') as f:
  user2category = pickle.load(f)

with open('venv/include/category2user.json', 'rb') as f:
  category2user = pickle.load(f)

with open('venv/include/usercategory2rating.json', 'rb') as f:
  usercategory2rating = pickle.load(f)



N = np.max(list(user2category.keys())) + 1
# the test set may contain movies the train set doesn't have data on
M = np.max(list(category2user.keys())) +1
print("N:", N, "M:", M)

if M > 251:
  print("N =", N, "are you sure you want to continue?")
  print("Comment out these lines if so...")
  exit()


# to find the user similarities, you have to do O(M^2 * N) calculations!
# in the "real-world" you'd want to parallelize this
# note: we really only have to do half the calculations, since w_ij is symmetric
K = 20 # number of neighbors we'd like to consider
limit = 5 # number of common movies users must have in common in order to consider
neighbors = [] # store neighbors in this list
averages = [] # each item's average rating for later use
deviations = [] # each item's deviation for later use

for i in range(M):
  # find the K closest items to item i
  users_i = category2user[i]
  users_i_set = set(users_i)

  # calculate avg and deviation
  ratings_i = { user:usercategory2rating[(user, i)] for user in users_i }
  avg_i = np.mean(list(ratings_i.values()))
  dev_i = { user:(rating - avg_i) for user, rating in ratings_i.items() }
  dev_i_values = np.array(list(dev_i.values()))
  sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

  # save these for later use
  averages.append(avg_i)
  deviations.append(dev_i)

  sl = SortedList()
  for j in range(M):
    # don't include yourself
    if j != i:
      users_j = category2user[j]
      users_j_set = set(users_j)
      common_users = (users_i_set & users_j_set) # intersection
      if len(common_users) > limit:
        # calculate avg and deviation
        ratings_j = { user:usercategory2rating[(user, j)] for user in users_j }
        avg_j = np.mean(list(ratings_j.values()))
        dev_j = { user:(rating - avg_j) for user, rating in ratings_j.items() }
        dev_j_values = np.array(list(dev_j.values()))
        sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

        # calculate correlation coefficient
        numerator = sum(dev_i[m]*dev_j[m] for m in common_users)
        w_ij = numerator / (sigma_i * sigma_j)

        # insert into sorted list and truncate
        # negate weight, because list is sorted ascending
        # maximum value (1) is "closest"
        sl.add((-w_ij, j))
        if len(sl) > K:
          del sl[-1]

  # store the neighbors
  neighbors.append(sl)

  # print out useful things
  if i % 1 == 0:
    print(i)



# using neighbors, calculate train and test MSE

def predict(i, u):
  # calculate the weighted sum of deviations
  numerator = 0
  denominator = 0
  for neg_w, j in neighbors[i]:
    # remember, the weight is stored as its negative
    # so the negative of the negative weight is the positive weight
    try:
      numerator += -neg_w * deviations[j][u]
      denominator += abs(neg_w)
    except KeyError:
      # neighbor may not have been rated by the same user
      # don't want to do dictionary lookup twice
      # so just throw exception
      pass

  if denominator == 0:
    prediction = averages[i]
  else:
    prediction = numerator / denominator + averages[i]
  return prediction



predictionArray = []
train_predictions = []
train_targets = []
for (u, m), target in usercategory2rating.items():
    # calculate the prediction for this movie
    prediction = predict(i, m)
    predictionArray.append([i, m, prediction])
    print("prediction", prediction, i, m)
    # save the prediction and target
    if (target == 0):
        train_predictions.append(prediction)
        train_targets.append(target)


# calculate accuracy
def mse(p, t):
  p = np.array(p)
  t = np.array(t)
  return np.mean((p - t)**2)

print('train mse:', mse(train_predictions, train_targets)),
print(len(train_predictions),len(train_targets))
