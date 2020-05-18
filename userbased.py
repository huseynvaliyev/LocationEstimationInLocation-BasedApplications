import pickle
import numpy as np
from sortedcontainers import SortedList

import os
if not os.path.exists('venv/include/user2categorytokyo.json') or \
   not os.path.exists('venv/include/category2usertokyo.json') or \
   not os.path.exists('venv/include/usercategory2ratingtokyo.json'):
   import preprocess2dict

with open('venv/include/user2categorytokyo.json', 'rb') as f:
  user2category = pickle.load(f)

with open('venv/include/category2usertokyo.json', 'rb') as f:
  category2user = pickle.load(f)

with open('venv/include/usercategory2ratingtokyo.json', 'rb') as f:
  usercategory2rating = pickle.load(f)


N = np.max(list(user2category.keys())) + 1
M = np.max(list(category2user.keys())) +1

if N > 1083:
  print("N =", N, "are you sure you want to continue?")
  print("Comment out these lines if so...")
  exit()

K = 10 # number of neighbors we'd like to consider
limit = 5 # number of common category users must have in common in order to consider
neighbors = [] # store neighbors in this list
averages = [] # each user's average rating for later use
deviations = [] # each user's deviation for later use

for i in range(N):
  # find the 10 closest users to user i
  category_i = user2category[i]
  category_i_set = set(category_i)

  # calculate avg and deviation
  ratings_i = { category:usercategory2rating[(i, category)] for category in category_i }
  avg_i = np.mean(list(ratings_i.values()))
  dev_i = { category:(rating - avg_i) for category, rating in ratings_i.items() }
  dev_i_values = np.array(list(dev_i.values()))
  sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

  # save these for later use
  averages.append(avg_i)
  deviations.append(dev_i)

  sl = SortedList()
  for j in range(N):
    # don't include yourself
    if j != i:
      category_j = user2category[j]
      category_j_set = set(category_j)
      common_category = (category_i_set & category_j_set) # intersection
      if len(common_category) > limit:
        # calculate avg and deviation
        ratings_j = { category:usercategory2rating[(j, category)] for category in category_j }
        avg_j = np.mean(list(ratings_j.values()))
        dev_j = { category:(rating - avg_j) for category, rating in ratings_j.items() }
        dev_j_values = np.array(list(dev_j.values()))
        sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

        # calculate correlation coefficient
        numerator = sum(dev_i[m]*dev_j[m] for m in common_category)
        w_ij = numerator / (sigma_i * sigma_j)

        # insert into sorted list and truncate
        # negate weight, because list is sorted ascending
        # maximum value (1) is "closest"
        sl.add((-w_ij, j))
        if len(sl) > K:
          del sl[-1]
  neighbors.append(sl)
  if i % 1 == 0:
    print(i)


def predict(i, m):
  # calculate the weighted sum of deviations
  numerator = 0
  denominator = 0
  for neg_w, j in neighbors[i]:
    try:
      numerator += -neg_w * deviations[j][m]
      denominator += abs(neg_w)
    except KeyError:
      pass

  if denominator == 0:
    prediction = averages[i]
  else:
    prediction = numerator / denominator + averages[i]
  return prediction

predictionArray = []
train_predictions = []
train_targets = []
for (i, m), target in usercategory2rating.items():
  # calculate the prediction for this movie
  prediction = predict(i, m)
  predictionArray.append([i,m,prediction])
  print("prediction",prediction,i,m)
  # save the prediction and target
  if(target == 0):
    train_predictions.append(prediction)
    train_targets.append(target)


def mse(p, t):
  p = np.array(p)
  t = np.array(t)
  return np.mean((p - t)**2)

print('train mse:', mse(train_predictions, train_targets))

print(len(train_predictions),len(train_targets))

