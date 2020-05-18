import pickle
import pandas as pd
from sklearn.utils import shuffle

df = pd.read_csv('venv/include/tokyoIdNew.csv')

N = df.userId.max() + 1 # number of users
M = df.category_id.max() + 1 # number of category

df = shuffle(df)

user2category = {}
category2user = {}
usercategory2rating = {}
count = 0
def update_user2category_and_category2user(row):
  i = int(row.userId)
  j = int(row.category_id)
  if i not in user2category:
    user2category[i] = [j]
  else:
    user2category[i].append(j)

  if j not in category2user:
    category2user[j] = [i]
  else:
    category2user[j].append(i)

  usercategory2rating[(i,j)] = row.rating
df.apply(update_user2category_and_category2user, axis=1)

# note: these are not really JSONs
with open('venv/include/user2categorytokyo.json', 'wb') as f:
  pickle.dump(user2category, f)

with open('venv/include/category2usertokyo.json', 'wb') as f:
  pickle.dump(category2user, f)

with open('venv/include/usercategory2ratingtokyo.json', 'wb') as f:
  pickle.dump(usercategory2rating, f)


